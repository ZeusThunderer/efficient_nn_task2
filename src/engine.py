"""Single-run executor: base, prune, fine-tune, scratch-E/B, uniform, guided, winning ticket."""

from __future__ import annotations

import json
import os
import platform
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import (
    augment_batch,
    dataset_stats,
    epoch_generator,
    load_cifar,
    normalize_batch,
    num_classes,
    stratified_split,
)
from src.device import get_device
from src.flops import analytical_resnet56_macs, analytical_vgg_macs, count_macs, scratch_b_epochs
from src.hparams import load_frozen, slim_lambda_for
from src.initutil import fingerprint, init_module, init_seed_for
from src.models import (
    DEFAULT_WIDTHS,
    arch_from_model,
    build_model,
    count_params,
    default_resnet_internals,
    model_from_arch,
)
from src.paths import DATA_ROOT, OUTPUT_ROOT, ROOT
from src.prune import (
    add_slimming_grad_,
    apply_grad_mask_,
    apply_weight_mask_,
    conv_nnz,
    gamma_small_fraction,
    guided_widths,
    keeps_from_json,
    keeps_to_json,
    l1_keeps,
    layer_densities,
    slice_model,
    slimming_keeps,
    uniform_internals,
    uniform_widths,
    unstructured_masks,
)
from src.serialize import atomic_json, atomic_torch_save, read_json
from src.tracking import Tracker

L1_RATIOS = {
    "L1-V16A": {i: (0.5 if i == 0 or i >= 7 else 0.0) for i in range(13)},
    "L1-R56A": [0.3] * 27,
    "L1-R56B": [0.3] * 9 + [0.6] * 18,
}
SLIM_RATIO = {
    "SL-V19-30": 0.30,
    "SL-V19-50": 0.50,
    "SL-V19-70": 0.70,
    "SL-R56-40": 0.40,
    "SL-R56-60": 0.60,
    "SL-R56-80": 0.80,
    "SL-V19-50-C100": 0.50,
}
UN_RATIO = {
    "UN-V19-80": 0.80,
    "UN-V19-95": 0.95,
    "UN-R56-80": 0.80,
    "UN-R56-95": 0.95,
}


def lr_milestones(epochs: int, schedule: str) -> list[int]:
    if schedule == "constant":
        return []
    return [epochs // 2, int(round(0.75 * epochs))]


def lr_at(epoch: int, epochs: int, base_lr: float, schedule: str) -> float:
    if schedule == "constant":
        return base_lr
    half, late = lr_milestones(epochs, schedule)
    if epoch >= late:
        return base_lr * 0.01
    if epoch >= half:
        return base_lr * 0.1
    return base_lr


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _checkpoint(path: Path, model, optimizer, epoch: int) -> None:
    payload = {
        "epoch": epoch,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optim": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        except Exception:  # noqa: BLE001
            pass
    atomic_torch_save(path, payload)


def _restore_rng(payload: dict) -> None:
    torch.random.set_rng_state(payload["torch_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    if "cuda_rng" in payload and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        except Exception:  # noqa: BLE001
            pass


def evaluate(model, images, labels, mean, std) -> float:
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()
    channels_last = images.device.type == "cuda"
    correct = 0
    total = 0
    try:
        with torch.inference_mode():
            for start in range(0, images.shape[0], 256):
                xb = normalize_batch(images[start : start + 256], mean, std)
                yb = labels[start : start + 256]
                if channels_last:
                    xb = xb.contiguous(memory_format=torch.channels_last)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.shape[0])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    return 100.0 * correct / total


def fit(
    model,
    train_images,
    train_labels,
    *,
    epochs: int,
    lr: float,
    lr_schedule: str,
    out_dir: Path,
    mean,
    std,
    data_seed: int,
    slim_bns=None,
    slim_lambda: float = 0.0,
    masks: dict | None = None,
    eval_images=None,
    eval_labels=None,
    eval_name: str | None = None,
    checkpoint_every: int = 5,
    tracker: Tracker | None = None,
    meta: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = train_images.device
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model.to(device)
    if masks is not None:
        masks = {name: mask.to(device) for name, mask in masks.items()}
        apply_weight_mask_(model, masks)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    start_epoch = 0
    ckpt_path = out_dir / "ckpt_last.pt"
    history_path = out_dir / "history.jsonl"
    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optim"])
        _restore_rng(payload)
        start_epoch = int(payload["epoch"])
        if history_path.exists():
            lines = history_path.read_text(encoding="utf-8").splitlines()
            history_path.write_text("\n".join(lines[:start_epoch]) + ("\n" if start_epoch else ""), encoding="utf-8")

    nnz_start = conv_nnz(model) if masks is not None else None
    use_amp = device.type == "cuda" and (meta or {}).get("precision") == "bf16"
    epoch_times = []
    if history_path.exists() and start_epoch:
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                epoch_times.append(json_time(line))

    n = train_images.shape[0]
    for epoch in range(start_epoch, epochs):
        sleep_at = os.environ.get("INJECT_SLEEP_EPOCH")
        if sleep_at is not None and int(sleep_at) == epoch:
            time.sleep(float(os.environ.get("INJECT_SLEEP_S", "30")))
        model.train()
        cur_lr = lr_at(epoch, epochs, lr, lr_schedule)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        generator = epoch_generator(data_seed, epoch)
        perm = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        correct = 0
        seen = 0
        t0 = time.perf_counter()
        for start in range(0, n, 128):
            index = perm[start : start + 128].to(device)
            xb = augment_batch(train_images.index_select(0, index), generator, mean, std)
            yb = train_labels.index_select(0, index)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_amp
                else nullcontext()
            )
            with amp:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
            loss.backward()
            if slim_bns and slim_lambda:
                add_slimming_grad_(slim_bns, slim_lambda)
            if masks is not None:
                apply_grad_mask_(model, masks)
            optimizer.step()
            if masks is not None:
                apply_weight_mask_(model, masks)
            bs = int(yb.shape[0])
            loss_sum += float(loss.detach()) * bs
            correct += int((logits.detach().argmax(1) == yb).sum().item())
            seen += bs
        elapsed = time.perf_counter() - t0
        epoch_times.append(elapsed)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_acc": 100.0 * correct / seen,
            "lr": cur_lr,
            "sec": elapsed,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        if tracker is not None:
            tracker.track("train_loss", row["train_loss"], epoch)
            tracker.track("train_acc", row["train_acc"], epoch)
            tracker.track("lr", cur_lr, epoch)
        print(
            f"{(meta or {}).get('run_id', out_dir.name)} epoch {epoch + 1}/{epochs} "
            f"loss {row['train_loss']:.4f} acc {row['train_acc']:.2f} lr {cur_lr:g} {elapsed:.2f}s",
            flush=True,
        )
        done = epoch + 1
        if done % checkpoint_every == 0 or done == epochs:
            _checkpoint(ckpt_path, model, optimizer, done)

    eval_acc = None
    n_test_evals = 0
    if eval_images is not None and eval_name == "test":
        eval_acc = evaluate(model, eval_images, eval_labels, mean, std)
        n_test_evals = 1
        if tracker is not None:
            tracker.track("test_acc", eval_acc, epochs - 1)
    elif eval_images is not None and eval_name == "val":
        eval_acc = evaluate(model, eval_images, eval_labels, mean, std)

    atomic_torch_save(out_dir / "ckpt_final.pt", {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}})
    result = {
        "epochs_planned": epochs,
        "epochs_done": epochs,
        "milestones": lr_milestones(epochs, lr_schedule),
        "lr_schedule": lr_schedule,
        "n_test_evals": n_test_evals,
        "test_acc_last": eval_acc if eval_name == "test" else None,
        "val_acc": eval_acc if eval_name == "val" else None,
        "sec_per_epoch": epoch_times,
        "sec_per_epoch_median": statistics.median(epoch_times) if epoch_times else None,
        "train_loss_last": json_last_loss(history_path),
        "nnz_start": nnz_start,
        "nnz_end": conv_nnz(model) if masks is not None else None,
        "init_fingerprint": (meta or {}).get("init_fingerprint"),
        "params": count_params(model),
        "wall_s": sum(epoch_times),
    }
    if masks is not None:
        # Report nonzero conv weights plus the untouched parameters.
        conv_numel = sum(int(mask.numel()) for mask in masks.values())
        nnz = int(sum(int(mask.sum().item()) for mask in masks.values()))
        result["params"] = count_params(model) - conv_numel + nnz
    return result


def json_time(line: str) -> float:
    return float(json.loads(line)["sec"])


def json_last_loss(path: Path) -> float | None:
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    return float(json.loads(lines[-1])["train_loss"])


def _run_dir(output_root: Path, run_id: str) -> Path:
    return output_root / run_id


def _load_state(path: Path, model):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    return model


def _splits(dataset: str, split: str, device, data_root: Path):
    (train_x, train_y), (test_x, test_y) = load_cifar(data_root, dataset, device)
    mean, std = dataset_stats(dataset)
    if split == "dev":
        tr_idx, va_idx = stratified_split(train_y, seed=0)
        tr_idx = tr_idx.to(device)
        va_idx = va_idx.to(device)
        return (
            train_x.index_select(0, tr_idx),
            train_y.index_select(0, tr_idx),
            train_x.index_select(0, va_idx),
            train_y.index_select(0, va_idx),
            "val",
            mean,
            std,
        )
    if split == "trainonly":
        return train_x, train_y, None, None, None, mean, std
    return train_x, train_y, test_x, test_y, "test", mean, std


def _check_frozen(job: dict) -> None:
    if job["split"] == "trainonly" or job.get("hparams_hash") in (None, "", "dev"):
        if job["split"] == "full":
            raise SystemExit(f"{job['run_id']}: full split requires a frozen hparams hash")
        return
    frozen = load_frozen()
    if frozen is None or frozen.get("hash") != job["hparams_hash"]:
        raise SystemExit(f"{job['run_id']}: hparams hash does not match {job['hparams_hash']}")
    if job["precision"] != frozen["precision"]:
        raise SystemExit(f"{job['run_id']}: precision {job['precision']} != frozen {frozen['precision']}")
    if job["mode"] == "FT" and abs(float(job["lr"]) - float(frozen["lr_ft"])) > 1e-12:
        raise SystemExit(f"{job['run_id']}: FT lr does not match frozen lr_ft")
    if job["mode"] == "BASES":
        expected = slim_lambda_for(frozen, job["model"])
        if abs(float(job["slim_lambda"]) - expected) > 1e-12:
            raise SystemExit(f"{job['run_id']}: slim_lambda {job['slim_lambda']} != frozen {expected}")


def _macs_pair(full_model, pruned_model=None, masks=None) -> tuple[int, int]:
    f_l = count_macs(full_model)
    if masks is not None:
        f_p = count_macs(full_model, conv_masks=masks)
    else:
        f_p = count_macs(pruned_model)
    return f_l, f_p


def _write_prune(out: Path, payload: dict, arch: dict, state_model, keeps=None, masks=None) -> None:
    atomic_json(out / "arch.json", arch)
    atomic_json(out / "prune.json", payload)
    atomic_torch_save(out / "pruned_state.pt", {"model": {k: v.detach().cpu() for k, v in state_model.state_dict().items()}})
    if keeps is not None:
        atomic_json(out / "keeps.json", {"keeps": keeps_to_json(keeps)})
    if masks is not None:
        atomic_torch_save(out / "mask.pt", {k: v.cpu() for k, v in masks.items()})


def execute(job: dict) -> None:
    output_root = Path(job.get("output_root") or OUTPUT_ROOT)
    data_root = Path(job.get("data_root") or DATA_ROOT)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if not data_root.is_absolute():
        data_root = ROOT / data_root
    out = _run_dir(output_root, job["run_id"])
    if (out / "result.json").exists():
        print(f"skip {job['run_id']}", flush=True)
        return
    _check_frozen(job)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", job)
    device = get_device()
    mode = job["mode"]
    classes = num_classes(job["dataset"])
    dep = job.get("dep_id") or ""
    dep_dir = _run_dir(output_root, dep) if dep and dep != "none" else None

    if mode == "PRUNE":
        _execute_prune(job, out, dep_dir, device, data_root, classes)
        return

    train_x, train_y, eval_x, eval_y, eval_name, mean, std = _splits(job["dataset"], job["split"], device, data_root)
    if not job.get("do_eval", True):
        eval_x, eval_y, eval_name = None, None, None

    model, masks, slim_bns, slim_lambda, epochs, lr, schedule = _prepare_model(
        job, dep_dir, output_root, device, classes
    )
    init_fp = job.get("_init_fingerprint") or fingerprint(model)
    if mode in ("BASE", "BASES"):
        init_path = out / "ckpt_init.pt"
        if not init_path.exists():
            atomic_torch_save(init_path, {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}})

    tags = {
        "mode": mode,
        "seed": job["seed"],
        "group": job.get("group", ""),
        "cfg": job["cfg_name"],
    }
    tracker = Tracker(
        enabled=bool(job.get("tracking", True)),
        repo=ROOT / ".aim",
        run_id=job["run_id"],
        experiment=job["cfg_name"],
        tags=tags,
        hparams={k: job[k] for k in ("mode", "model", "dataset", "seed", "precision", "lr", "epochs", "slim_lambda") if k in job},
    )
    if mode == "BASES" and tracker.run is not None:
        gammas = torch.cat([bn.weight.detach().float().cpu().reshape(-1) for bn in model.prunable_bns()])
        tracker.track_distribution("gamma", gammas.numpy(), 0)
    meta = {"run_id": job["run_id"], "precision": job["precision"], "init_fingerprint": init_fp}
    try:
        trained = fit(
            model,
            train_x,
            train_y,
            epochs=epochs,
            lr=lr,
            lr_schedule=schedule,
            out_dir=out,
            mean=mean,
            std=std,
            data_seed=int(job["seed"]),
            slim_bns=slim_bns,
            slim_lambda=slim_lambda,
            masks=masks,
            eval_images=eval_x,
            eval_labels=eval_y,
            eval_name=eval_name,
            checkpoint_every=int(os.environ.get("CHECKPOINT_EVERY", "5")),
            tracker=tracker,
            meta=meta,
        )
    finally:
        if mode == "BASES" and tracker.run is not None:
            gammas = torch.cat([bn.weight.detach().float().cpu().reshape(-1) for bn in model.prunable_bns()])
            tracker.track_distribution("gamma", gammas.numpy(), epochs)
        tracker.close()

    arch = arch_from_model(model)
    if masks is None:
        macs = count_macs(model)
    else:
        macs = count_macs(model, conv_masks=masks)
    result = {
        **trained,
        "run_id": job["run_id"],
        "mode": mode,
        "model": job["model"],
        "dataset": job["dataset"],
        "cfg_name": job["cfg_name"],
        "seed": int(job["seed"]),
        "init_seed": init_seed_for(int(job["seed"]), mode),
        "split": job["split"],
        "group": job.get("group", ""),
        "precision": job["precision"],
        "hparams_hash": job.get("hparams_hash", "dev"),
        "macs": macs,
        "arch": arch,
        "gamma_small_fraction": gamma_small_fraction(model) if mode in ("BASE", "BASES") else None,
        "versions": _versions(),
        "init_fingerprint": init_fp,
        "alias": False,
    }
    if "_param_match" in job:
        target = job["_param_match"]["target"]
        got = job["_param_match"]["got"]
        result["param_match_target"] = target
        result["param_match_got"] = got
        result["param_match_rel"] = abs(got - target) / max(target, 1)
    if mode in ("SB", "SBU") and dep_dir is not None:
        prune_meta = read_json(dep_dir / "prune.json")
        result["E_B"] = prune_meta["E_B"] if mode == "SB" else prune_meta["E_B_uncapped"]
        result["kappa"] = prune_meta["kappa_capped"] if mode == "SB" else prune_meta["kappa_raw"]
    atomic_json(out / "result.json", result)


def _prepare_model(job, dep_dir, output_root, device, classes):
    mode = job["mode"]
    seed = int(job["seed"])
    init_seed = init_seed_for(seed, mode)
    epochs = int(job["epochs"])
    lr = float(job["lr"])
    schedule = job["lr_schedule"]
    masks = None
    slim_bns = None
    slim_lambda = 0.0

    if mode in ("BASE", "BASES"):
        model = build_model(job["model"], classes)
        init_module(model, init_seed)
        if mode == "BASES":
            slim_bns = model.prunable_bns()
            slim_lambda = float(job["slim_lambda"])
        job["_init_fingerprint"] = fingerprint(model)
        return model, masks, slim_bns, slim_lambda, epochs, lr, schedule

    if dep_dir is None:
        raise SystemExit(f"{job['run_id']}: mode {mode} needs dep_id")
    arch = read_json(dep_dir / "arch.json")
    prune_meta = read_json(dep_dir / "prune.json")

    if mode in ("SB", "SBU") and not job.get("force_epochs", False):
        epochs = int(prune_meta["E_B"] if mode == "SB" else prune_meta["E_B_uncapped"])

    if mode == "FT":
        model = model_from_arch(arch)
        _load_state(dep_dir / "pruned_state.pt", model)
        if (dep_dir / "mask.pt").exists():
            masks = torch.load(dep_dir / "mask.pt", map_location="cpu", weights_only=False)
        job["_init_fingerprint"] = fingerprint(model)
        return model, masks, None, 0.0, epochs, lr, schedule

    if mode == "WT":
        keeps = keeps_from_json(read_json(dep_dir / "keeps.json")["keeps"])
        base_id = prune_meta["base_run_id"]
        base = build_model(job["model"], classes)
        _load_state(_run_dir(output_root, base_id) / "ckpt_init.pt", base)
        model = slice_model(base, keeps)
        job["_init_fingerprint"] = fingerprint(model)
        return model, None, None, 0.0, epochs, lr, schedule

    if mode in ("UF", "GD"):
        target = int(prune_meta["params"])
        if job["model"] == "resnet56":
            internals, got = uniform_internals(target, classes)
            built = {"kind": "resnet56", "internals": internals, "num_classes": classes}
        elif mode == "UF":
            widths, got = uniform_widths(job["model"], target, classes)
            built = {"kind": job["model"], "widths": widths, "num_classes": classes}
        else:
            widths, got = guided_widths(job["model"], arch["widths"], target, classes)
            built = {"kind": job["model"], "widths": widths, "num_classes": classes}
        model = model_from_arch(built)
        init_module(model, init_seed)
        job["_init_fingerprint"] = fingerprint(model)
        job["_param_match"] = {"target": target, "got": got}
        atomic_json(Path(job.get("output_root") or OUTPUT_ROOT) / job["run_id"] / "arch.json", built)
        return model, None, None, 0.0, epochs, lr, schedule

    # SE / SB / SBU
    model = model_from_arch(arch)
    if (dep_dir / "mask.pt").exists():
        masks = torch.load(dep_dir / "mask.pt", map_location="cpu", weights_only=False)
        init_module(model, init_seed, conv_density=layer_densities(masks))
        apply_weight_mask_(model, masks)
    else:
        init_module(model, init_seed)
    job["_init_fingerprint"] = fingerprint(model)
    return model, masks, None, 0.0, epochs, lr, schedule


def _execute_prune(job, out, dep_dir, device, data_root, classes) -> None:
    if dep_dir is None:
        raise SystemExit(f"{job['run_id']}: prune needs a base run")
    cfg_name = job["cfg_name"]
    full = build_model(job["model"], classes)
    _load_state(dep_dir / "ckpt_final.pt", full)
    full.to(device)
    method = job["method"]
    keeps = None
    masks = None
    hit_min1 = False
    actual_ratio = None
    if method == "l1":
        ratios = L1_RATIOS[cfg_name]
        keeps = l1_keeps(full, ratios)
        pruned = slice_model(full.cpu(), keeps)
        actual_ratio = _channel_ratio(full, keeps)
    elif method == "slimming":
        keeps, hit_min1, actual_ratio = slimming_keeps(full, SLIM_RATIO[cfg_name])
        pruned = slice_model(full.cpu(), keeps)
    elif method == "unstructured":
        masks = unstructured_masks(full, UN_RATIO[cfg_name])
        pruned = full.cpu()
        apply_weight_mask_(pruned, masks)
        actual_ratio = UN_RATIO[cfg_name]
    else:
        raise SystemExit(f"unknown prune method {method}")

    f_l, f_p = _macs_pair(full.cpu() if method != "unstructured" else build_model(job["model"], classes), pruned if method != "unstructured" else None, masks)
    if method != "unstructured":
        # full was moved; recount F_L on a clean copy of the original shapes via analytical path
        f_l = count_macs(build_model(job["model"], classes))
        f_p = count_macs(pruned)
    e_b, kappa_cap = scratch_b_epochs(f_l, f_p, kappa_max=2.0)
    e_raw, kappa_raw = scratch_b_epochs(f_l, f_p, kappa_max=float("inf"))
    if method == "unstructured":
        conv_numel = sum(int(m.numel()) for m in masks.values())
        nnz = int(sum(int(m.sum().item()) for m in masks.values()))
        params = count_params(pruned) - conv_numel + nnz
    else:
        params = count_params(pruned)

    # Diagnostic accuracy. Dev uses val so the test set stays untouched.
    train_x, train_y, eval_x, eval_y, eval_name, mean, std = _splits(job["dataset"], job["split"], device, data_root)
    acc = evaluate(pruned.to(device), eval_x, eval_y, mean, std)
    densities = {}
    if masks is not None:
        densities = {name: float(mask.mean()) for name, mask in masks.items()}
        if any(value < 0.01 for value in densities.values()):
            densities["_flag_layer_below_1pct"] = True

    arch = arch_from_model(pruned.cpu() if method != "unstructured" else build_model(job["model"], classes))
    if method == "unstructured":
        arch = arch_from_model(build_model(job["model"], classes))
    payload = {
        "cfg_name": cfg_name,
        "method": method,
        "base_run_id": job["dep_id"],
        "F_L": f_l,
        "F_P": f_p,
        "kappa_raw": kappa_raw,
        "kappa_capped": kappa_cap,
        "E_B": e_b,
        "E_B_uncapped": e_raw,
        "params": params,
        "params_base": count_params(build_model(job["model"], classes)),
        "hit_min1": hit_min1,
        "acc_after_prune": acc,
        "diagnostic_split": eval_name,
        "actual_prune_ratio": actual_ratio,
        "layer_density": densities,
        "gamma_small_fraction_base": gamma_small_fraction(full.cpu()),
    }
    _write_prune(out, payload, arch, pruned.cpu(), keeps=keeps, masks=masks)
    result = {
        "run_id": job["run_id"],
        "mode": "PRUNE",
        "model": job["model"],
        "dataset": job["dataset"],
        "cfg_name": cfg_name,
        "seed": int(job["seed"]),
        "init_seed": init_seed_for(int(job["seed"]), "PRUNE"),
        "split": job["split"],
        "group": job.get("group", ""),
        "precision": job["precision"],
        "hparams_hash": job.get("hparams_hash", "dev"),
        "epochs_planned": 0,
        "epochs_done": 0,
        "milestones": [],
        "lr_schedule": "none",
        "n_test_evals": 0,
        "test_acc_last": None,
        "val_acc": None,
        "params": params,
        "macs": f_p,
        "E_B": e_b,
        "kappa": kappa_cap,
        "hit_min1": hit_min1,
        "acc_after_prune": acc,
        "versions": _versions(),
        "alias": False,
        "wall_s": 0.0,
    }
    atomic_json(out / "result.json", result)
    print(
        f"{job['run_id']} prune F_P/F_L={f_p / f_l:.4f} E_B={e_b} E_raw={e_raw} "
        f"acc_after={acc:.2f} hit_min1={hit_min1}",
        flush=True,
    )


def _channel_ratio(model, keeps) -> float:
    if hasattr(model, "convs"):
        total = sum(conv.out_channels for conv in model.convs)
    else:
        total = sum(block.bn1.weight.numel() for block in model.blocks)
    kept = sum(int(k.numel()) for k in keeps)
    return 1.0 - kept / total


# Silence unused-import lints for helpers used by tests via this module.
_ = (analytical_vgg_macs, analytical_resnet56_macs, DEFAULT_WIDTHS, default_resnet_internals)
