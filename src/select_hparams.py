"""Choose lambda, fine-tune LR and precision from the dev split. Never reads the test set."""

from __future__ import annotations

import csv
from pathlib import Path

from src.hparams import PAPER_ACC, write_frozen
from src.paths import FROZEN_HPARAMS, OUTPUT_ROOT, ROOT
from src.serialize import read_json


def _val(run_id: str) -> float:
    payload = read_json(OUTPUT_ROOT / run_id / "result.json")
    if payload.get("val_acc") is None:
        raise SystemExit(f"{run_id} has no val_acc")
    return float(payload["val_acc"])


def _hit(run_id: str) -> bool:
    return bool(read_json(OUTPUT_ROOT / run_id / "prune.json")["hit_min1"])


def choose_lambda(plain: float, candidates: list[tuple[float, float, bool]]) -> tuple[float, str]:
    """candidates: (lambda, val_acc, hit_min1). Returns (lambda, tag)."""
    ok = [lam for lam, acc, hit in candidates if acc >= plain - 0.5 and not hit]
    if not ok:
        return 1e-5, "l1e-5"
    best = max(ok)
    tag = "l1e-4" if abs(best - 1e-4) < 1e-12 else "l5e-5"
    return best, tag


def choose_lr(mean_0001: float, mean_001: float) -> float:
    if mean_001 >= mean_0001 + 0.2:
        return 0.01
    return 0.001


def choose_precision(bf16: float, fp32: float) -> str:
    if abs(bf16 - fp32) > 0.3:
        return "fp32"
    return "bf16"


def build_summary(out_csv: Path | None = None) -> dict:
    vgg_plain = _val("dev-V19.BASE.s0")
    res_plain = _val("dev-R56.BASE.s0")
    vgg_cands = []
    res_cands = []
    for lam, tag in ((5e-5, "l5e-5"), (1e-4, "l1e-4")):
        vgg_cands.append((lam, _val(f"dev-V19.BASES.{tag}.s0"), _hit(f"dev-SL-V19-70.PRUNE.{tag}.s0")))
        res_cands.append((lam, _val(f"dev-R56.BASES.{tag}.s0"), _hit(f"dev-SL-R56-80.PRUNE.{tag}.s0")))
    lam_vgg, tag_vgg = choose_lambda(vgg_plain, vgg_cands)
    lam_res, tag_res = choose_lambda(res_plain, res_cands)
    lr_rows = {
        "l1_0001": _val("dev-L1-R56B.FT.lr0001.s0"),
        "l1_001": _val("dev-L1-R56B.FT.lr001.s0"),
        "sl_0001": _val("dev-SL-V19-70.FT.lr0001.s0"),
        "sl_001": _val("dev-SL-V19-70.FT.lr001.s0"),
    }
    lr_ft = choose_lr(
        (lr_rows["l1_0001"] + lr_rows["sl_0001"]) / 2,
        (lr_rows["l1_001"] + lr_rows["sl_001"]) / 2,
    )
    bf16 = _val("dev-V16.BASE.s0")
    fp32 = _val("dev-V16.BASE.fp32.s0")
    precision = choose_precision(bf16, fp32)
    rows = [
        ("dev-V16.BASE.s0", "vgg16", "cifar10", bf16, PAPER_ACC[("vgg16", "cifar10")]),
        ("dev-V19.BASE.s0", "vgg19", "cifar10", vgg_plain, PAPER_ACC[("vgg19", "cifar10")]),
        ("dev-R56.BASE.s0", "resnet56", "cifar10", res_plain, PAPER_ACC[("resnet56", "cifar10")]),
        ("dev-V19C100.BASE.s0", "vgg19", "cifar100", _val("dev-V19C100.BASE.s0"), PAPER_ACC[("vgg19", "cifar100")]),
        ("dev-V16.BASE.fp32.s0", "vgg16", "cifar10", fp32, PAPER_ACC[("vgg16", "cifar10")]),
    ]
    out_csv = out_csv or (ROOT / "dev" / "summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "model", "dataset", "val_acc", "paper", "gap"])
        for run_id, model, dataset, acc, paper in rows:
            writer.writerow([run_id, model, dataset, f"{acc:.4f}", paper, f"{acc - paper:.4f}"])
        writer.writerow([])
        writer.writerow(["lambda_vgg", lam_vgg, "tag", tag_vgg])
        writer.writerow(["lambda_resnet", lam_res, "tag", tag_res])
        writer.writerow(["lr_ft", lr_ft])
        writer.writerow(["precision", precision, "bf16_minus_fp32", f"{bf16 - fp32:.4f}"])
    payload = write_frozen(FROZEN_HPARAMS, precision, lr_ft, lam_vgg, lam_res)
    payload["tag_vgg"] = tag_vgg
    payload["tag_resnet"] = tag_res
    return payload


if __name__ == "__main__":
    import json

    print(json.dumps(build_summary(), indent=2))
