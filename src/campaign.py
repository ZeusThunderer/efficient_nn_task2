"""Run the dev split, the canary, then the final matrix in three phases."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.budget import C_FIX, allowance_hours, phase_of, select_enabled
from src.catalog import canary_runs, dev_phase_a, dev_phase_b, final_runs
from src.flops import scratch_b_epochs
from src.paths import BUDGET_ROOT, FROZEN_HPARAMS, OUTPUT_ROOT
from src.queue import run_queue, write_queue
from src.select_hparams import _hit, _val, choose_lambda
from src.serialize import atomic_json, read_json


def _k() -> int:
    path = BUDGET_ROOT / "f0.json"
    if not path.exists():
        return 1
    return int(read_json(path).get("k", 1))


def _phi() -> float:
    path = BUDGET_ROOT / "f0.json"
    if not path.exists():
        return 1.0
    return float(read_json(path).get("phi", 1.0))


def _done(run_id: str) -> bool:
    return (OUTPUT_ROOT / run_id / "result.json").exists()


def run_dev() -> dict:
    started = time.time()
    stamp = BUDGET_ROOT / "dev_wall.json"
    phase_a = BUDGET_ROOT / "queue_dev_a.json"
    write_queue(phase_a, dev_phase_a())
    print("dev phase A", flush=True)
    run_queue(phase_a, k=_k())

    vgg_plain = _val("dev-V19.BASE.s0")
    res_plain = _val("dev-R56.BASE.s0")
    vgg_cands, res_cands = [], []
    for lam, tag in ((5e-5, "l5e-5"), (1e-4, "l1e-4")):
        vgg_cands.append((lam, _val(f"dev-V19.BASES.{tag}.s0"), _hit(f"dev-SL-V19-70.PRUNE.{tag}.s0")))
        res_cands.append((lam, _val(f"dev-R56.BASES.{tag}.s0"), _hit(f"dev-SL-R56-80.PRUNE.{tag}.s0")))
    lam_vgg, tag_vgg = choose_lambda(vgg_plain, vgg_cands)
    lam_res, tag_res = choose_lambda(res_plain, res_cands)
    print(f"lambda vgg {lam_vgg} ({tag_vgg}) resnet {lam_res} ({tag_res})", flush=True)

    extra = []
    if tag_vgg == "l1e-5":
        from src.catalog import _job

        extra.append(
            _job(
                "dev-V19.BASES.l1e-5.s0",
                "BASES",
                "vgg19",
                "cifar10",
                "V19",
                "slimming",
                0,
                [],
                "dev",
                "dev",
                160,
                0.1,
                "stepwise",
                "",
                slim_lambda=1e-5,
                split="dev",
                hparams_hash="dev",
            )
        )
        extra.append(
            _job(
                "dev-SL-V19-70.PRUNE.l1e-5.s0",
                "PRUNE",
                "vgg19",
                "cifar10",
                "SL-V19-70",
                "slimming",
                0,
                ["dev-V19.BASES.l1e-5.s0"],
                "dev",
                "dev",
                0,
                0.0,
                "none",
                "dev-V19.BASES.l1e-5.s0",
                slim_lambda=1e-5,
                split="dev",
                hparams_hash="dev",
            )
        )
    if extra:
        path = BUDGET_ROOT / "queue_dev_fallback.json"
        write_queue(path, extra)
        run_queue(path, k=_k())

    phase_b = BUDGET_ROOT / "queue_dev_b.json"
    write_queue(phase_b, dev_phase_b(tag_vgg))
    print("dev phase B (fine-tune LR)", flush=True)
    run_queue(phase_b, k=_k())
    wall = time.time() - started
    atomic_json(stamp, {"wall_s": wall, "wall_h": wall / 3600.0, "tag_vgg": tag_vgg, "tag_resnet": tag_res, "lam_vgg": lam_vgg, "lam_res": lam_res})
    return {"tag_vgg": tag_vgg, "tag_resnet": tag_res}


def run_canary(hparams: dict) -> None:
    runs = canary_runs(hparams)
    path = BUDGET_ROOT / "queue_canary.json"
    write_queue(path, runs)
    print("canary", flush=True)
    run_queue(path, k=_k())
    for run in runs:
        if run["overrides"]["mode"] != "PRUNE":
            continue
        meta = read_json(OUTPUT_ROOT / run["run_id"] / "prune.json")
        epochs, _ = scratch_b_epochs(meta["F_L"], meta["F_P"], kappa_max=2.0)
        if epochs != int(meta["E_B"]):
            raise SystemExit(f"{run['run_id']} E_B {meta['E_B']} != formula {epochs}")
    missing = [r["run_id"] for r in runs if not _done(r["run_id"])]
    if missing:
        raise SystemExit(f"canary missing {missing}")
    print("canary ok", flush=True)


def _measured(runs: list[dict]) -> tuple[dict, dict]:
    from statistics import median

    actual = {}
    samples: dict[str, list[float]] = {}
    sec_by_cfg: dict[tuple[str, int], float] = {}
    for run in runs:
        path = OUTPUT_ROOT / run["run_id"] / "result.json"
        if not path.exists():
            continue
        payload = read_json(path)
        actual[run["run_id"]] = float(payload.get("wall_s") or 0.0) / 3600.0 + (0.0 if payload.get("wall_s") else C_FIX)
        if payload.get("sec_per_epoch_median") and payload["mode"] in {"BASE", "BASES"}:
            samples.setdefault(payload["model"], []).append(float(payload["sec_per_epoch_median"]))
        if payload.get("sec_per_epoch_median") and payload["mode"] in {"SE", "FT", "UF", "GD"}:
            sec_by_cfg[(payload["cfg_name"], int(payload["seed"]))] = float(payload["sec_per_epoch_median"])
    f0_path = BUDGET_ROOT / "f0.json"
    t_ep = {"vgg16": 4.0, "vgg19": 5.0, "resnet56": 7.0}
    if f0_path.exists():
        t_ep.update(read_json(f0_path)["t_ep"])
    for model, values in samples.items():
        t_ep[model] = float(median(values))
    return actual, {"t_ep": t_ep, "sec_by_cfg": sec_by_cfg}


def _refresh_epochs(runs: list[dict]) -> None:
    for run in runs:
        mode = run["overrides"]["mode"]
        if mode not in {"SB", "SBU"} or run["overrides"].get("force_epochs"):
            continue
        dep = run["overrides"]["dep_id"]
        path = OUTPUT_ROOT / dep / "prune.json"
        if not path.exists():
            continue
        meta = read_json(path)
        epochs = int(meta["E_B"] if mode == "SB" else meta["E_B_uncapped"])
        run["epochs"] = epochs
        run["overrides"]["epochs"] = epochs


def _cost_fn(bundle):
    t_ep = bundle["t_ep"]
    sec_by_cfg = bundle["sec_by_cfg"]

    def cost(run: dict) -> float:
        from src.budget import estimate_hours

        cfg = run["overrides"]["cfg_name"]
        seed = int(run["overrides"]["seed"])
        mode = run["overrides"]["mode"]
        if mode in {"SE", "SB", "SBU", "FT", "UF", "GD", "WT"} and (cfg, seed) in sec_by_cfg:
            return run["epochs"] * sec_by_cfg[(cfg, seed)] / 3600.0 + C_FIX
        return estimate_hours(run, t_ep)

    return cost


def run_final(hparams: dict) -> None:
    runs = final_runs(hparams)
    dev_wall = 3.0
    wall_path = BUDGET_ROOT / "dev_wall.json"
    if wall_path.exists():
        dev_wall = float(read_json(wall_path)["wall_h"])
    for phase in (1, 2, 3):
        _refresh_epochs(runs)
        actual, bundle = _measured(runs)
        allowance = allowance_hours(_phi(), t_dev=dev_wall, t_lat=0.5)
        enabled, cost = select_enabled(runs, allowance, bundle["t_ep"], actual, cost_fn=_cost_fn(bundle))
        plan_rows = []
        for run in runs:
            plan_rows.append(
                {
                    "run_id": run["run_id"],
                    "enabled": enabled[run["run_id"]],
                    "phase": phase_of(run),
                    "cut_class": run["cut_class"],
                    "epochs": run["epochs"],
                    "done": _done(run["run_id"]),
                }
            )
        atomic_json(BUDGET_ROOT / f"plan_f{phase - 1}.json", {"allowance_h": allowance, "cost_h": cost, "runs": plan_rows})
        # CSV sibling for the report.
        lines = ["run_id,enabled,phase,cut_class,epochs,done"]
        for row in plan_rows:
            lines.append(",".join(str(row[k]) for k in ("run_id", "enabled", "phase", "cut_class", "epochs", "done")))
        (BUDGET_ROOT / f"plan_f{phase - 1}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        enable_now = {}
        for run in runs:
            if _done(run["run_id"]):
                enable_now[run["run_id"]] = True
            else:
                enable_now[run["run_id"]] = bool(enabled[run["run_id"]] and phase_of(run) == phase)
        queued = sum(1 for run in runs if enable_now[run["run_id"]] and not _done(run["run_id"]))
        print(f"phase {phase}: allowance {allowance:.2f}h planned {cost:.2f}h launching {queued}", flush=True)
        path = BUDGET_ROOT / f"queue_phase{phase}.json"
        write_queue(path, runs, enable_now)
        if queued:
            run_queue(path, k=_k())


def dev_wall_hours() -> float:
    path = BUDGET_ROOT / "dev_wall.json"
    if not path.exists():
        return 3.0
    return float(read_json(path)["wall_h"])


def frozen_or_die() -> dict:
    if not FROZEN_HPARAMS.exists():
        raise SystemExit("conf/frozen_hparams.json missing")
    return json.loads(Path(FROZEN_HPARAMS).read_text(encoding="utf-8"))
