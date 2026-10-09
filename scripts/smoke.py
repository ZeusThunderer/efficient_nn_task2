"""Gate G1. Measures seconds/epoch and parallel throughput. Writes budget/f0.json."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import VGG16_WIDTHS, build_model  # noqa: E402
from src.paths import BUDGET_ROOT, OUTPUT_ROOT  # noqa: E402
from src.queue import hydra_args  # noqa: E402
from src.serialize import atomic_json  # noqa: E402


def _job(run_id, model, epochs=3, dep="none", mode="BASE", method="none", cfg=None):
    return {
        "run_id": run_id,
        "overrides": {
            "run_id": run_id,
            "mode": mode,
            "model": model,
            "dataset": "cifar10",
            "cfg_name": cfg or run_id,
            "method": method,
            "seed": 0,
            "split": "trainonly",
            "precision": "bf16",
            "lr_schedule": "stepwise",
            "lr": 0.1,
            "epochs": epochs,
            "dep_id": dep,
            "kappa_max": 2.0,
            "slim_lambda": 0.0,
            "tracking": False,
            "do_eval": False,
            "force_epochs": True,
            "hparams_hash": "dev",
            "group": "smoke",
        },
    }


def _launch(job, env):
    cmd = [sys.executable, "-m", "src.main", *hydra_args(job)]
    log = OUTPUT_ROOT / job["run_id"] / "train.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return proc, handle


def _median_epoch(run_id: str) -> float:
    result = json.loads((OUTPUT_ROOT / run_id / "result.json").read_text(encoding="utf-8"))
    times = result["sec_per_epoch"]
    tail = times[1:] if len(times) > 1 else times
    return float(statistics.median(tail))


def _gpu_mem_mb() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(out.strip().splitlines()[0])


def main() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PRUNE_DEVICE", None)
    jobs = [
        _job("smoke-V16.BASE.s0", "vgg16", cfg="V16"),
        _job("smoke-V19.BASE.s0", "vgg19", cfg="V19"),
        _job("smoke-R56.BASE.s0", "resnet56", cfg="R56"),
    ]
    # Pruned architectures: widths are fixed by the L1 ratios, so no trained base is required.
    pruned_v = list(VGG16_WIDTHS)
    pruned_v[0] = 32
    for i in range(7, 13):
        pruned_v[i] = 256
    arch_dir = OUTPUT_ROOT / "smoke-arch-V16A"
    arch_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(arch_dir / "arch.json", {"kind": "vgg16", "widths": pruned_v, "num_classes": 10})
    atomic_json(arch_dir / "prune.json", {"E_B": 3, "E_B_uncapped": 3, "kappa_capped": 1, "kappa_raw": 1, "params": 1, "base_run_id": "none"})
    jobs.append(_job("smoke-V16A.SE.s0", "vgg16", dep="smoke-arch-V16A", mode="SE", method="l1", cfg="L1-V16A"))

    serial = {}
    for job in jobs:
        if (OUTPUT_ROOT / job["run_id"] / "result.json").exists():
            serial[job["run_id"]] = _median_epoch(job["run_id"])
            print(f"have {job['run_id']} {serial[job['run_id']]:.2f}s", flush=True)
            continue
        proc, handle = _launch(job, env)
        proc.wait()
        handle.close()
        if proc.returncode != 0:
            raise SystemExit(f"{job['run_id']} failed, see train.log")
        serial[job["run_id"]] = _median_epoch(job["run_id"])
        print(f"timed {job['run_id']} {serial[job['run_id']]:.2f}s/epoch", flush=True)

    # Parallel K=2: one VGG and one ResNet, 3 epochs, separate run ids.
    par_jobs = [
        _job("smoke-par-V16.BASE.s0", "vgg16", cfg="V16"),
        _job("smoke-par-R56.BASE.s0", "resnet56", cfg="R56"),
    ]
    peak = _gpu_mem_mb()
    t0 = time.perf_counter()
    procs = []
    if all((OUTPUT_ROOT / j["run_id"] / "result.json").exists() for j in par_jobs):
        wall = None
    else:
        procs = [_launch(job, env) for job in par_jobs]
        while any(p.poll() is None for p, _ in procs):
            peak = max(peak, _gpu_mem_mb())
            time.sleep(1.0)
        wall = time.perf_counter() - t0
        for proc, handle in procs:
            handle.close()
            if proc.returncode != 0:
                print("parallel smoke failed; keeping K=1", flush=True)
                wall = None
                break
    if wall is None:
        # Already measured; recover phi from a previous f0 if present.
        prev = BUDGET_ROOT / "f0.json"
        wall = json.loads(prev.read_text())["parallel_wall_s"] if prev.exists() else None
    serial_sum = (serial["smoke-V16.BASE.s0"] + serial["smoke-R56.BASE.s0"]) * 3
    phi = 1.0
    k = 1
    if wall and wall > 0 and peak <= 14000:
        phi = max(1.0, serial_sum / wall)
        # Two jobs fit in memory, but K=2 is only useful when they actually overlap.
        k = 2 if phi >= 1.15 else 1
    t_ep = {
        "vgg16": serial["smoke-V16.BASE.s0"],
        "vgg19": serial["smoke-V19.BASE.s0"],
        "resnet56": serial["smoke-R56.BASE.s0"],
    }
    payload = {
        "t_ep": t_ep,
        "t_ep_vgg16_pruned": serial["smoke-V16A.SE.s0"],
        "phi": phi,
        "k": k,
        "peak_mem_mb": peak,
        "parallel_wall_s": wall,
        "serial_epoch_s": serial,
    }
    atomic_json(BUDGET_ROOT / "f0.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
