"""DAG queue. A worker takes a run only after every dependency has result.json."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from src.paths import ROOT
from src.serialize import atomic_json, read_json


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def claim(lock_dir: Path, run_id: str) -> bool:
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{run_id}.lock"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or "0")
            except ValueError:
                pid = 0
            if pid and _alive(pid):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    return False


def release(lock_dir: Path, run_id: str) -> None:
    path = lock_dir / f"{run_id}.lock"
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def hydra_args(run: dict) -> list[str]:
    args = []
    for key, value in run["overrides"].items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        args.append(f"{key}={value}")
    args.extend(["hydra.output_subdir=null", "hydra.run.dir=.", "hydra.job.chdir=false"])
    return args


def subprocess_executor(run: dict) -> None:
    out = ROOT / "outputs" / run["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"
    env = os.environ.copy()
    # Fastest-first order on this machine: 0 is the 5070 Ti, 1 is the 2070.
    # nvidia-smi PCI order is the reverse, so do not use nvidia-smi index 1 here.
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.pop("PRUNE_DEVICE", None)
    env.pop("INJECT_SLEEP_EPOCH", None)
    cmd = [sys.executable, "-m", "src.main", *hydra_args(run)]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"{run['run_id']} exited {proc.returncode}. See {log_path}")


def _done(output_root: Path, run_id: str) -> bool:
    return (output_root / run_id / "result.json").exists()


def run_queue(queue_path: Path, k: int = 1, executor=None, output_root: Path | None = None) -> None:
    executor = executor or subprocess_executor
    output_root = output_root or (ROOT / "outputs")
    lock_dir = queue_path.parent / "locks" / queue_path.stem
    attempts: dict[str, int] = {}
    failed: set[str] = set()
    lock = threading.Lock()

    def wrapped(run):
        executor(run)
        if not _done(output_root, run["run_id"]):
            raise RuntimeError(f"{run['run_id']} finished without result.json")

    with ThreadPoolExecutor(max_workers=max(1, k)) as pool:
        inflight = {}
        while True:
            queue = read_json(queue_path)
            runs = queue["runs"]
            enabled = [r for r in runs if r.get("enabled", True) and r["run_id"] not in failed]
            pending = [
                r
                for r in enabled
                if not _done(output_root, r["run_id"]) and r["run_id"] not in {v["run_id"] for v in inflight.values()}
            ]
            if not pending and not inflight:
                blocked = [r["run_id"] for r in enabled if not _done(output_root, r["run_id"])]
                if blocked:
                    raise RuntimeError(f"queue blocked on {blocked}")
                return
            ready = []
            for run in pending:
                if all(_done(output_root, dep) for dep in run["deps"]):
                    ready.append(run)
            ready.sort(key=lambda r: (r["priority"], -int(r["epochs"]), r["run_id"]))
            while ready and len(inflight) < k:
                run = ready.pop(0)
                if not claim(lock_dir, run["run_id"]):
                    continue
                if _done(output_root, run["run_id"]):
                    release(lock_dir, run["run_id"])
                    continue
                future = pool.submit(wrapped, run)
                inflight[future] = run
            if not inflight:
                time.sleep(0.2)
                continue
            finished, _ = wait(inflight, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in finished:
                run = inflight.pop(future)
                release(lock_dir, run["run_id"])
                try:
                    future.result()
                except Exception as exc:
                    with lock:
                        attempts[run["run_id"]] = attempts.get(run["run_id"], 0) + 1
                        n = attempts[run["run_id"]]
                    print(f"FAIL {run['run_id']} attempt {n}: {exc}", flush=True)
                    if n >= 2:
                        failed.add(run["run_id"])
                        atomic_json(
                            output_root / run["run_id"] / "failed.json",
                            {"run_id": run["run_id"], "error": str(exc)},
                        )
                        raise RuntimeError(f"{run['run_id']} failed twice") from exc


def write_queue(path: Path, runs: list[dict], enabled: dict[str, bool] | None = None) -> None:
    payload = []
    for run in runs:
        item = {
            "run_id": run["run_id"],
            "enabled": True if enabled is None else bool(enabled[run["run_id"]]),
            "deps": list(run["deps"]),
            "priority": run["priority"],
            "epochs": run["epochs"],
            "group": run["group"],
            "cut_class": run["cut_class"],
            "overrides": run["overrides"],
        }
        payload.append(item)
    atomic_json(path, {"runs": payload})
