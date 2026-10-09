"""GPU latency after training. FP32 (TF32 off) and FP16, batch 1 and 256."""

from __future__ import annotations

import statistics
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.device import get_device  # noqa: E402
from src.models import model_from_arch  # noqa: E402
from src.paths import OUTPUT_ROOT, ROOT as REPO  # noqa: E402
from src.serialize import atomic_json, read_json  # noqa: E402


def _idle() -> None:
    out = subprocess.check_output(
        ["nvidia-smi", "--id=1", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    )
    pids = [line.strip() for line in out.splitlines() if line.strip() and line.strip() != "[N/A]"]
    if pids:
        raise SystemExit(f"GPU 1 (5070 Ti) still has compute apps: {pids}")


def _targets() -> list[tuple[str, dict, dict]]:
    found = []
    for result_path in sorted(OUTPUT_ROOT.glob("*/result.json")):
        row = read_json(result_path)
        if row.get("split") != "full" or int(row.get("seed", -1)) != 1:
            continue
        if row.get("mode") not in {"BASE", "SE"}:
            continue
        if row.get("method") == "unstructured" or str(row.get("cfg_name", "")).startswith("UN-"):
            continue
        ckpt = result_path.parent / "ckpt_final.pt"
        arch_path = result_path.parent / "arch.json"
        if not ckpt.exists():
            continue
        arch = read_json(arch_path) if arch_path.exists() else row.get("arch")
        if not arch:
            continue
        found.append((row["run_id"], arch, torch.load(ckpt, map_location="cpu", weights_only=False)["model"]))
    return found


def _time_once(model, batch: torch.Tensor, warmup: int, measure: int) -> float:
    for _ in range(warmup):
        model(batch)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(measure):
        start.record()
        model(batch)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return float(statistics.median(samples))


def main() -> None:
    _idle()
    device = get_device()
    specs = _targets()
    if not specs:
        raise SystemExit("no structured checkpoints to time")
    rows = []
    settings = [(1, "fp32", 300, 1000), (256, "fp32", 100, 300), (1, "fp16", 300, 1000), (256, "fp16", 100, 300)]
    loaded = []
    for run_id, arch, state in specs:
        model = model_from_arch(arch).to(device).eval()
        model.load_state_dict(state)
        loaded.append((run_id, model))
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for batch, dtype, warmup, measure in settings:
            models = []
            for run_id, model in loaded:
                clone = model
                if dtype == "fp16":
                    clone = model.half()
                else:
                    clone = model.float()
                clone = clone.to(memory_format=torch.channels_last)
                x = torch.zeros(batch, 3, 32, 32, device=device)
                if dtype == "fp16":
                    x = x.half()
                x = x.contiguous(memory_format=torch.channels_last)
                models.append((run_id, clone, x))
            repeats = {run_id: [] for run_id, _, _ in models}
            for _ in range(5):
                for run_id, clone, x in models:
                    with torch.inference_mode():
                        repeats[run_id].append(_time_once(clone, x, warmup, measure))
            for run_id, samples in repeats.items():
                ordered = sorted(samples)
                mid = statistics.median(ordered)
                q1 = ordered[len(ordered) // 4]
                q3 = ordered[(3 * len(ordered)) // 4]
                rows.append(
                    {
                        "run_id": run_id,
                        "batch": batch,
                        "dtype": dtype,
                        "median_ms": f"{mid:.4f}",
                        "iqr_ms": f"{q3 - q1:.4f}",
                    }
                )
                print(rows[-1], flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
    out = REPO / "report" / "tables" / "latency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "run_id,batch,dtype,median_ms,iqr_ms\n"
    body = "".join(f"{r['run_id']},{r['batch']},{r['dtype']},{r['median_ms']},{r['iqr_ms']}\n" for r in rows)
    out.write_text(header + body, encoding="utf-8")
    atomic_json(out.with_suffix(".json"), rows)


if __name__ == "__main__":
    main()
