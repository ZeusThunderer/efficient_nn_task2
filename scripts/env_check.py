"""Gate G0. Writes logs/env_check.txt and exits non-zero on failure."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CIFAR10_MD5, CIFAR100_MD5, load_cifar  # noqa: E402
from src.paths import DATA_ROOT  # noqa: E402


def main() -> None:
    lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    smi = subprocess.check_output(["nvidia-smi"], text=True)
    log(smi)
    if "5070 Ti" not in smi:
        raise SystemExit("RTX 5070 Ti not visible")

    import torch
    import torch.nn as nn

    log(f"torch {torch.__version__} cuda {torch.version.cuda}")
    arch = torch.cuda.get_arch_list()
    log(f"arch_list {arch}")
    if "sm_120" not in arch:
        raise SystemExit("sm_120 missing from this wheel")
    device = None
    for index in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        log(f"device {index} {name} cap {cap}")
        if cap == (12, 0):
            device = torch.device(f"cuda:{index}")
    if device is None:
        raise SystemExit("no sm_120 device")
    x = torch.randn(2, 3, 32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
    layer = nn.Conv2d(3, 8, 3, padding=1).to(device=device, dtype=torch.bfloat16)
    y = layer(x)
    y.sum().backward()
    x32 = torch.randn(2, 3, 32, 32, device=device)
    y32 = nn.Conv2d(3, 8, 3, padding=1).to(device)(x32)
    log(f"bf16 conv {tuple(y.shape)} fp32 conv {tuple(y32.shape)}")

    load_cifar(DATA_ROOT, "cifar10", torch.device("cpu"))
    load_cifar(DATA_ROOT, "cifar100", torch.device("cpu"))
    for name, digest in (
        ("cifar-10-python.tar.gz", CIFAR10_MD5),
        ("cifar-100-python.tar.gz", CIFAR100_MD5),
    ):
        path = DATA_ROOT / name
        if not path.exists():
            raise SystemExit(f"missing {path}")
        got = hashlib.md5(path.read_bytes()).hexdigest()
        log(f"md5 {name} {got}")
        if got != digest:
            raise SystemExit(f"md5 mismatch for {name}")

    usage = shutil.disk_usage(ROOT)
    log(f"disk_free_gb {usage.free / 2**30:.1f}")
    if usage.free < 30 * 2**30:
        raise SystemExit("need >= 30 GiB free")

    from src.tracking import Tracker

    tracker = Tracker(True, ROOT / ".aim", "env-check", "g0", {"gate": "g0"}, {"ok": 1})
    if tracker.run is None:
        raise SystemExit(f"aim failed: {tracker.error}")
    tracker.track("ok", 1.0, 0)
    tracker.close()
    disabled = Tracker(False, ROOT / ".aim", "env-off", "g0", {}, {})
    if disabled.run is not None:
        raise SystemExit("tracking.enabled=false still opened a run")
    log("aim ok")
    out = ROOT / "logs" / "env_check.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
