"""Pick the Blackwell GPU. Never fall back to the RTX 2070 (sm_75)."""

from __future__ import annotations

import os

import torch


def get_device() -> torch.device:
    forced = os.environ.get("PRUNE_DEVICE")
    if forced:
        return torch.device(forced)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Set PRUNE_DEVICE=cpu only for unit tests.")
    for index in range(torch.cuda.device_count()):
        if torch.cuda.get_device_capability(index) == (12, 0):
            torch.cuda.set_device(index)
            return torch.device(f"cuda:{index}")
    visible = [
        (i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
        for i in range(torch.cuda.device_count())
    ]
    raise RuntimeError(
        "No sm_120 GPU visible. Train on the RTX 5070 Ti "
        f"(CUDA_VISIBLE_DEVICES=1). Visible: {visible}"
    )
