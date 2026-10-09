"""Tiny CPU run used by the resume test. Not part of the paper matrix."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from src.device import get_device
from src.engine import fit
from src.initutil import init_module
from src.models import VGG


def main() -> None:
    torch.set_num_threads(1)
    device = get_device()
    out = Path(os.environ["RUN_DIR"])
    model = VGG("vgg16", [8] * 13, num_classes=10)
    init_module(model, 0)
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(0, 255, (256, 3, 32, 32), dtype=torch.uint8, generator=generator)
    labels = torch.randint(0, 10, (256,), generator=generator)
    fit(
        model,
        images.to(device),
        labels.to(device),
        epochs=4,
        lr=0.1,
        lr_schedule="stepwise",
        out_dir=out,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        data_seed=1,
        checkpoint_every=int(os.environ.get("CHECKPOINT_EVERY", "1")),
        meta={"run_id": "tiny", "precision": "fp32", "init_fingerprint": "x"},
    )


if __name__ == "__main__":
    main()
