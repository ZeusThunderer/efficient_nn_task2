"""He init (fan_out) and the unstructured density-scaled init from the paper."""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn

# Plan section 3.0. FT/WT/PRUNE do not draw a fresh init, but ids stay unique.
MODE_ID = {
    "BASE": 0,
    "SE": 1,
    "SB": 2,
    "UF": 3,
    "GD": 4,
    "BASES": 5,
    "SBU": 6,
    "FT": 7,
    "WT": 8,
    "PRUNE": 9,
}


def init_seed_for(seed: int, mode: str) -> int:
    if mode not in MODE_ID:
        raise KeyError(mode)
    return int(seed) + 1000 * MODE_ID[mode]


def init_module(model: nn.Module, seed: int, conv_density: dict[str, float] | None = None) -> None:
    cpu_state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight_name = f"{name}.weight"
            if conv_density is not None and weight_name in conv_density:
                density = max(float(conv_density[weight_name]), 1.0 / module.weight.numel())
                fan_in = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
                std = math.sqrt(2.0 / (fan_in * density))
                nn.init.normal_(module.weight, 0.0, std)
            else:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in = module.weight.size(1)
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)
    torch.random.set_rng_state(cpu_state)


def fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha1()
    for param in model.parameters():
        digest.update(param.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()
