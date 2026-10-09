"""MAC counter. FLOPs in this project means MACs, matching the paper's convention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models import DEFAULT_WIDTHS, VGG_POOL_AFTER, model_from_arch


def scratch_b_epochs(f_l: float, f_p: float, e_l: int = 160, kappa_max: float = 2.0) -> tuple[int, float]:
    if f_p <= 0:
        raise ValueError("F_P must be positive")
    raw = f_l / f_p
    kappa = min(kappa_max, raw) if math.isfinite(kappa_max) else raw
    epochs = 4 * int(round(e_l * kappa / 4.0))
    return int(epochs), float(kappa)


def count_macs(model: nn.Module, input_size=(1, 3, 32, 32), conv_masks: dict | None = None, device=None) -> int:
    """Count MACs from real module shapes. Masks, if given, are keyed by parameter name."""
    was_training = model.training
    model.eval()
    device = device or next(model.parameters()).device
    total = [0]
    hooks = []

    def conv_hook(param_name):
        def _hook(module, _inputs, output):
            height, width = output.shape[2], output.shape[3]
            if conv_masks is not None and param_name in conv_masks:
                nnz = int(conv_masks[param_name].detach().sum().item())
                total[0] += nnz * height * width
            else:
                k_h, k_w = module.kernel_size
                total[0] += (
                    height
                    * width
                    * module.out_channels
                    * (module.in_channels // module.groups)
                    * k_h
                    * k_w
                )

        return _hook

    def linear_hook(module, _inputs, _output):
        total[0] += module.in_features * module.out_features

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook(f"{name}.weight")))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(torch.zeros(*input_size, device=device))
    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return int(total[0])


def analytical_vgg_macs(kind: str, widths: list[int], num_classes: int) -> int:
    pools = VGG_POOL_AFTER[kind]
    cin = 3
    spatial = 32
    total = 0
    for index, width in enumerate(widths):
        total += cin * width * 9 * spatial * spatial
        cin = width
        if index in pools:
            spatial //= 2
    total += cin * num_classes
    return int(total)


def analytical_resnet56_macs(internals: list[int], num_classes: int) -> int:
    total = 3 * 16 * 9 * 32 * 32
    outs = [16] * 9 + [32] * 9 + [64] * 9
    in_planes = 16
    spatial = 32
    for index, internal in enumerate(internals):
        stage = index // 9
        block = index % 9
        if block == 0 and stage > 0:
            spatial //= 2
        total += in_planes * internal * 9 * spatial * spatial
        total += internal * outs[index] * 9 * spatial * spatial
        in_planes = outs[index]
    total += 64 * num_classes
    return int(total)


def macs_of_arch(arch: dict, conv_masks: dict | None = None) -> int:
    model = model_from_arch(arch)
    if conv_masks is None and arch["kind"] in DEFAULT_WIDTHS:
        return analytical_vgg_macs(arch["kind"], arch["widths"], arch["num_classes"])
    if conv_masks is None and arch["kind"] == "resnet56":
        return analytical_resnet56_macs(arch["internals"], arch["num_classes"])
    return count_macs(model, conv_masks=conv_masks)
