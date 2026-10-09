"""L1 filter pruning, Network Slimming, unstructured magnitude, uniform and guided widths."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models import (
    DEFAULT_WIDTHS,
    VGG_STAGES,
    ResNet56,
    VGG,
    build_model,
    count_params,
    default_resnet_internals,
    resnet_param_count,
    vgg_param_count,
)


def _as_index(idx: torch.Tensor, device) -> torch.Tensor:
    return idx.to(device=device, dtype=torch.long)


def copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d, idx: torch.Tensor | None = None) -> None:
    if idx is None:
        dst.load_state_dict({k: v.detach().clone() for k, v in src.state_dict().items()})
        return
    idx = _as_index(idx, src.weight.device)
    dst.weight.data.copy_(src.weight.data.index_select(0, idx))
    dst.bias.data.copy_(src.bias.data.index_select(0, idx))
    dst.running_mean.copy_(src.running_mean.index_select(0, idx))
    dst.running_var.copy_(src.running_var.index_select(0, idx))
    dst.num_batches_tracked.copy_(src.num_batches_tracked)


def l1_keep_indices(weight: torch.Tensor, ratio: float) -> torch.Tensor:
    """Keep the filters with the largest L1 norm. Ties break toward lower indices (stable argsort)."""
    channels = weight.shape[0]
    n_remove = int(round(float(ratio) * channels))
    n_keep = max(1, channels - n_remove)
    norms = weight.detach().float().abs().sum(dim=(1, 2, 3))
    order = torch.argsort(norms, stable=True)
    kept = order[channels - n_keep :]
    return torch.sort(kept.cpu()).values


def _slice_vgg(model: VGG, keeps: list[torch.Tensor]) -> VGG:
    widths = [int(k.numel()) for k in keeps]
    new = VGG(model.kind, widths, num_classes=int(model.fc.out_features))
    with torch.no_grad():
        for i, idx in enumerate(keeps):
            src = model.convs[i]
            in_idx = keeps[i - 1] if i else torch.arange(src.in_channels)
            in_idx = _as_index(in_idx, src.weight.device)
            out_idx = _as_index(idx, src.weight.device)
            new.convs[i].weight.copy_(src.weight.index_select(0, out_idx).index_select(1, in_idx))
            copy_bn(new.bns[i], model.bns[i], idx)
        in_idx = _as_index(keeps[-1], model.fc.weight.device)
        new.fc.weight.copy_(model.fc.weight.index_select(1, in_idx))
        new.fc.bias.copy_(model.fc.bias)
    return new


def _slice_resnet(model: ResNet56, keeps: list[torch.Tensor]) -> ResNet56:
    internals = [int(k.numel()) for k in keeps]
    new = ResNet56(internals, num_classes=int(model.fc.out_features))
    with torch.no_grad():
        new.stem.weight.copy_(model.stem.weight)
        copy_bn(new.stem_bn, model.stem_bn)
        for i, idx in enumerate(keeps):
            src = model.blocks[i]
            dst = new.blocks[i]
            out_idx = _as_index(idx, src.conv1.weight.device)
            dst.conv1.weight.copy_(src.conv1.weight.index_select(0, out_idx))
            copy_bn(dst.bn1, src.bn1, idx)
            in_idx = _as_index(idx, src.conv2.weight.device)
            dst.conv2.weight.copy_(src.conv2.weight.index_select(1, in_idx))
            copy_bn(dst.bn2, src.bn2)
        new.fc.weight.copy_(model.fc.weight)
        new.fc.bias.copy_(model.fc.bias)
    return new


def slice_model(model: nn.Module, keeps: list[torch.Tensor]) -> nn.Module:
    if isinstance(model, VGG):
        return _slice_vgg(model, keeps)
    if isinstance(model, ResNet56):
        return _slice_resnet(model, keeps)
    raise TypeError(type(model))


def l1_keeps(model: nn.Module, ratios: list[float] | dict[int, float]) -> list[torch.Tensor]:
    if isinstance(model, VGG):
        keeps = []
        for i, conv in enumerate(model.convs):
            ratio = ratios[i] if isinstance(ratios, dict) else ratios[i]
            if ratio <= 0:
                keeps.append(torch.arange(conv.out_channels))
            else:
                keeps.append(l1_keep_indices(conv.weight.data, ratio))
        return keeps
    if isinstance(model, ResNet56):
        keeps = []
        for i, block in enumerate(model.blocks):
            ratio = ratios[i] if isinstance(ratios, dict) else ratios[i]
            keeps.append(l1_keep_indices(block.conv1.weight.data, ratio))
        return keeps
    raise TypeError(type(model))


def _gamma_entries(model: nn.Module) -> list[tuple[float, int, int]]:
    entries = []
    if isinstance(model, VGG):
        modules = list(model.bns)
    elif isinstance(model, ResNet56):
        modules = [block.bn1 for block in model.blocks]
    else:
        raise TypeError(type(model))
    for layer, bn in enumerate(modules):
        gamma = bn.weight.detach().float().abs()
        for channel, score in enumerate(gamma.tolist()):
            entries.append((float(score), layer, channel))
    return entries


def slimming_keeps(model: nn.Module, ratio: float) -> tuple[list[torch.Tensor], bool, float]:
    """Global top-k on |gamma|. Returns keeps, hit_min1, actual pruned fraction."""
    entries = _gamma_entries(model)
    n_total = len(entries)
    n_keep = int(round((1.0 - float(ratio)) * n_total))
    n_keep = min(max(n_keep, 1), n_total)
    ranked = sorted(entries, key=lambda item: (-item[0], item[1], item[2]))
    chosen = ranked[:n_keep]
    if isinstance(model, VGG):
        sizes = [bn.weight.numel() for bn in model.bns]
    else:
        sizes = [block.bn1.weight.numel() for block in model.blocks]
    by_layer: dict[int, list[int]] = {i: [] for i in range(len(sizes))}
    for _score, layer, channel in chosen:
        by_layer[layer].append(channel)
    hit_min1 = False
    for layer, size in enumerate(sizes):
        if by_layer[layer]:
            continue
        hit_min1 = True
        best = max((item for item in entries if item[1] == layer), key=lambda item: (item[0], -item[2]))
        by_layer[layer] = [best[2]]
    keeps = [torch.tensor(sorted(by_layer[i]), dtype=torch.long) for i in range(len(sizes))]
    kept = sum(int(k.numel()) for k in keeps)
    actual = 1.0 - kept / n_total
    return keeps, hit_min1, actual


def unstructured_masks(model: nn.Module, ratio: float) -> dict[str, torch.Tensor]:
    """One-shot global top-k on |conv weight|. Mask is 1 for kept weights. FC and BN stay dense."""
    chunks = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            chunks.append(module.weight.detach().float().abs().reshape(-1).cpu())
            names.append(f"{name}.weight")
    flat = torch.cat(chunks)
    n_prune = int(round(float(ratio) * flat.numel()))
    keep = torch.ones(flat.numel(), dtype=torch.bool)
    if n_prune > 0:
        _values, index = torch.topk(flat, n_prune, largest=False, sorted=False)
        keep[index] = False
    masks = {}
    offset = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        numel = module.weight.numel()
        masks[f"{name}.weight"] = keep[offset : offset + numel].reshape(module.weight.shape).float()
        offset += numel
    return masks


def apply_weight_mask_(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))


def apply_grad_mask_(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    for name, param in model.named_parameters():
        if name in masks and param.grad is not None:
            param.grad.mul_(masks[name].to(param.grad.device))


def conv_nnz(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total += int(torch.count_nonzero(module.weight.detach()).item())
    return total


def layer_densities(masks: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(mask.mean().item()) for name, mask in masks.items()}


def add_slimming_grad_(bns: list[nn.BatchNorm2d], lam: float) -> None:
    """L1 on gamma in fp32, added to the gradient. Not implemented via weight decay."""
    if lam == 0:
        return
    for bn in bns:
        if bn.weight.grad is None:
            continue
        penalty = (lam * bn.weight.detach().float().sign()).to(dtype=bn.weight.grad.dtype)
        bn.weight.grad.add_(penalty)


def gamma_small_fraction(model: nn.Module, threshold: float = 1e-2) -> float:
    gammas = []
    for bn in model.prunable_bns():
        gammas.append(bn.weight.detach().float().abs().reshape(-1))
    if not gammas:
        return 0.0
    flat = torch.cat(gammas)
    return float((flat < threshold).float().mean().item())


def uniform_widths(kind: str, target_params: int, num_classes: int) -> tuple[list[int], int]:
    base = list(DEFAULT_WIDTHS[kind])
    best_widths = list(base)
    best_params = vgg_param_count(base, num_classes)
    best_err = abs(best_params - target_params)
    for step in range(1, 1001):
        scale = step / 1000.0
        widths = [max(1, int(round(scale * c))) for c in base]
        params = vgg_param_count(widths, num_classes)
        err = abs(params - target_params)
        if err < best_err or (err == best_err and params < best_params):
            best_err = err
            best_widths = widths
            best_params = params
    return best_widths, best_params


def uniform_internals(target_params: int, num_classes: int) -> tuple[list[int], int]:
    base = default_resnet_internals()
    best = list(base)
    best_params = resnet_param_count(base, num_classes)
    best_err = abs(best_params - target_params)
    for step in range(1, 1001):
        scale = step / 1000.0
        internals = [max(1, int(round(scale * c))) for c in base]
        params = resnet_param_count(internals, num_classes)
        err = abs(params - target_params)
        if err < best_err or (err == best_err and params < best_params):
            best_err = err
            best = internals
            best_params = params
    return best, best_params


def guided_widths(kind: str, src_widths: list[int], target_params: int, num_classes: int) -> tuple[list[int], int]:
    stages = VGG_STAGES[kind]
    averages = []
    for stage in stages:
        averages.append(sum(src_widths[i] for i in stage) / len(stage))
    best_widths: list[int] | None = None
    best_params = 0
    best_err = 10**18
    for step in range(1, 4001):
        scale = step / 1000.0
        widths: list[int] = []
        for stage, average in zip(stages, averages):
            width = max(1, int(round(average * scale)))
            widths.extend([width] * len(stage))
        params = vgg_param_count(widths, num_classes)
        err = abs(params - target_params)
        if err < best_err or (err == best_err and params < best_params):
            best_err = err
            best_widths = widths
            best_params = params
    assert best_widths is not None
    return best_widths, best_params


def keeps_to_json(keeps: list[torch.Tensor]) -> list[list[int]]:
    return [k.cpu().tolist() for k in keeps]


def keeps_from_json(raw: list[list[int]]) -> list[torch.Tensor]:
    return [torch.tensor(item, dtype=torch.long) for item in raw]
