"""CIFAR in GPU memory. Augmentation is per-image, not per-batch."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CIFAR100_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def dataset_stats(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if name == "cifar10":
        return CIFAR10_MEAN, CIFAR10_STD
    if name == "cifar100":
        return CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(name)


def num_classes(name: str) -> int:
    return 10 if name == "cifar10" else 100


def load_cifar(root: Path | str, name: str, device: torch.device, download: bool = True):
    cls = torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
    train = cls(root=str(root), train=True, download=download)
    test = cls(root=str(root), train=False, download=download)

    def pack(ds):
        images = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()
        labels = torch.tensor(ds.targets, dtype=torch.long)
        return images.to(device), labels.to(device)

    return pack(train), pack(test)


def stratified_split(labels: torch.Tensor, val_fraction: float = 0.1, seed: int = 0):
    """Deterministic stratified split. Returns CPU long tensors of indices."""
    labels_cpu = labels.detach().cpu()
    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    val_parts = []
    for class_id in sorted(labels_cpu.unique().tolist()):
        idx = torch.where(labels_cpu == class_id)[0]
        perm = idx[torch.randperm(idx.numel(), generator=generator)]
        n_val = int(round(idx.numel() * val_fraction))
        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])
    train_idx = torch.sort(torch.cat(train_parts)).values
    val_idx = torch.sort(torch.cat(val_parts)).values
    return train_idx, val_idx


def aug_params(n: int, generator: torch.Generator):
    tops = torch.randint(0, 9, (n,), generator=generator)
    lefts = torch.randint(0, 9, (n,), generator=generator)
    flips = torch.rand(n, generator=generator) < 0.5
    return tops, lefts, flips


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = x.new_tensor(mean).view(1, 3, 1, 1)
    std_t = x.new_tensor(std).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def augment_batch(x_uint8: torch.Tensor, generator: torch.Generator, mean, std) -> torch.Tensor:
    """Pad-4 crop and horizontal flip, independently for every image."""
    n = x_uint8.shape[0]
    tops, lefts, flips = aug_params(n, generator)
    x = F.pad(x_uint8.float(), (4, 4, 4, 4))
    device = x.device
    tops = tops.to(device)
    lefts = lefts.to(device)
    rows = tops[:, None, None] + torch.arange(32, device=device).view(1, 32, 1)
    cols = lefts[:, None, None] + torch.arange(32, device=device).view(1, 1, 32)
    rows = rows.expand(n, 32, 32)
    cols = cols.expand(n, 32, 32)
    flat = x.reshape(n, x.shape[1], 40 * 40)
    index = (rows * 40 + cols).reshape(n, 1, 32 * 32).expand(n, x.shape[1], 32 * 32)
    cropped = flat.gather(2, index).reshape(n, x.shape[1], 32, 32)
    flips = flips.to(device)
    if bool(flips.any()):
        cropped = cropped.clone()
        cropped[flips] = cropped[flips].flip(-1)
    return _normalize(cropped / 255.0, mean, std)


def normalize_batch(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    return _normalize(x_uint8.float() / 255.0, mean, std)


def epoch_generator(data_seed: int, epoch: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(data_seed) * 1_000_003 + int(epoch))
