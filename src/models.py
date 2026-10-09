"""CIFAR VGG-16/19 (BN, single FC) and ResNet-56 (option A). Widths are configurable."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

VGG16_WIDTHS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
VGG19_WIDTHS = [
    64, 64, 128, 128, 256, 256, 256, 256,
    512, 512, 512, 512, 512, 512, 512, 512,
]
VGG_POOL_AFTER = {
    "vgg16": {1, 3, 6, 9, 12},
    "vgg19": {1, 3, 7, 11, 15},
}
VGG_STAGES = {
    "vgg16": [[0, 1], [2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
    "vgg19": [[0, 1], [2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
}
DEFAULT_WIDTHS = {"vgg16": VGG16_WIDTHS, "vgg19": VGG19_WIDTHS}

RESNET_STAGE_OUT = [16, 32, 64]
RESNET_BLOCKS_PER_STAGE = 9


def default_resnet_internals() -> list[int]:
    return [16] * 9 + [32] * 9 + [64] * 9


class VGG(nn.Module):
    def __init__(self, kind: str, widths: list[int], num_classes: int = 10):
        super().__init__()
        if kind not in DEFAULT_WIDTHS:
            raise ValueError(kind)
        if len(widths) != len(DEFAULT_WIDTHS[kind]):
            raise ValueError(f"{kind} expects {len(DEFAULT_WIDTHS[kind])} convs, got {len(widths)}")
        self.kind = kind
        self.widths = [int(w) for w in widths]
        self.pool_after = VGG_POOL_AFTER[kind]
        convs, bns, relus = [], [], []
        in_c = 3
        for out_c in self.widths:
            convs.append(nn.Conv2d(in_c, out_c, 3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(out_c))
            relus.append(nn.ReLU(inplace=False))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.relus = nn.ModuleList(relus)
        self.fc = nn.Linear(self.widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn, relu) in enumerate(zip(self.convs, self.bns, self.relus)):
            x = relu(bn(conv(x)))
            if i in self.pool_after:
                x = F.max_pool2d(x, 2)
        return self.fc(torch.flatten(x, 1))

    def prunable_bns(self) -> list[nn.BatchNorm2d]:
        return list(self.bns)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, internal: int, out_planes: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, internal, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(internal, out_planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes

    def shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1 and self.in_planes == self.out_planes:
            return x
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.in_planes != self.out_planes:
            pad = self.out_planes - self.in_planes
            x = F.pad(x, (0, 0, 0, 0, pad // 2, pad - pad // 2))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + self.shortcut(x))


class ResNet56(nn.Module):
    def __init__(self, internals: list[int] | None = None, num_classes: int = 10):
        super().__init__()
        if internals is None:
            internals = default_resnet_internals()
        if len(internals) != 27:
            raise ValueError(f"ResNet-56 expects 27 internal widths, got {len(internals)}")
        self.internals = [int(v) for v in internals]
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(16)
        self.stem_relu = nn.ReLU(inplace=False)
        blocks = []
        in_planes = 16
        for stage, out_planes in enumerate(RESNET_STAGE_OUT):
            for b in range(RESNET_BLOCKS_PER_STAGE):
                stride = 2 if (b == 0 and stage > 0) else 1
                internal = self.internals[stage * 9 + b]
                blocks.append(BasicBlock(in_planes, internal, out_planes, stride))
                in_planes = out_planes
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            x = block(x)
        x = F.avg_pool2d(x, x.shape[-1])
        return self.fc(torch.flatten(x, 1))

    def prunable_bns(self) -> list[nn.BatchNorm2d]:
        # Network Slimming on ResNet-56 prunes internal channels only (bn1).
        return [block.bn1 for block in self.blocks]


def build_model(kind: str, num_classes: int = 10, widths=None, internals=None) -> nn.Module:
    if kind in DEFAULT_WIDTHS:
        return VGG(kind, list(widths) if widths is not None else list(DEFAULT_WIDTHS[kind]), num_classes)
    if kind == "resnet56":
        return ResNet56(internals, num_classes)
    raise ValueError(kind)


def arch_from_model(model: nn.Module) -> dict:
    if isinstance(model, VGG):
        return {"kind": model.kind, "widths": list(model.widths), "num_classes": int(model.fc.out_features)}
    if isinstance(model, ResNet56):
        return {
            "kind": "resnet56",
            "internals": list(model.internals),
            "num_classes": int(model.fc.out_features),
        }
    raise TypeError(type(model))


def model_from_arch(arch: dict) -> nn.Module:
    kind = arch["kind"]
    num_classes = int(arch["num_classes"])
    if kind in DEFAULT_WIDTHS:
        return build_model(kind, num_classes, widths=arch["widths"])
    if kind == "resnet56":
        return build_model("resnet56", num_classes, internals=arch["internals"])
    raise ValueError(kind)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def vgg_param_count(widths: list[int], num_classes: int) -> int:
    total = 0
    prev = 3
    for width in widths:
        total += prev * width * 9
        total += 2 * width
        prev = width
    total += prev * num_classes + num_classes
    return total


def resnet_param_count(internals: list[int], num_classes: int) -> int:
    total = 3 * 16 * 9 + 2 * 16
    in_planes = 16
    outs = [16] * 9 + [32] * 9 + [64] * 9
    for i, internal in enumerate(internals):
        out_planes = outs[i]
        total += in_planes * internal * 9 + 2 * internal
        total += internal * out_planes * 9 + 2 * out_planes
        in_planes = out_planes
    total += 64 * num_classes + num_classes
    return total
