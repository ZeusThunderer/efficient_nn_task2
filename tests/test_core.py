import torch

from src.data import aug_params, augment_batch, stratified_split
from src.flops import (
    analytical_resnet56_macs,
    analytical_vgg_macs,
    count_macs,
    scratch_b_epochs,
)
from src.initutil import init_module, init_seed_for
from src.models import (
    VGG16_WIDTHS,
    VGG19_WIDTHS,
    build_model,
    count_params,
    default_resnet_internals,
    resnet_param_count,
    vgg_param_count,
)
from src.prune import (
    apply_grad_mask_,
    apply_weight_mask_,
    conv_nnz,
    keeps_from_json,
    keeps_to_json,
    l1_keep_indices,
    slice_model,
    unstructured_masks,
)


def _l1_widths(ratios_by_index):
    widths = list(VGG16_WIDTHS)
    for index, ratio in ratios_by_index.items():
        widths[index] = max(1, VGG16_WIDTHS[index] - int(round(ratio * VGG16_WIDTHS[index])))
    return widths


def _l1_internals(ratios):
    base = default_resnet_internals()
    return [max(1, c - int(round(r * c))) for c, r in zip(base, ratios)]


def test_param_formulas_match_modules():
    vgg = build_model("vgg16", 10)
    vgg19 = build_model("vgg19", 100)
    res = build_model("resnet56", 10)
    assert count_params(vgg) == vgg_param_count(VGG16_WIDTHS, 10)
    assert count_params(vgg19) == vgg_param_count(VGG19_WIDTHS, 100)
    assert count_params(res) == resnet_param_count(default_resnet_internals(), 10)
    assert vgg(torch.zeros(2, 3, 32, 32)).shape == (2, 10)
    assert res(torch.zeros(2, 3, 32, 32)).shape == (2, 10)


def test_macs_match_analytical():
    vgg = build_model("vgg16", 10)
    res = build_model("resnet56", 10)
    assert count_macs(vgg) == analytical_vgg_macs("vgg16", VGG16_WIDTHS, 10)
    assert count_macs(res) == analytical_resnet56_macs(default_resnet_internals(), 10)


def test_scratch_b_epochs_for_l1_configs():
    full_v = count_macs(build_model("vgg16", 10))
    pruned_w = _l1_widths({0: 0.5, 7: 0.5, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5, 12: 0.5})
    pruned_v = count_macs(build_model("vgg16", 10, widths=pruned_w))
    e_v, _ = scratch_b_epochs(full_v, pruned_v)
    assert abs(e_v - 248) <= 8

    full_r = count_macs(build_model("resnet56", 10))
    a = _l1_internals([0.3] * 27)
    b = _l1_internals([0.3] * 9 + [0.6] * 18)
    e_a, _ = scratch_b_epochs(full_r, count_macs(build_model("resnet56", 10, internals=a)))
    e_b, _ = scratch_b_epochs(full_r, count_macs(build_model("resnet56", 10, internals=b)))
    assert abs(e_a - 228) <= 8
    assert abs(e_b - 320) <= 8
    assert scratch_b_epochs(100, 100)[0] == 160
    assert scratch_b_epochs(500, 100)[0] == 320


def test_vgg_shrink_matches_relu_mask():
    torch.manual_seed(0)
    full = build_model("vgg16", 10)
    init_module(full, 1)
    keeps = []
    for index, width in enumerate(VGG16_WIDTHS):
        if index == 2:
            keeps.append(torch.arange(0, width, 2))
        else:
            keeps.append(torch.arange(width))
    small = slice_model(full, keeps)
    full.eval()
    small.eval()
    dropped = [c for c in range(VGG16_WIDTHS[2]) if c not in set(keeps[2].tolist())]

    def hook(_module, _inputs, output):
        output = output.clone()
        output[:, dropped] = 0
        return output

    handle = full.relus[2].register_forward_hook(hook)
    x = torch.randn(2, 3, 32, 32)
    masked = full(x)
    handle.remove()
    assert torch.allclose(masked, small(x), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(full(x), small(x), atol=1e-3)


def test_resnet_shrink_matches_relu_mask():
    torch.manual_seed(0)
    full = build_model("resnet56", 10)
    init_module(full, 2)
    keeps = []
    for index, width in enumerate(default_resnet_internals()):
        keeps.append(torch.arange(0, width, 2) if index == 0 else torch.arange(width))
    small = slice_model(full, keeps)
    full.eval()
    small.eval()
    dropped = [c for c in range(default_resnet_internals()[0]) if c not in set(keeps[0].tolist())]

    def hook(_module, _inputs, output):
        output = output.clone()
        output[:, dropped] = 0
        return output

    handle = full.blocks[0].relu1.register_forward_hook(hook)
    x = torch.randn(2, 3, 32, 32)
    masked = full(x)
    handle.remove()
    assert torch.allclose(masked, small(x), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(full(x), small(x), atol=1e-3)
    assert keeps_from_json(keeps_to_json(keeps))[0].tolist() == keeps[0].tolist()


def test_unstructured_mask_stays_zero():
    torch.manual_seed(0)
    model = build_model("vgg16", 10)
    init_module(model, 3)
    masks = unstructured_masks(model, 0.8)
    apply_weight_mask_(model, masks)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    start = conv_nnz(model)
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        apply_grad_mask_(model, masks)
        opt.step()
        apply_weight_mask_(model, masks)
    assert conv_nnz(model) == start
    for name, param in model.named_parameters():
        if name in masks:
            assert torch.count_nonzero(param[masks[name] == 0]) == 0


def test_per_image_augmentation():
    generator = torch.Generator().manual_seed(123)
    tops, lefts, flips = aug_params(128, generator)
    offsets = list(zip(tops.tolist(), lefts.tolist()))
    assert len(set(offsets)) > 20
    assert 0.35 <= float(flips.float().mean()) <= 0.65
    images = torch.zeros(128, 3, 32, 32)
    images[:, 0, 0, 0] = 255
    again = torch.Generator().manual_seed(123)
    out = augment_batch(images, again, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    checked = 0
    for i in range(128):
        top, left = int(tops[i]), int(lefts[i])
        if top > 4 or left > 4:
            continue
        row, col = divmod(int(out[i, 0].argmax()), 32)
        expect_col = 4 - left
        if bool(flips[i]):
            expect_col = 31 - expect_col
        assert row == 4 - top
        assert col == expect_col
        checked += 1
    assert checked > 20


def test_init_seeds_do_not_overlap():
    base = build_model("vgg16", 10)
    scratch = build_model("vgg16", 10)
    init_module(base, init_seed_for(1, "BASE"))
    init_module(scratch, init_seed_for(1, "SE"))
    for left, right in zip(base.modules(), scratch.modules()):
        if isinstance(left, torch.nn.Conv2d):
            assert not torch.equal(left.weight, right.weight)
        if isinstance(left, torch.nn.Linear):
            assert not torch.equal(left.weight, right.weight)
    assert not torch.equal(base.convs[0].weight, scratch.convs[0].weight)


def test_stratified_split_is_deterministic():
    labels = torch.arange(50000) % 10
    train, val = stratified_split(labels, seed=0)
    assert train.numel() == 45000
    assert val.numel() == 5000
    again, _ = stratified_split(labels, seed=0)
    assert torch.equal(train, again)
    for class_id in range(10):
        assert int((labels[val] == class_id).sum()) == 500
    labels100 = torch.arange(50000) % 100
    _, val100 = stratified_split(labels100, seed=0)
    assert val100.numel() == 5000


def test_l1_keep_count():
    weight = torch.randn(64, 3, 3, 3)
    kept = l1_keep_indices(weight, 0.5)
    assert kept.numel() == 32


def test_source_has_no_best_epoch_selection():
    from pathlib import Path

    text = "\n".join(path.read_text(encoding="utf-8") for path in Path("src").rglob("*.py"))
    assert "best_acc" not in text
    assert "best_epoch" not in text
