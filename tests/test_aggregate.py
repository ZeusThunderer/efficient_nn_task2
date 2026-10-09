from src.aggregate import H1_CFGS, hypotheses


def _row(cfg, mode, seed, acc, eb=None):
    return {
        "cfg_name": cfg,
        "mode": mode,
        "seed": seed,
        "test_acc_last": acc,
        "E_B": eb,
        "split": "full",
    }


def _idx(rows):
    return {(r["cfg_name"], r["mode"], r["seed"]): r for r in rows}


def test_h1_confirmed_when_scratch_matches_finetune():
    rows = []
    for cfg in H1_CFGS:
        for seed in (1, 2, 3):
            rows.append(_row(cfg, "FT", seed, 90.0))
            rows.append(_row(cfg, "SB", seed, 90.0, eb=200))
            rows.append(_row(cfg, "SE", seed, 89.9))
    for cfg in ("L1-V16A", "L1-R56A", "L1-R56B"):
        rows.append(_row(cfg, "WT", 1, 89.9))
    verdict = hypotheses(_idx(rows))
    assert verdict["H1"] == "подтверждена"
    assert verdict["H1c"] == "подтверждена"


def test_h1_not_checked_when_a_config_is_missing():
    rows = []
    for cfg in H1_CFGS[:-1]:
        rows.append(_row(cfg, "FT", 1, 90.0))
        rows.append(_row(cfg, "SB", 1, 90.0, eb=200))
        rows.append(_row(cfg, "SE", 1, 90.0))
    verdict = hypotheses(_idx(rows))
    assert verdict["H1"] == "не проверено"


def test_h4_splits_the_two_sparsities():
    rows = [
        _row("UN-V19-80", "FT", 1, 90.0),
        _row("UN-V19-80", "SB", 1, 89.8, eb=320),
        _row("UN-R56-95", "FT", 1, 90.0),
        _row("UN-R56-95", "SB", 1, 88.0, eb=320),
    ]
    verdict = hypotheses(_idx(rows))
    assert verdict["H4"] == "подтверждена"
