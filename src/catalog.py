"""Run matrix. Priority is lower-first. cut_class is what the budget cutter removes."""

from __future__ import annotations

# cfg, model, dataset, method, modes, sparse base, group
_P1 = [
    ("L1-V16A", "vgg16", "cifar10", "l1", ("FT", "SE", "SB"), False),
    ("L1-R56A", "resnet56", "cifar10", "l1", ("FT", "SE", "SB"), False),
    ("L1-R56B", "resnet56", "cifar10", "l1", ("FT", "SE", "SB"), False),
    ("SL-V19-70", "vgg19", "cifar10", "slimming", ("FT", "SE", "SB"), True),
    ("SL-R56-60", "resnet56", "cifar10", "slimming", ("FT", "SE", "SB"), True),
]
_P2A = [
    ("SL-R56-40", "resnet56", "cifar10", "slimming", ("FT", "SE", "SB"), True),
    ("SL-R56-80", "resnet56", "cifar10", "slimming", ("FT", "SE", "SB"), True),
    ("SL-V19-30", "vgg19", "cifar10", "slimming", ("SE",), True),
    ("SL-V19-50", "vgg19", "cifar10", "slimming", ("SE",), True),
    ("UN-V19-80", "vgg19", "cifar10", "unstructured", ("FT", "SE", "SB"), False),
    ("UN-V19-95", "vgg19", "cifar10", "unstructured", ("FT", "SE", "SB"), False),
    ("UN-R56-95", "resnet56", "cifar10", "unstructured", ("FT", "SE", "SB"), False),
]
_UF = [
    ("UF-V19-30", "SL-V19-30", "vgg19", "cifar10"),
    ("UF-V19-50", "SL-V19-50", "vgg19", "cifar10"),
    ("UF-V19-70", "SL-V19-70", "vgg19", "cifar10"),
    ("UF-R56-40", "SL-R56-40", "resnet56", "cifar10"),
    ("UF-R56-60", "SL-R56-60", "resnet56", "cifar10"),
    ("UF-R56-80", "SL-R56-80", "resnet56", "cifar10"),
]
_GD = [
    ("GD-V19-30", "SL-V19-30"),
    ("GD-V19-50", "SL-V19-50"),
    ("GD-V19-70", "SL-V19-70"),
]

# Prior FLOP ratios used only until a real prune.json exists.
FLOP_RATIO_PRIOR = {
    "L1-V16A": 0.658,
    "L1-R56A": 0.694,
    "L1-R56B": 0.504,
    "SL-V19-30": 0.70,
    "SL-V19-50": 0.45,
    "SL-V19-70": 0.25,
    "SL-R56-40": 0.60,
    "SL-R56-60": 0.42,
    "SL-R56-80": 0.22,
    "SL-V19-50-C100": 0.45,
    "UN-V19-80": 0.20,
    "UN-V19-95": 0.05,
    "UN-R56-80": 0.20,
    "UN-R56-95": 0.05,
    "UF-V19-30": 0.70,
    "UF-V19-50": 0.45,
    "UF-V19-70": 0.25,
    "UF-R56-40": 0.60,
    "UF-R56-60": 0.42,
    "UF-R56-80": 0.22,
    "GD-V19-30": 0.70,
    "GD-V19-50": 0.45,
    "GD-V19-70": 0.25,
}

# Removed from the tail until the budget fits. Later entries are cut later.
CUT_ORDER = [
    "p2a-s3",
    "p2a-s2",
    "p2b-c100",
    "p2b-gd",
    "p2b-un80",
    "p2b-sbu",
    "p2b-wt",
    "p1-s3-expensive",
    "p1-s3-r56a",
    "p1-s3-sl70",
    "p1-s3-v16",
]

T_EP_PRIOR = {"vgg16": 4.0, "vgg19": 5.0, "resnet56": 7.0}
RHO = {"vgg16": 0.40, "vgg19": 0.40, "resnet56": 0.80}
U_HOURS = 20.4
T_DEV_PRIOR = 3.0
T_LAT_PRIOR = 0.5
C_FIX = 0.01


def _family(model: str, dataset: str, sparse: bool) -> str:
    if dataset == "cifar100":
        stem = "V19C100"
    elif model == "vgg16":
        stem = "V16"
    elif model == "vgg19":
        stem = "V19"
    else:
        stem = "R56"
    return f"{stem}.{'BASES' if sparse else 'BASE'}"


def _cut_class(group: str, cfg: str, seed: int, mode: str) -> str:
    if group == "P2b":
        if cfg.startswith("GD"):
            return "p2b-gd"
        if cfg == "UN-R56-80":
            return "p2b-un80"
        if mode == "SBU" or cfg.startswith("SBU"):
            return "p2b-sbu"
        if mode == "WT":
            return "p2b-wt"
        if "C100" in cfg:
            return "p2b-c100"
        return "p2b-c100"
    if group == "P2a":
        return f"p2a-s{seed}"
    if group == "P1" and seed == 3 and cfg in {"L1-R56B", "SL-R56-60"}:
        return "p1-s3-expensive"
    if group == "P1" and seed == 3 and cfg == "L1-R56A":
        return "p1-s3-r56a"
    if group == "P1" and seed == 3 and cfg == "SL-V19-70":
        return "p1-s3-sl70"
    if group == "P1" and seed == 3 and cfg == "L1-V16A":
        return "p1-s3-v16"
    if group == "P1":
        return "p1-keep"
    return "keep"


def _band(group: str, seed: int, mode: str, cut_class: str) -> int:
    if cut_class.startswith("p2b"):
        return {"p2b-wt": 70, "p2b-sbu": 71, "p2b-un80": 72, "p2b-gd": 73, "p2b-c100": 74}[cut_class]
    if group == "P2a" and seed >= 2:
        return 80 if seed == 2 else 90
    if mode in {"BASE", "BASES"}:
        return 10 if seed == 1 else 50
    if mode == "PRUNE":
        if group == "P2a" and seed >= 2:
            return 79 if seed == 2 else 89
        return 20 if seed == 1 else 55
    if group == "P1":
        return 30 if seed == 1 else 60
    if group == "P2a":
        return 40
    return 95


def _job(
    run_id,
    mode,
    model,
    dataset,
    cfg_name,
    method,
    seed,
    deps,
    group,
    cut_class,
    epochs,
    lr,
    lr_schedule,
    dep_id,
    slim_lambda=1e-4,
    precision="bf16",
    split="full",
    hparams_hash="PENDING",
    do_eval=True,
    force_epochs=False,
    kappa_max=2.0,
) -> dict:
    priority = _band(group, seed, mode, cut_class) * 10000 - int(epochs)
    return {
        "run_id": run_id,
        "enabled": True,
        "deps": deps,
        "priority": priority,
        "epochs": int(epochs),
        "group": group,
        "cut_class": cut_class,
        "overrides": {
            "run_id": run_id,
            "mode": mode,
            "model": model,
            "dataset": dataset,
            "cfg_name": cfg_name,
            "method": method,
            "seed": seed,
            "split": split,
            "precision": precision,
            "lr_schedule": lr_schedule,
            "lr": lr,
            "epochs": int(epochs),
            "dep_id": dep_id or "none",
            "kappa_max": kappa_max,
            "slim_lambda": slim_lambda,
            "tracking": True,
            "do_eval": do_eval,
            "force_epochs": force_epochs,
            "hparams_hash": hparams_hash,
            "group": group,
        },
    }


def _lr_schedule(mode: str) -> tuple[float, str, int]:
    if mode == "FT":
        return 0.001, "constant", 40
    if mode == "PRUNE":
        return 0.0, "none", 0
    return 0.1, "stepwise", 160


def final_runs(hparams: dict | None = None) -> list[dict]:
    """Every reportable run. Epochs for SB/SBU are priors; execute replaces them from prune.json."""
    from src.flops import scratch_b_epochs

    hparams = hparams or {}
    precision = hparams.get("precision", "bf16")
    lr_ft = float(hparams.get("lr_ft", 0.001))
    lam_vgg = float(hparams.get("slim_lambda_vgg", 1e-4))
    lam_res = float(hparams.get("slim_lambda_resnet", 1e-4))
    hsh = hparams.get("hash", "PENDING")
    runs: list[dict] = []
    bases: set[tuple] = set()

    def lam_of(model: str) -> float:
        return lam_res if model.startswith("resnet") else lam_vgg

    def add_base(model, dataset, seed, sparse, group, cut_class):
        key = (model, dataset, seed, sparse)
        if key in bases:
            return
        bases.add(key)
        fam = _family(model, dataset, sparse)
        mode = "BASES" if sparse else "BASE"
        runs.append(
            _job(
                f"{fam}.s{seed}",
                mode,
                model,
                dataset,
                fam.split(".")[0],
                "slimming" if sparse else "none",
                seed,
                [],
                group,
                cut_class,
                160,
                0.1,
                "stepwise",
                "",
                slim_lambda=lam_of(model) if sparse else 0.0,
                precision=precision,
                hparams_hash=hsh,
            )
        )

    def add_cfg(cfg, model, dataset, method, modes, sparse, seed, group):
        cut = _cut_class(group, cfg, seed, modes[0] if modes else "SE")
        base_cut = "p2b-c100" if dataset == "cifar100" else f"base-s{seed}"
        add_base(model, dataset, seed, sparse, "base", base_cut)
        if not sparse and method == "unstructured":
            add_base(model, dataset, seed, False, "base", base_cut)
        base_id = f"{_family(model, dataset, sparse)}.s{seed}"
        prune_id = f"{cfg}.PRUNE.s{seed}"
        runs.append(
            _job(
                prune_id,
                "PRUNE",
                model,
                dataset,
                cfg,
                method,
                seed,
                [base_id],
                group,
                cut,
                0,
                0.0,
                "none",
                base_id,
                slim_lambda=lam_of(model),
                precision=precision,
                hparams_hash=hsh,
            )
        )
        for mode in modes:
            lr, schedule, epochs = _lr_schedule(mode)
            if mode == "FT":
                lr = lr_ft
            if mode == "SB":
                ratio = FLOP_RATIO_PRIOR[cfg]
                epochs, _ = scratch_b_epochs(1.0, ratio, kappa_max=2.0)
            run_id = f"{cfg}.{mode}.s{seed}"
            runs.append(
                _job(
                    run_id,
                    mode,
                    model,
                    dataset,
                    cfg,
                    method,
                    seed,
                    [prune_id],
                    group,
                    cut,
                    epochs,
                    lr,
                    schedule,
                    prune_id,
                    precision=precision,
                    hparams_hash=hsh,
                )
            )

    for seed in (1, 2, 3):
        for spec in _P1:
            add_cfg(*spec, seed, "P1")
        for spec in _P2A:
            add_cfg(*spec, seed, "P2a")
        for cfg, match, model, dataset in _UF:
            prune_id = f"{match}.PRUNE.s{seed}"
            cut = _cut_class("P2a", cfg, seed, "UF")
            runs.append(
                _job(
                    f"{cfg}.UF.s{seed}",
                    "UF",
                    model,
                    dataset,
                    cfg,
                    "uniform",
                    seed,
                    [prune_id],
                    "P2a",
                    cut,
                    160,
                    0.1,
                    "stepwise",
                    prune_id,
                    precision=precision,
                    hparams_hash=hsh,
                )
            )
        if seed == 1:
            for cfg, match in _GD:
                prune_id = f"{match}.PRUNE.s1"
                runs.append(
                    _job(
                        f"{cfg}.GD.s1",
                        "GD",
                        "vgg19",
                        "cifar10",
                        cfg,
                        "guided",
                        1,
                        [prune_id],
                        "P2b",
                        "p2b-gd",
                        160,
                        0.1,
                        "stepwise",
                        prune_id,
                        precision=precision,
                        hparams_hash=hsh,
                    )
                )
            add_cfg("UN-R56-80", "resnet56", "cifar10", "unstructured", ("FT", "SE", "SB"), False, 1, "P2b")
            for cfg in ("L1-V16A", "L1-R56A", "L1-R56B"):
                model = "vgg16" if "V16" in cfg else "resnet56"
                prune_id = f"{cfg}.PRUNE.s1"
                runs.append(
                    _job(
                        f"{cfg}.WT.s1",
                        "WT",
                        model,
                        "cifar10",
                        cfg,
                        "l1",
                        1,
                        [prune_id],
                        "P2b",
                        "p2b-wt",
                        160,
                        0.1,
                        "stepwise",
                        prune_id,
                        precision=precision,
                        hparams_hash=hsh,
                    )
                )
            ratio = FLOP_RATIO_PRIOR["SL-R56-80"]
            e_raw, _ = scratch_b_epochs(1.0, ratio, kappa_max=float("inf"))
            runs.append(
                _job(
                    "SL-R56-80.SBU.s1",
                    "SBU",
                    "resnet56",
                    "cifar10",
                    "SL-R56-80",
                    "slimming",
                    1,
                    ["SL-R56-80.PRUNE.s1"],
                    "P2b",
                    "p2b-sbu",
                    e_raw,
                    0.1,
                    "stepwise",
                    "SL-R56-80.PRUNE.s1",
                    precision=precision,
                    kappa_max=1e9,
                    hparams_hash=hsh,
                )
            )
            add_cfg(
                "SL-V19-50-C100",
                "vgg19",
                "cifar100",
                "slimming",
                ("FT", "SE", "SB"),
                True,
                1,
                "P2b",
            )
    return runs


def dev_phase_a() -> list[dict]:
    runs = []

    def add(**kwargs):
        runs.append(_job(**kwargs, hparams_hash="dev", split="dev", group="dev", cut_class="dev"))

    specs = [
        ("dev-V16.BASE.s0", "BASE", "vgg16", "cifar10", "V16", "none", 0.0),
        ("dev-V19.BASE.s0", "BASE", "vgg19", "cifar10", "V19", "none", 0.0),
        ("dev-R56.BASE.s0", "BASE", "resnet56", "cifar10", "R56", "none", 0.0),
        ("dev-V19C100.BASE.s0", "BASE", "vgg19", "cifar100", "V19C100", "none", 0.0),
    ]
    for run_id, mode, model, dataset, cfg, method, lam in specs:
        add(
            run_id=run_id,
            mode=mode,
            model=model,
            dataset=dataset,
            cfg_name=cfg,
            method=method,
            seed=0,
            deps=[],
            epochs=160,
            lr=0.1,
            lr_schedule="stepwise",
            dep_id="",
            slim_lambda=lam,
            precision="bf16",
        )
    add(
        run_id="dev-V16.BASE.fp32.s0",
        mode="BASE",
        model="vgg16",
        dataset="cifar10",
        cfg_name="V16",
        method="none",
        seed=0,
        deps=[],
        epochs=160,
        lr=0.1,
        lr_schedule="stepwise",
        dep_id="",
        precision="fp32",
    )
    for lam, tag in ((5e-5, "l5e-5"), (1e-4, "l1e-4")):
        add(
            run_id=f"dev-V19.BASES.{tag}.s0",
            mode="BASES",
            model="vgg19",
            dataset="cifar10",
            cfg_name="V19",
            method="slimming",
            seed=0,
            deps=[],
            epochs=160,
            lr=0.1,
            lr_schedule="stepwise",
            dep_id="",
            slim_lambda=lam,
        )
        add(
            run_id=f"dev-R56.BASES.{tag}.s0",
            mode="BASES",
            model="resnet56",
            dataset="cifar10",
            cfg_name="R56",
            method="slimming",
            seed=0,
            deps=[],
            epochs=160,
            lr=0.1,
            lr_schedule="stepwise",
            dep_id="",
            slim_lambda=lam,
        )
        add(
            run_id=f"dev-SL-V19-70.PRUNE.{tag}.s0",
            mode="PRUNE",
            model="vgg19",
            dataset="cifar10",
            cfg_name="SL-V19-70",
            method="slimming",
            seed=0,
            deps=[f"dev-V19.BASES.{tag}.s0"],
            epochs=0,
            lr=0.0,
            lr_schedule="none",
            dep_id=f"dev-V19.BASES.{tag}.s0",
            slim_lambda=lam,
        )
        add(
            run_id=f"dev-SL-R56-80.PRUNE.{tag}.s0",
            mode="PRUNE",
            model="resnet56",
            dataset="cifar10",
            cfg_name="SL-R56-80",
            method="slimming",
            seed=0,
            deps=[f"dev-R56.BASES.{tag}.s0"],
            epochs=0,
            lr=0.0,
            lr_schedule="none",
            dep_id=f"dev-R56.BASES.{tag}.s0",
            slim_lambda=lam,
        )
    add(
        run_id="dev-L1-R56B.PRUNE.s0",
        mode="PRUNE",
        model="resnet56",
        dataset="cifar10",
        cfg_name="L1-R56B",
        method="l1",
        seed=0,
        deps=["dev-R56.BASE.s0"],
        epochs=0,
        lr=0.0,
        lr_schedule="none",
        dep_id="dev-R56.BASE.s0",
    )
    return runs


def dev_phase_b(vgg_tag: str) -> list[dict]:
    runs = []
    prune_l1 = "dev-L1-R56B.PRUNE.s0"
    prune_sl = f"dev-SL-V19-70.PRUNE.{vgg_tag}.s0"
    for lr, tag in ((0.001, "lr0001"), (0.01, "lr001")):
        for run_id, model, cfg, method, dep in (
            (f"dev-L1-R56B.FT.{tag}.s0", "resnet56", "L1-R56B", "l1", prune_l1),
            (f"dev-SL-V19-70.FT.{tag}.s0", "vgg19", "SL-V19-70", "slimming", prune_sl),
        ):
            runs.append(
                _job(
                    run_id,
                    "FT",
                    model,
                    "cifar10",
                    cfg,
                    method,
                    0,
                    [dep],
                    "dev",
                    "dev",
                    40,
                    lr,
                    "constant",
                    dep,
                    split="dev",
                    hparams_hash="dev",
                )
            )
    return runs


def canary_runs(hparams: dict) -> list[dict]:
    """Short end-to-end of every mode. Epoch counts are forced, not the real Scratch-B length."""
    precision = hparams["precision"]
    lr_ft = float(hparams["lr_ft"])
    lam_vgg = float(hparams["slim_lambda_vgg"])
    lam_res = float(hparams["slim_lambda_resnet"])
    hsh = hparams["hash"]
    runs = []

    def add(run_id, mode, model, dataset, cfg, method, deps, epochs, lr, schedule, dep, lam=0.0, force=False):
        runs.append(
            _job(
                run_id,
                mode,
                model,
                dataset,
                cfg,
                method,
                0,
                deps,
                "canary",
                "canary",
                epochs,
                lr,
                schedule,
                dep,
                slim_lambda=lam,
                precision=precision,
                split="dev",
                hparams_hash=hsh,
                force_epochs=force,
            )
        )

    add("canary-R56.BASE.s0", "BASE", "resnet56", "cifar10", "R56", "none", [], 4, 0.1, "stepwise", "")
    add("canary-V19.BASE.s0", "BASE", "vgg19", "cifar10", "V19", "none", [], 4, 0.1, "stepwise", "")
    add(
        "canary-V19.BASES.s0",
        "BASES",
        "vgg19",
        "cifar10",
        "V19",
        "slimming",
        [],
        4,
        0.1,
        "stepwise",
        "",
        lam_vgg,
    )
    add(
        "canary-L1-R56B.PRUNE.s0",
        "PRUNE",
        "resnet56",
        "cifar10",
        "L1-R56B",
        "l1",
        ["canary-R56.BASE.s0"],
        0,
        0.0,
        "none",
        "canary-R56.BASE.s0",
    )
    add(
        "canary-SL-V19-70.PRUNE.s0",
        "PRUNE",
        "vgg19",
        "cifar10",
        "SL-V19-70",
        "slimming",
        ["canary-V19.BASES.s0"],
        0,
        0.0,
        "none",
        "canary-V19.BASES.s0",
        lam_vgg,
    )
    add(
        "canary-UN-V19-80.PRUNE.s0",
        "PRUNE",
        "vgg19",
        "cifar10",
        "UN-V19-80",
        "unstructured",
        ["canary-V19.BASE.s0"],
        0,
        0.0,
        "none",
        "canary-V19.BASE.s0",
    )
    add(
        "canary-L1-R56B.FT.s0",
        "FT",
        "resnet56",
        "cifar10",
        "L1-R56B",
        "l1",
        ["canary-L1-R56B.PRUNE.s0"],
        1,
        lr_ft,
        "constant",
        "canary-L1-R56B.PRUNE.s0",
        force=True,
    )
    add(
        "canary-L1-R56B.SE.s0",
        "SE",
        "resnet56",
        "cifar10",
        "L1-R56B",
        "l1",
        ["canary-L1-R56B.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-L1-R56B.PRUNE.s0",
        force=True,
    )
    add(
        "canary-L1-R56B.SB.s0",
        "SB",
        "resnet56",
        "cifar10",
        "L1-R56B",
        "l1",
        ["canary-L1-R56B.PRUNE.s0"],
        8,
        0.1,
        "stepwise",
        "canary-L1-R56B.PRUNE.s0",
        force=True,
    )
    add(
        "canary-L1-R56B.WT.s0",
        "WT",
        "resnet56",
        "cifar10",
        "L1-R56B",
        "l1",
        ["canary-L1-R56B.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-L1-R56B.PRUNE.s0",
        force=True,
    )
    add(
        "canary-SL-V19-70.FT.s0",
        "FT",
        "vgg19",
        "cifar10",
        "SL-V19-70",
        "slimming",
        ["canary-SL-V19-70.PRUNE.s0"],
        1,
        lr_ft,
        "constant",
        "canary-SL-V19-70.PRUNE.s0",
        force=True,
    )
    add(
        "canary-SL-V19-70.SE.s0",
        "SE",
        "vgg19",
        "cifar10",
        "SL-V19-70",
        "slimming",
        ["canary-SL-V19-70.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-SL-V19-70.PRUNE.s0",
        force=True,
    )
    add(
        "canary-SL-V19-70.SB.s0",
        "SB",
        "vgg19",
        "cifar10",
        "SL-V19-70",
        "slimming",
        ["canary-SL-V19-70.PRUNE.s0"],
        8,
        0.1,
        "stepwise",
        "canary-SL-V19-70.PRUNE.s0",
        force=True,
    )
    add(
        "canary-UN-V19-80.FT.s0",
        "FT",
        "vgg19",
        "cifar10",
        "UN-V19-80",
        "unstructured",
        ["canary-UN-V19-80.PRUNE.s0"],
        1,
        lr_ft,
        "constant",
        "canary-UN-V19-80.PRUNE.s0",
        force=True,
    )
    add(
        "canary-UN-V19-80.SE.s0",
        "SE",
        "vgg19",
        "cifar10",
        "UN-V19-80",
        "unstructured",
        ["canary-UN-V19-80.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-UN-V19-80.PRUNE.s0",
        force=True,
    )
    add(
        "canary-UN-V19-80.SB.s0",
        "SB",
        "vgg19",
        "cifar10",
        "UN-V19-80",
        "unstructured",
        ["canary-UN-V19-80.PRUNE.s0"],
        8,
        0.1,
        "stepwise",
        "canary-UN-V19-80.PRUNE.s0",
        force=True,
    )
    add(
        "canary-UF-V19-70.UF.s0",
        "UF",
        "vgg19",
        "cifar10",
        "UF-V19-70",
        "uniform",
        ["canary-SL-V19-70.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-SL-V19-70.PRUNE.s0",
        force=True,
    )
    add(
        "canary-GD-V19-70.GD.s0",
        "GD",
        "vgg19",
        "cifar10",
        "GD-V19-70",
        "guided",
        ["canary-SL-V19-70.PRUNE.s0"],
        4,
        0.1,
        "stepwise",
        "canary-SL-V19-70.PRUNE.s0",
        force=True,
    )
    # lam_res is part of the frozen payload; referenced so a mismatch is obvious in review.
    _ = lam_res
    return runs
