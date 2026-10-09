"""Read result.json files and write the tables the report is allowed to quote."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from src.models import DEFAULT_WIDTHS, VGG_STAGES, default_resnet_internals
from src.paths import OUTPUT_ROOT, ROOT
from src.serialize import atomic_json, read_json

H1_CFGS = ["L1-V16A", "L1-R56A", "L1-R56B", "SL-V19-70", "SL-R56-60"]
H1_L1 = ["L1-V16A", "L1-R56A", "L1-R56B"]
SL_R56 = ["SL-R56-40", "SL-R56-60", "SL-R56-80"]
VGG_POINTS = ["30", "50", "70"]
RES_POINTS = ["40", "60", "80"]

TRAIN_MODES = {"BASE", "BASES", "FT", "SE", "SB", "SBU", "UF", "GD", "WT"}


def load_results(root: Path | None = None) -> list[dict]:
    root = root or OUTPUT_ROOT
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/result.json")):
        payload = read_json(path)
        payload["_dir"] = str(path.parent)
        rows.append(payload)
    return rows


def _acc(row: dict) -> float | None:
    value = row.get("test_acc_last")
    if value is None:
        return None
    return float(value)


def index_full(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    out = {}
    for row in rows:
        if row.get("split") != "full":
            continue
        if row.get("mode") not in TRAIN_MODES | {"PRUNE"}:
            continue
        out[(row["cfg_name"], row["mode"], int(row["seed"]))] = row
    return out


def _paired(idx, cfg: str, left: str, right: str) -> list[tuple[int, float]]:
    seeds = sorted({seed for (c, mode, seed) in idx if c == cfg and mode in {left, right}})
    diffs = []
    for seed in seeds:
        a = idx.get((cfg, left, seed))
        b = idx.get((cfg, right, seed))
        if a is None or b is None:
            continue
        av, bv = _acc(a), _acc(b)
        if av is None or bv is None:
            continue
        diffs.append((seed, av - bv))
    return diffs


def _mean(diffs: list[tuple[int, float]]) -> float | None:
    if not diffs:
        return None
    return sum(d for _, d in diffs) / len(diffs)


def _fmt_mean_std(values: list[float]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{statistics.mean(values):.2f}±{statistics.stdev(values):.2f}"


def stage_keep(row: dict) -> dict | None:
    arch = row.get("arch") or {}
    kind = arch.get("kind")
    if kind in VGG_STAGES and "widths" in arch:
        full = DEFAULT_WIDTHS[kind]
        rates = []
        for stage in VGG_STAGES[kind]:
            kept = sum(arch["widths"][i] for i in stage)
            base = sum(full[i] for i in stage)
            rates.append(kept / base)
        return {"rates": rates, "later_more_pruned": rates[-1] < rates[0]}
    if kind == "resnet56" and "internals" in arch:
        full = default_resnet_internals()
        rates = []
        for stage in range(3):
            sl = slice(stage * 9, (stage + 1) * 9)
            kept = sum(arch["internals"][sl])
            base = sum(full[sl])
            rates.append(kept / base)
        return {"rates": rates, "range": max(rates) - min(rates)}
    return None


def hypotheses(idx) -> dict:
    h1_rows = []
    sb_ok = se_ok = 0
    present = 0
    refute = 0
    for cfg in H1_CFGS:
        sb = _mean(_paired(idx, cfg, "SB", "FT"))
        se = _mean(_paired(idx, cfg, "SE", "FT"))
        h1_rows.append({"cfg": cfg, "sb_minus_ft": sb, "se_minus_ft": se})
        if sb is None:
            continue
        present += 1
        if sb >= -0.2:
            sb_ok += 1
        if sb < -0.5:
            refute += 1
        if se is not None and se >= -0.3:
            se_ok += 1
    if present < 5:
        h1 = "не проверено"
    elif refute >= 2:
        h1 = "опровергнута"
    elif sb_ok >= 4 and se_ok >= 3:
        h1 = "подтверждена"
    else:
        h1 = "неоднозначно"

    wt_rows = []
    wt_ok = wt_present = 0
    for cfg in H1_L1:
        diff = _mean(_paired(idx, cfg, "WT", "SE"))
        wt_rows.append({"cfg": cfg, "wt_minus_se": diff})
        if diff is None:
            continue
        wt_present += 1
        if abs(diff) < 0.3:
            wt_ok += 1
    if wt_present < 2:
        h1c = "не проверено"
    elif wt_ok >= 2:
        h1c = "подтверждена"
    else:
        h1c = "опровергнута"

    eb_rows = []
    eb_hits = eb_total = 0
    for (cfg, mode, seed), row in idx.items():
        if mode != "SB" or row.get("E_B") is None:
            continue
        if int(row["E_B"]) <= 160:
            continue
        se = idx.get((cfg, "SE", seed))
        if se is None or _acc(se) is None or _acc(row) is None:
            continue
        diff = _acc(row) - _acc(se)
        eb_rows.append({"cfg": cfg, "seed": seed, "sb_minus_se": diff, "E_B": row["E_B"]})
        eb_total += 1
        if diff >= 0.1:
            eb_hits += 1
    if eb_total == 0:
        h2_budget = "не проверено"
    elif eb_hits / eb_total >= 0.70:
        h2_budget = "подтверждена"
    else:
        h2_budget = "опровергнута"

    sl_curve = []
    for cfg in SL_R56:
        sl_curve.append({"cfg": cfg, "sb_minus_ft": _mean(_paired(idx, cfg, "SB", "FT"))})
    values = [row["sb_minus_ft"] for row in sl_curve]
    if any(v is None for v in values):
        h2_curve = "не проверено"
    else:
        monotone = values[0] <= values[1] <= values[2] and values[2] >= 1.0
        h2_curve = "подтверждена" if monotone else "опровергнута"
    sbu = _mean(_paired(idx, "SL-R56-80", "SBU", "FT"))

    vgg_uf = []
    for point in VGG_POINTS:
        seeds = sorted({s for (c, m, s) in idx if c == f"UF-V19-{point}" and m == "UF"})
        diffs = []
        for seed in seeds:
            uf = idx.get((f"UF-V19-{point}", "UF", seed))
            sl = idx.get((f"SL-V19-{point}", "SE", seed))
            if uf is None or sl is None or _acc(uf) is None or _acc(sl) is None:
                continue
            diffs.append(_acc(sl) - _acc(uf))
        vgg_uf.append({"point": point, "sl_minus_uf": (sum(diffs) / len(diffs)) if diffs else None})
    vgg_known = [r for r in vgg_uf if r["sl_minus_uf"] is not None]
    if len(vgg_known) < 3:
        h3_vgg = "не проверено"
    elif sum(1 for r in vgg_known if r["sl_minus_uf"] >= 0.2) >= 2:
        h3_vgg = "подтверждена"
    else:
        h3_vgg = "опровергнута"

    gd_rows = []
    for point in VGG_POINTS:
        gd = idx.get((f"GD-V19-{point}", "GD", 1))
        uf = idx.get((f"UF-V19-{point}", "UF", 1))
        sl = idx.get((f"SL-V19-{point}", "SE", 1))
        if any(x is None or _acc(x) is None for x in (gd, uf, sl)):
            gd_rows.append({"point": point, "gd_minus_uf": None, "gd_minus_sl": None})
            continue
        gd_rows.append(
            {
                "point": point,
                "gd_minus_uf": _acc(gd) - _acc(uf),
                "gd_minus_sl": _acc(gd) - _acc(sl),
            }
        )
    gd_known = [r for r in gd_rows if r["gd_minus_uf"] is not None]
    if len(gd_known) < 3:
        h3_gd = "не проверено"
    elif sum(1 for r in gd_known if r["gd_minus_uf"] >= 0.1 and abs(r["gd_minus_sl"]) < 0.3) >= 2:
        h3_gd = "подтверждена"
    else:
        h3_gd = "опровергнута"

    res_rows = []
    for point in RES_POINTS:
        seeds = sorted({s for (c, m, s) in idx if c == f"UF-R56-{point}" and m == "UF"})
        diffs = []
        for seed in seeds:
            uf = idx.get((f"UF-R56-{point}", "UF", seed))
            sl = idx.get((f"SL-R56-{point}", "SE", seed))
            if uf is None or sl is None or _acc(uf) is None or _acc(sl) is None:
                continue
            diffs.append(abs(_acc(sl) - _acc(uf)))
        res_rows.append({"point": point, "abs_sl_minus_uf": (sum(diffs) / len(diffs)) if diffs else None})
    res_known = [r for r in res_rows if r["abs_sl_minus_uf"] is not None]
    if len(res_known) < 3:
        h3_res = "не проверено"
    elif sum(1 for r in res_known if r["abs_sl_minus_uf"] < 0.3) >= 2:
        h3_res = "подтверждена"
    else:
        h3_res = "опровергнута"

    un80 = _mean(_paired(idx, "UN-V19-80", "SB", "FT"))
    un95 = _mean(_paired(idx, "UN-R56-95", "SB", "FT"))
    if un80 is None or un95 is None:
        h4 = "не проверено"
    elif un80 >= -0.3 and un95 <= -0.5:
        h4 = "подтверждена"
    else:
        h4 = "опровергнута"

    return {
        "H1": h1,
        "H1_rows": h1_rows,
        "H1c": h1c,
        "H1c_rows": wt_rows,
        "H2_budget": h2_budget,
        "H2_budget_hits": eb_hits,
        "H2_budget_total": eb_total,
        "H2_curve": h2_curve,
        "H2_curve_rows": sl_curve,
        "H2_sbu_minus_ft": sbu,
        "H3_vgg": h3_vgg,
        "H3_vgg_rows": vgg_uf,
        "H3_gd": h3_gd,
        "H3_gd_rows": gd_rows,
        "H3_resnet": h3_res,
        "H3_resnet_rows": res_rows,
        "H4": h4,
        "H4_un80_sb_minus_ft": un80,
        "H4_un95_sb_minus_ft": un95,
    }


def validate(rows: list[dict]) -> list[str]:
    errors = []
    full = [r for r in rows if r.get("split") == "full" and r.get("mode") in TRAIN_MODES]
    hashes = {r.get("hparams_hash") for r in full}
    hashes.discard(None)
    if len(hashes) > 1:
        errors.append(f"смешаны хэши гиперпараметров: {sorted(hashes)}")
    by_key = defaultdict(list)
    for row in full:
        by_key[(row["cfg_name"], row["mode"])].append(row)
        if row.get("epochs_done") != row.get("epochs_planned"):
            errors.append(f"{row['run_id']}: epochs_done != epochs_planned")
        if row.get("n_test_evals") != 1:
            errors.append(f"{row['run_id']}: n_test_evals={row.get('n_test_evals')}")
        schedule = row.get("lr_schedule")
        epochs = int(row.get("epochs_planned") or 0)
        miles = row.get("milestones")
        if schedule == "constant" and miles not in ([], None):
            errors.append(f"{row['run_id']}: FT milestones {miles}")
        if schedule == "stepwise" and epochs:
            expect = [epochs // 2, round(0.75 * epochs)]
            if miles != expect:
                errors.append(f"{row['run_id']}: milestones {miles} != {expect}")
        if row["mode"] == "SB":
            if row.get("E_B") is not None and int(row["epochs_done"]) != int(row["E_B"]):
                errors.append(f"{row['run_id']}: epochs {row['epochs_done']} != E_B {row['E_B']}")
        if row["mode"] == "SBU" and row.get("E_B") is not None and int(row["epochs_done"]) != int(row["E_B"]):
            errors.append(f"{row['run_id']}: SBU epochs != E_B_uncapped")
        if any(str(key).startswith("best_") for key in row):
            errors.append(f"{row['run_id']}: checkpoint was chosen by a best-field")
    for (cfg, mode), group in by_key.items():
        prints = [r.get("init_fingerprint") for r in group]
        if len(prints) >= 2 and len(set(prints)) < len(prints):
            errors.append(f"{cfg} {mode}: одинаковые init fingerprint у разных сидов")
    for cfg in {r["cfg_name"] for r in full}:
        for seed in {int(r["seed"]) for r in full if r["cfg_name"] == cfg}:
            se = next((r for r in full if r["cfg_name"] == cfg and r["mode"] == "SE" and int(r["seed"]) == seed), None)
            ft = next((r for r in full if r["cfg_name"] == cfg and r["mode"] == "FT" and int(r["seed"]) == seed), None)
            if se and ft and se.get("init_fingerprint") == ft.get("init_fingerprint"):
                errors.append(f"{cfg} s{seed}: SE и FT стартуют с одного fingerprint")
    return errors


def _write_csv(path: Path, header: list[str], records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def build(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (ROOT / "report" / "tables")
    rows = load_results()
    full = [r for r in rows if r.get("split") == "full"]
    idx = index_full(rows)
    summary = []
    grouped = defaultdict(list)
    for row in full:
        if row.get("mode") not in TRAIN_MODES:
            continue
        grouped[(row["cfg_name"], row["mode"], row["dataset"])].append(row)
    for (cfg, mode, dataset), group in sorted(grouped.items()):
        accs = [a for a in (_acc(r) for r in group) if a is not None]
        summary.append(
            {
                "cfg": cfg,
                "mode": mode,
                "dataset": dataset,
                "n": len(accs),
                "test_acc": _fmt_mean_std(accs),
                "test_acc_mean": "" if not accs else f"{statistics.mean(accs):.4f}",
                "params": group[0].get("params"),
                "macs": group[0].get("macs"),
            }
        )
    runs_out = []
    for row in full:
        runs_out.append(
            {
                "run_id": row.get("run_id"),
                "cfg": row.get("cfg_name"),
                "mode": row.get("mode"),
                "seed": row.get("seed"),
                "dataset": row.get("dataset"),
                "test_acc": row.get("test_acc_last"),
                "params": row.get("params"),
                "macs": row.get("macs"),
                "epochs": row.get("epochs_done"),
                "E_B": row.get("E_B", ""),
                "hparams_hash": row.get("hparams_hash"),
            }
        )
    keeps = []
    for row in full:
        if row.get("mode") != "SE" or int(row.get("seed", 0)) != 1:
            continue
        info = stage_keep(row)
        if not info:
            continue
        keeps.append({"cfg": row["cfg_name"], "rates": " ".join(f"{v:.3f}" for v in info["rates"]), **{k: v for k, v in info.items() if k != "rates"}})
    hyp = hypotheses(idx)
    errors = validate(rows)
    _write_csv(out_dir / "runs.csv", ["run_id", "cfg", "mode", "seed", "dataset", "test_acc", "params", "macs", "epochs", "E_B", "hparams_hash"], runs_out)
    _write_csv(out_dir / "summary.csv", ["cfg", "mode", "dataset", "n", "test_acc", "test_acc_mean", "params", "macs"], summary)
    atomic_json(out_dir / "hypotheses.json", hyp)
    atomic_json(out_dir / "stage_keep.json", keeps)
    atomic_json(out_dir / "invariants.json", {"ok": not errors, "errors": errors})
    return {"hypotheses": hyp, "errors": errors, "n_full": len(full)}
