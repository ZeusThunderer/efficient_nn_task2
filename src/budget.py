"""GPU-hour estimates and the cut order from the plan."""

from __future__ import annotations

from src.catalog import (
    C_FIX,
    CUT_ORDER,
    FLOP_RATIO_PRIOR,
    RHO,
    T_DEV_PRIOR,
    T_EP_PRIOR,
    T_LAT_PRIOR,
    U_HOURS,
)


def allowance_hours(phi: float, t_dev: float = T_DEV_PRIOR, t_lat: float = T_LAT_PRIOR) -> float:
    return (U_HOURS - t_dev - t_lat) * phi


def estimate_hours(run: dict, t_ep: dict | None = None, ratios: dict | None = None) -> float:
    t_ep = t_ep or T_EP_PRIOR
    ratios = ratios or FLOP_RATIO_PRIOR
    mode = run["overrides"]["mode"]
    model = run["overrides"]["model"]
    cfg = run["overrides"]["cfg_name"]
    epochs = int(run["epochs"])
    if mode == "PRUNE" or epochs <= 0:
        return C_FIX
    base = float(t_ep[model])
    if mode in {"BASE", "BASES"}:
        seconds = base
    elif run["overrides"]["method"] == "unstructured":
        seconds = 1.05 * base
    else:
        ratio = float(ratios.get(cfg, 1.0))
        seconds = base * max(ratio, RHO[model])
    return epochs * seconds / 3600.0 + C_FIX


def select_enabled(
    runs: list[dict],
    allowance: float,
    t_ep: dict | None = None,
    actual_hours: dict | None = None,
    cost_fn=None,
):
    """Disable cut-classes from the tail until the set fits. Completed runs stay."""
    actual_hours = actual_hours or {}
    enabled = {run["run_id"]: True for run in runs}
    protected = set(actual_hours)

    def one(run: dict) -> float:
        if run["run_id"] in actual_hours:
            return actual_hours[run["run_id"]]
        if cost_fn is not None:
            return float(cost_fn(run))
        return estimate_hours(run, t_ep)

    def total() -> float:
        return sum(one(run) for run in runs if enabled[run["run_id"]])

    for cut in CUT_ORDER:
        if total() <= allowance + 1e-9:
            break
        for run in runs:
            if run["cut_class"] == cut and run["run_id"] not in protected:
                enabled[run["run_id"]] = False

    changed = True
    while changed:
        changed = False
        referenced = set()
        for run in runs:
            if enabled[run["run_id"]]:
                referenced.update(run["deps"])
        for run in runs:
            mode = run["overrides"]["mode"]
            if run["run_id"] in protected or not enabled[run["run_id"]]:
                continue
            if mode in {"BASE", "BASES", "PRUNE"} and run["run_id"] not in referenced:
                enabled[run["run_id"]] = False
                changed = True
    return enabled, total()


def phase_of(run: dict) -> int:
    cut = run["cut_class"]
    seed = int(run["overrides"]["seed"])
    if seed == 1 and cut in {"base-s1", "p1-keep", "p2a-s1"}:
        return 1
    if seed >= 2 and (cut.startswith("p1") or cut.startswith("base-")):
        return 2
    return 3
