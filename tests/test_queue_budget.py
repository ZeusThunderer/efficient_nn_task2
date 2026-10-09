import threading
import time
from pathlib import Path

from src.budget import allowance_hours, phase_of, select_enabled
from src.catalog import final_runs
from src.queue import run_queue, write_queue


def test_budget_cuts_match_plan_scenarios():
    runs = final_runs()
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids))

    enabled, cost = select_enabled(runs, allowance_hours(1.0))
    assert cost <= allowance_hours(1.0) + 1e-6
    kept = {r["cut_class"] for r in runs if enabled[r["run_id"]]}
    assert "p2a-s2" not in kept
    assert "p2a-s3" not in kept
    assert "p2b-c100" not in kept
    assert "p2b-wt" not in kept
    assert "p1-keep" in kept
    assert "p2a-s1" in kept
    # Seed-1 and seed-2 of the main comparison stay.
    assert enabled["L1-R56B.SB.s1"]
    assert enabled["L1-R56B.SB.s2"]
    assert not enabled["L1-R56B.SB.s3"]

    enabled14, cost14 = select_enabled(runs, allowance_hours(1.4))
    assert cost14 <= allowance_hours(1.4) + 1e-6
    assert enabled14["SL-R56-80.SBU.s1"]
    assert enabled14["L1-V16A.WT.s1"]
    assert enabled14["GD-V19-70.GD.s1"]
    assert not enabled14["UN-V19-80.SB.s2"]


def test_phase_order():
    runs = {r["run_id"]: r for r in final_runs()}
    assert phase_of(runs["V16.BASE.s1"]) == 1
    assert phase_of(runs["L1-R56B.SB.s1"]) == 1
    assert phase_of(runs["SL-V19-30.SE.s1"]) == 1
    assert phase_of(runs["L1-R56B.SB.s2"]) == 2
    assert phase_of(runs["SL-R56-80.SBU.s1"]) == 3
    assert phase_of(runs["UN-V19-80.SB.s2"]) == 3


def test_queue_respects_dependencies(tmp_path: Path):
    runs = []
    for run_id, deps, priority in (
        ("A", [], 1),
        ("B", ["A"], 2),
        ("C", ["A"], 2),
        ("D", ["B", "C"], 3),
    ):
        runs.append(
            {
                "run_id": run_id,
                "enabled": True,
                "deps": deps,
                "priority": priority,
                "epochs": 1,
                "group": "test",
                "cut_class": "keep",
                "overrides": {"run_id": run_id},
            }
        )
    queue_path = tmp_path / "queue.json"
    write_queue(queue_path, runs)
    order = []
    lock = threading.Lock()

    def executor(run):
        with lock:
            order.append(run["run_id"])
        time.sleep(0.05)
        dest = tmp_path / "outputs" / run["run_id"]
        dest.mkdir(parents=True)
        (dest / "result.json").write_text("{}", encoding="utf-8")

    run_queue(queue_path, k=3, executor=executor, output_root=tmp_path / "outputs")
    assert order.count("A") == 1
    assert set(order) == {"A", "B", "C", "D"}
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
