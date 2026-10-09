import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.tracking import Tracker


def _losses(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line)["train_loss"])
    return rows


def test_resume_after_kill(tmp_path: Path):
    env_base = os.environ.copy()
    env_base["PRUNE_DEVICE"] = "cpu"
    env_base["OMP_NUM_THREADS"] = "1"
    env_base["CHECKPOINT_EVERY"] = "1"
    ref = tmp_path / "ref"
    ref.mkdir()
    env_ref = dict(env_base)
    env_ref["RUN_DIR"] = str(ref)
    subprocess.run([sys.executable, "-m", "src.tiny_resume"], cwd=Path.cwd(), env=env_ref, check=True)

    killed = tmp_path / "killed"
    killed.mkdir()
    env_kill = dict(env_base)
    env_kill["RUN_DIR"] = str(killed)
    env_kill["INJECT_SLEEP_EPOCH"] = "1"
    env_kill["INJECT_SLEEP_S"] = "60"
    proc = subprocess.Popen([sys.executable, "-m", "src.tiny_resume"], cwd=Path.cwd(), env=env_kill)
    history = killed / "history.jsonl"
    deadline = time.time() + 120
    while time.time() < deadline:
        if history.exists() and "\"epoch\": 0" in history.read_text(encoding="utf-8"):
            break
        time.sleep(0.2)
    else:
        proc.kill()
        raise AssertionError("epoch 0 checkpoint did not appear")
    time.sleep(0.5)
    os.kill(proc.pid, 9)
    proc.wait(timeout=30)

    env_resume = dict(env_base)
    env_resume["RUN_DIR"] = str(killed)
    subprocess.run([sys.executable, "-m", "src.tiny_resume"], cwd=Path.cwd(), env=env_resume, check=True)
    assert _losses(ref / "history.jsonl") == _losses(killed / "history.jsonl")


def test_aim_track_and_disable(tmp_path: Path):
    tracker = Tracker(True, tmp_path / ".aim", "unit", "exp", {"seed": "0"}, {"lr": 0.1})
    assert tracker.run is not None, tracker.error
    tracker.track("train_loss", 1.5, 0)
    tracker.close()
    disabled = Tracker(False, tmp_path / ".aim", "off", "exp", {}, {})
    assert disabled.run is None
    disabled.track("train_loss", 1.0, 0)
    disabled.close()
