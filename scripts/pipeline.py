"""Dev split, freeze, canary, three budget phases, then the report inputs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.campaign import frozen_or_die, run_canary, run_dev, run_final  # noqa: E402
from src.select_hparams import build_summary  # noqa: E402


def _tag() -> None:
    tags = subprocess.check_output(["git", "tag", "--list", "hparams-v1"], cwd=ROOT, text=True).strip()
    if tags == "hparams-v1":
        print("tag hparams-v1 already exists", flush=True)
        return
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Freeze hyperparameters after the dev split."],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["git", "tag", "hparams-v1"], cwd=ROOT, check=True)
    print("tagged hparams-v1", flush=True)


def main() -> None:
    run_dev()
    summary = build_summary()
    print(summary, flush=True)
    _tag()
    hparams = frozen_or_die()
    run_canary(hparams)
    run_final(hparams)
    print("waves finished", flush=True)


if __name__ == "__main__":
    main()
