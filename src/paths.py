from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
OUTPUT_ROOT = ROOT / "outputs"
CONF_ROOT = ROOT / "conf"
BUDGET_ROOT = ROOT / "budget"
FROZEN_HPARAMS = CONF_ROOT / "frozen_hparams.json"
