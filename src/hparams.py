import hashlib
import json
from pathlib import Path

from src.paths import FROZEN_HPARAMS

PAPER_ACC = {
    ("vgg16", "cifar10"): 93.63,
    ("resnet56", "cifar10"): 93.14,
    ("vgg19", "cifar10"): 93.53,
    ("vgg19", "cifar100"): 72.63,
}


def canonical_payload(precision: str, lr_ft: float, slim_lambda_vgg: float, slim_lambda_resnet: float) -> dict:
    return {
        "augment": "crop4_hflip_persample",
        "base_lr": 0.1,
        "batch_size": 128,
        "epochs_base": 160,
        "epochs_ft": 40,
        "kappa_max": 2,
        "lr_ft": float(lr_ft),
        "momentum": 0.9,
        "optimizer": "sgd_nesterov",
        "precision": precision,
        "slim_lambda_resnet": float(slim_lambda_resnet),
        "slim_lambda_vgg": float(slim_lambda_vgg),
        "weight_decay": 1e-4,
    }


def payload_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def write_frozen(path: Path, precision: str, lr_ft: float, slim_lambda_vgg: float, slim_lambda_resnet: float) -> dict:
    payload = canonical_payload(precision, lr_ft, slim_lambda_vgg, slim_lambda_resnet)
    payload["hash"] = payload_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_frozen(path: Path | None = None) -> dict | None:
    path = path or FROZEN_HPARAMS
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def slim_lambda_for(frozen: dict, model: str) -> float:
    if model.startswith("resnet"):
        return float(frozen["slim_lambda_resnet"])
    return float(frozen["slim_lambda_vgg"])
