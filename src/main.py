"""Hydra entry point for one run. Queue workers call this as a subprocess."""

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from src.engine import execute


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg) -> None:
    job = OmegaConf.to_container(cfg, resolve=True)
    job["tracking"] = _as_bool(job["tracking"])
    job["do_eval"] = _as_bool(job["do_eval"])
    job["force_epochs"] = _as_bool(job["force_epochs"])
    job["seed"] = int(job["seed"])
    job["epochs"] = int(job["epochs"])
    job["lr"] = float(job["lr"])
    job["slim_lambda"] = float(job["slim_lambda"])
    job["kappa_max"] = float(job["kappa_max"])
    execute(job)


if __name__ == "__main__":
    main()
