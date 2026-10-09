"""Local Aim tracker. Failures are logged and never stop training."""

from __future__ import annotations

from pathlib import Path


class Tracker:
    def __init__(self, enabled: bool, repo: Path, run_id: str, experiment: str, tags: dict, hparams: dict):
        self.enabled = enabled
        self.run = None
        self.error = None
        if not enabled:
            return
        try:
            from aim import Run

            self.run = Run(
                repo=str(repo),
                experiment=experiment,
                system_tracking_interval=None,
                log_system_params=False,
                capture_terminal_logs=False,
            )
            self.run.name = run_id
            for key, value in tags.items():
                self.run.add_tag(f"{key}:{value}")
            for key, value in hparams.items():
                self.run.set(key, value, strict=False)
        except Exception as exc:  # noqa: BLE001 — tracking must not block training
            self.error = repr(exc)
            self.run = None
            self.enabled = False

    def track(self, name: str, value: float, step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.track(float(value), name=name, step=int(step))
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            self.run = None

    def track_distribution(self, name: str, values, step: int) -> None:
        if self.run is None:
            return
        try:
            from aim import Distribution

            self.run.track(Distribution(values), name=name, step=int(step))
        except Exception as exc:  # noqa: BLE001
            self.error = repr(exc)
            self.run = None

    def close(self) -> None:
        if self.run is None:
            return
        try:
            self.run.close()
        except Exception:  # noqa: BLE001
            pass
        self.run = None
