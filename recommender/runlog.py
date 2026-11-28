"""Save training logs as JSON, for the submitted logs and configurations.

A saved log keeps everything a training function returns (optimizer, seed,
configuration, status, per-update traces and every checkpoint's work, time,
losses and metrics) except the parameter arrays, which are large and are kept
only for frozen models (``recommender.frozen``). ``selected_work`` records the
checkpoint chosen by ordinary-validation NDCG@10, when metrics are present.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recommender.evaluate import select_checkpoint

LOG_DIR = Path(__file__).resolve().parent.parent / "results" / "logs"


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def log_record(log: dict, **extra) -> dict:
    """JSON-ready copy of a training log without parameter arrays."""
    record = {k: v for k, v in log.items() if k != "checkpoints"}
    record["checkpoints"] = [{k: v for k, v in c.items() if k != "params"}
                             for c in log["checkpoints"]]
    if log["checkpoints"] and "val_ndcg" in log["checkpoints"][0]:
        record["selected_work"] = select_checkpoint(log)["work"]
    record.update(extra)
    return _plain(record)


def save_log(log: dict, name: str, directory: Path | str = LOG_DIR, **extra) -> Path:
    """Write ``<directory>/<name>.json``; ``extra`` fields (for example the
    experiment name) are stored alongside. Non-finite numbers become null."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(log_record(log, **extra)))
    return path


def load_log(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
