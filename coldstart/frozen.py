"""Freezing selected checkpoints and evaluating them afterwards.

Test integrity (task distribution, rule 6): every configuration and checkpoint
is chosen with ordinary-validation NDCG@10 only, written to disk with
``save_frozen``, and only then evaluated on test data with ``evaluate_frozen``.
Cold-start results come from the same frozen parameters: nothing is retrained
or re-selected for the cold task, and cold-validation numbers are diagnostic.

Files: ``<directory>/<name>.npz`` holds P, Q, W, b; ``<name>.json`` holds the
optimizer, seed, configuration and the selected checkpoint's work and
validation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recommender.baselines import description_similarity, random_scores
from recommender.evaluate import evaluate_model, evaluate_ranking, select_checkpoint

FROZEN_DIR = Path(__file__).resolve().parent.parent / "results" / "frozen"
PARAM_NAMES = ("P", "Q", "W", "b")


def save_frozen(log: dict, directory: Path | str = FROZEN_DIR) -> str:
    """Freeze the selected checkpoint of a training log; returns its name
    ``<optimizer>_seed<seed>``. Refuses to overwrite an existing frozen model."""
    best = select_checkpoint(log)
    name = f"{log['optimizer']}_seed{log['seed']}"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / f"{name}.npz").exists():
        raise FileExistsError(f"{name} is already frozen; delete it explicitly to replace it")
    np.savez(directory / f"{name}.npz", **{k: best["params"][k] for k in PARAM_NAMES})
    meta = {
        "optimizer": log["optimizer"], "seed": log["seed"], "config": log["config"],
        "status": log["status"],
        **{k: best[k] for k in ("work", "updates", "train_time", "train_loss", "val_loss",
                                "val_ndcg", "val_recall") if k in best},
    }
    (directory / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return name


def load_frozen(name: str, directory: Path | str = FROZEN_DIR) -> tuple[dict, dict]:
    directory = Path(directory)
    with np.load(directory / f"{name}.npz") as archive:
        params = {k: archive[k] for k in PARAM_NAMES}
    return params, json.loads((directory / f"{name}.json").read_text())


def evaluate_frozen(names, data, task: str, split: str, directory: Path | str = FROZEN_DIR,
                    baseline_seeds=(616, 617, 618)) -> list[dict]:
    """Recall@10 / NDCG@10 of frozen models and of the task's baselines.

    For the cold task the baselines are description similarity and random
    (mean over ``baseline_seeds``); popularity is not defined for cold products.
    """
    rows = []
    for name in names:
        params, meta = load_frozen(name, directory)
        result = evaluate_model(params, data, task, split)
        rows.append({"model": name, "optimizer": meta["optimizer"], "seed": meta["seed"], **result})
    if task == "cold":
        result = evaluate_ranking(description_similarity(data), data, task, split)
        rows.append({"model": "description_similarity", "optimizer": "baseline", "seed": None, **result})
    randoms = [evaluate_ranking(random_scores(s), data, task, split) for s in baseline_seeds]
    rows.append({"model": "random", "optimizer": "baseline", "seed": None,
                 "recall": float(np.mean([r["recall"] for r in randoms])),
                 "ndcg": float(np.mean([r["ndcg"] for r in randoms])),
                 "n_users": randoms[0]["n_users"]})
    return rows
