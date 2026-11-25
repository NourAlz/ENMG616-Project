"""Plain mini-batch SGD, without momentum, on the hybrid model in
``recommender.model``.

Each update draws a mini-batch B from a shuffled pass over the training
triplets and takes one step on the regularized objective of the batch:

    theta <- theta - lr * grad F_B(theta),    F_B = L_B + penalty.

The learning rate is constant. The penalty gradient covers every learned
parameter at every update, not only the rows that appear in the batch.

Gradient work: one update costs |B| triplet-gradient units, so one pass over
the data costs N = 5,436 units.

The loop, the shuffle stream, the checkpoint fields and the timing boundary
match ``recommender.salr.train_salr``: for a given seed both methods start from
the same parameters and see the same batches in the same order, and
``train_time`` covers the gradient and the update but not the checkpoints.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from recommender.model import data_loss, init_params, loss_and_grad, penalty
from recommender.salr import epoch_batches


@dataclass(frozen=True)
class SGDConfig:
    lr: float
    lambda_f: float
    lambda_w: float
    batch_size: int = 256


def train_sgd(
    data,
    seed: int,
    cfg: SGDConfig,
    total_work: int,
    checkpoint_every: int,
    evaluate=None,
) -> dict:
    """Train from the seed's shared starting point for ``total_work``
    triplet-gradient units, checkpointing every ``checkpoint_every`` units.

    ``evaluate(params)`` may return a dict of extra metrics (for example
    ordinary-validation NDCG@10) to store with each checkpoint.

    Returns a log with one entry per update (``work``, ``batch_loss``: the
    regularized batch objective before the step) and a list of
    ``checkpoints``; ``status`` is "failed" if a loss or parameter became
    non-finite.
    """
    params = init_params(data, seed)
    shuffle = np.random.default_rng(seed + 1000)  # separate stream from the initialization
    log = {
        "optimizer": "sgd", "seed": seed, "config": asdict(cfg), "status": "ok",
        "work": [], "batch_loss": [], "checkpoints": [],
    }
    work, updates, train_time = 0, 0, 0.0

    def checkpoint():
        entry = {
            "work": work,
            "updates": updates,
            "train_time": train_time,
            "train_loss": data_loss(params, data, data.train_triplets),
            "val_loss": data_loss(params, data, data.validation_triplets),
            "penalty": penalty(params, cfg.lambda_f, cfg.lambda_w),
            "params": {name: value.copy() for name, value in params.items()},
        }
        if evaluate is not None:
            entry.update(evaluate(params))
        log["checkpoints"].append(entry)

    checkpoint()
    next_checkpoint = checkpoint_every
    while work < total_work:
        for batch in epoch_batches(data.train_triplets, cfg.batch_size, shuffle):
            started = time.perf_counter()
            loss, grads = loss_and_grad(params, data, batch, cfg.lambda_f, cfg.lambda_w)
            for name in params:
                params[name] -= cfg.lr * grads[name]
            train_time += time.perf_counter() - started

            work += len(batch)
            updates += 1
            log["work"].append(work)
            log["batch_loss"].append(loss)

            if not np.isfinite(loss) or not all(np.isfinite(v).all() for v in params.values()):
                log["status"] = "failed"
                return log
            if work >= next_checkpoint:
                checkpoint()
                next_checkpoint += checkpoint_every
            if work >= total_work:
                break
    return log
