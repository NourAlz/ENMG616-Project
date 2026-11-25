"""SGD with a sharpness-aware learning rate (SALR), after Yue, Nouiehed and
Al Kontar (arXiv 2011.05348), on the hybrid model in ``recommender.model``.

How it works
------------
SALR is plain mini-batch SGD whose learning rate follows the local sharpness of
the loss: large steps in sharp regions (to leave them), small steps in flat
regions (to settle there).

Every ``probe_every`` updates, on the current mini-batch B and iterate theta:

1. Probe downhill: ``n_descent`` steps of length ``probe_step`` along -g/|g|.
2. Probe uphill:   ``n_ascent`` steps of length ``probe_step`` along +g/|g|.
   Both probes start from theta, and theta itself is not moved by them.
3. Sharpness  S = F_B(uphill end) - F_B(downhill end).
4. Learning rate  lr = base_lr * S / median(all valid S so far).

Then the usual update  theta <- theta - lr * grad F_B(theta).  Between probes
the last learning rate is kept. An estimate that is not finite or not positive
is invalid: it is logged, left out of the median, and the learning rate is kept.

F_B is the regularized objective of ``model.loss_and_grad``. Probe steps are
normalized by the norm of the mini-batch gradient (the paper's Definition 1).

Gradient work
-------------
The first step of each probe reuses the gradient already computed for the
update, so a probe adds ``(n_descent - 1) + (n_ascent - 1)`` gradients of |B|
units each. The two loss evaluations at the probe ends add no gradient units,
but their time is part of the training time.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from recommender.model import Params, data_loss, init_params, loss_and_grad, penalty


@dataclass(frozen=True)
class SALRConfig:
    base_lr: float
    probe_step: float  # gamma: length of one probe step in parameter space
    lambda_f: float
    lambda_w: float
    n_descent: int = 1
    n_ascent: int = 1
    probe_every: int = 1  # c: probe once every this many updates
    batch_size: int = 256


def _moved(params: Params, direction: Params, scale: float) -> Params:
    return {name: value + scale * direction[name] for name, value in params.items()}


def _norm(grads: Params) -> float:
    return float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))


def epoch_batches(triplets: np.ndarray, batch_size: int, rng: np.random.Generator):
    """One shuffled pass over the triplets; the final smaller batch is kept."""
    order = rng.permutation(len(triplets))
    for start in range(0, len(triplets), batch_size):
        yield triplets[order[start : start + batch_size]]


def sharpness(params: Params, grads: Params, data, batch: np.ndarray, cfg: SALRConfig):
    """Local sharpness at ``params`` on ``batch``, given the gradient there.
    Returns the estimate and the number of extra gradient evaluations used."""
    ends, extra_gradients = [], 0
    for sign, n_steps in ((-1.0, cfg.n_descent), (1.0, cfg.n_ascent)):
        point, direction = params, grads
        for step in range(n_steps):
            if step > 0:
                _, direction = loss_and_grad(point, data, batch, cfg.lambda_f, cfg.lambda_w)
                extra_gradients += 1
            length = _norm(direction)
            if length == 0.0:
                break
            point = _moved(point, direction, sign * cfg.probe_step / length)
        ends.append(data_loss(point, data, batch) + penalty(point, cfg.lambda_f, cfg.lambda_w))
    return ends[1] - ends[0], extra_gradients


def next_learning_rate(estimate: float, history: list[float], current: float, base_lr: float):
    """Apply lr = base_lr * S / median(S). Returns (learning rate, was it valid);
    a valid estimate is appended to ``history``."""
    if not np.isfinite(estimate) or estimate <= 0.0:
        return current, False
    history.append(estimate)
    return base_lr * estimate / float(np.median(history)), True


def train_salr(
    data,
    seed: int,
    cfg: SALRConfig,
    total_work: int,
    checkpoint_every: int,
    evaluate=None,
) -> dict:
    """Train from the seed's shared starting point for ``total_work``
    triplet-gradient units, checkpointing every ``checkpoint_every`` units.

    ``evaluate(params)`` may return a dict of extra metrics (for example
    ordinary-validation NDCG@10) to store with each checkpoint.

    Returns a log with one entry per update (``work``, ``lr``, ``probed``,
    ``sharpness``, ``valid``) and a list of ``checkpoints``; ``status`` is
    "failed" if a loss or parameter became non-finite.
    """
    params = init_params(data, seed)
    shuffle = np.random.default_rng(seed + 1000)  # separate stream from the initialization
    log = {
        "optimizer": "salr", "seed": seed, "config": asdict(cfg), "status": "ok",
        "work": [], "lr": [], "probed": [], "sharpness": [], "valid": [], "checkpoints": [],
    }
    history: list[float] = []
    lr, work, updates, train_time = cfg.base_lr, 0, 0, 0.0

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
            gradients, probed, estimate, valid = 1, updates % cfg.probe_every == 0, np.nan, False
            if probed:
                estimate, extra = sharpness(params, grads, data, batch, cfg)
                gradients += extra
                lr, valid = next_learning_rate(estimate, history, lr, cfg.base_lr)
            params = _moved(params, grads, -lr)
            train_time += time.perf_counter() - started

            work += gradients * len(batch)
            updates += 1
            for key, value in zip(
                ("work", "lr", "probed", "sharpness", "valid"), (work, lr, probed, estimate, valid)
            ):
                log[key].append(value)

            if not np.isfinite(loss) or not all(np.isfinite(v).all() for v in params.values()):
                log["status"] = "failed"
                return log
            if work >= next_checkpoint:
                checkpoint()
                next_checkpoint += checkpoint_every
            if work >= total_work:
                break
    return log
