"""Sharpness-aware minimization (SAM), after Foret, Kleiner, Mobahi and
Neyshabur (ICLR 2021), on the hybrid model in ``recommender.model``.

How it works
------------
SGD lowers the loss at the current point. SAM lowers the worst loss within a
small ball around it, which favours regions where the loss stays low nearby
(flat minima):

    min over theta of   max over |eps| <= rho of   F_B(theta + eps)

Each update, on the current mini-batch B and iterate theta:

1. Gradient at theta:        g = grad F_B(theta).
2. Worst-case perturbation:  eps = rho * g / |g|   (one ascent step of length rho).
3. Gradient at the perturbed point:  g_sam = grad F_B(theta + eps).
4. Update from the original point:   theta <- theta - lr * g_sam.

The perturbation is never kept: theta + eps is only where the second gradient
is taken. With rho = 0 the update is exactly plain SGD. F_B is the regularized
mini-batch objective, and the norm |g| is over all parameters together.

Gradient work
-------------
Both passes use the same mini-batch and both are counted: one update costs
2 |B| triplet-gradient units, so one pass over the data costs 2 N = 10,872.

The loop, the shuffle stream, the checkpoint fields and the timing boundary
match ``recommender.sgd.train_sgd``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from recommender.model import data_loss, init_params, loss_and_grad, penalty
from recommender.salr import epoch_batches


@dataclass(frozen=True)
class SAMConfig:
    lr: float
    rho: float  # perturbation radius, in parameter space
    lambda_f: float
    lambda_w: float
    batch_size: int = 256


def train_sam(
    data,
    seed: int,
    cfg: SAMConfig,
    total_work: int,
    checkpoint_every: int,
    evaluate=None,
) -> dict:
    """Train from the seed's shared starting point for ``total_work``
    triplet-gradient units, checkpointing every ``checkpoint_every`` units.

    ``evaluate(params)`` may return a dict of extra metrics (for example
    ordinary-validation NDCG@10) to store with each checkpoint.

    Returns a log with one entry per update (``work``; ``batch_loss``, the
    regularized batch objective at the unperturbed point before the step;
    ``grad_norm``, the gradient norm there) and a list of ``checkpoints``;
    ``status`` is "failed" if a loss or parameter became non-finite.
    """
    params = init_params(data, seed)
    shuffle = np.random.default_rng(seed + 1000)  # separate stream from the initialization
    log = {
        "optimizer": "sam", "seed": seed, "config": asdict(cfg), "status": "ok",
        "work": [], "batch_loss": [], "grad_norm": [], "checkpoints": [],
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
            grad_norm = float(np.sqrt(sum(np.sum(g**2) for g in grads.values())))
            gradients = 1
            if grad_norm > 0.0:  # at an exact stationary point there is no ascent direction
                scale = cfg.rho / grad_norm
                perturbed = {name: params[name] + scale * grads[name] for name in params}
                _, grads = loss_and_grad(perturbed, data, batch, cfg.lambda_f, cfg.lambda_w)
                gradients = 2
            for name in params:
                params[name] -= cfg.lr * grads[name]
            train_time += time.perf_counter() - started

            work += gradients * len(batch)
            updates += 1
            log["work"].append(work)
            log["batch_loss"].append(loss)
            log["grad_norm"].append(grad_norm)

            if not np.isfinite(loss) or not all(np.isfinite(v).all() for v in params.values()):
                log["status"] = "failed"
                return log
            if work >= next_checkpoint:
                checkpoint()
                next_checkpoint += checkpoint_every
            if work >= total_work:
                break
    return log
