"""Block coordinate descent (BCD) on the hybrid model in ``recommender.model``.

Blocks
------
The parameters are split into the two blocks of the report's block-convexity
result:

    block 1:  P                (Q, W, b fixed)  -> lambda_f-strongly convex
    block 2:  (Q, W, b)        (P fixed)        -> min(lambda_f, lambda_w)-strongly convex

One outer iteration solves block 1 approximately, then block 2. Because each
block problem is strongly convex, it has exactly one minimizer, and gradient
descent with a backtracking line search converges to it at a linear rate.

Block solve (inner loop)
------------------------
On the full training objective F (mean pairwise loss over all N training
triplets plus the whole penalty), restricted to the block:

    g = grad_block F(theta)
    stop if |g| <= inner_tol                      (block solved accurately enough)
    or after max_inner gradient evaluations       (work cap per block visit)
    otherwise:  step t, starting from 2 * (last accepted step of this block),
                halved until  F(theta - t g) <= F(theta) - armijo * t * |g|^2,
                then theta_block <- theta_block - t g.

Outer stopping rule: if, in one outer iteration, both blocks already satisfy
``|g| <= inner_tol`` at their first gradient, the iterate is a (approximate)
block-wise stationary point and training stops with status "converged"; the
unused budget is not spent.

The Armijo condition guarantees that F never increases. If no step passes
within ``max_backtracks`` halvings, the block solve stops (logged as a line
search failure); the iterate is left unchanged.

Gradient work
-------------
Every gradient is a full-batch gradient: N = 5,436 triplet-gradient units,
including the final one that only confirms ``|g| <= inner_tol``. Line-search
trial points need only the objective value, not a gradient, so they add no
gradient units; their count is logged and their time is part of
``train_time``, as for the SALR probe ends.

BCD uses no mini-batches and no shuffling, so for a fixed configuration the
seed only changes the shared starting point. The loop, checkpoint fields and
timing boundary otherwise match ``train_sgd`` and ``train_salr``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from recommender.model import Params, data_loss, init_params, loss_and_grad, penalty

BLOCKS = (("P",), ("Q", "W", "b"))


@dataclass(frozen=True)
class BCDConfig:
    lambda_f: float
    lambda_w: float
    inner_tol: float = 1e-4  # stop a block solve once the block gradient norm is this small
    max_inner: int = 20  # at most this many gradients per block visit
    init_step: float = 1.0  # first trial step of each block
    max_step: float = 1e4
    armijo: float = 1e-4
    max_backtracks: int = 40


def objective(params: Params, data, cfg: BCDConfig) -> float:
    """Full training objective F: mean pairwise loss on all training triplets plus penalty."""
    return data_loss(params, data, data.train_triplets) + penalty(params, cfg.lambda_f, cfg.lambda_w)


def _block_norm(grads: Params, block) -> float:
    return float(np.sqrt(sum(np.sum(grads[name] ** 2) for name in block)))


def line_search(params: Params, grads: Params, block, value: float, gnorm: float,
                first_step: float, data, cfg: BCDConfig):
    """Backtracking (Armijo) step on one block. Returns
    (new params, new value, accepted step or 0.0, number of trial evaluations)."""
    step = first_step
    for trial in range(1, cfg.max_backtracks + 1):
        candidate = dict(params)
        for name in block:
            candidate[name] = params[name] - step * grads[name]
        new_value = objective(candidate, data, cfg)
        if np.isfinite(new_value) and new_value <= value - cfg.armijo * step * gnorm**2:
            return candidate, new_value, step, trial
        step *= 0.5
    return params, value, 0.0, cfg.max_backtracks


def train_bcd(
    data,
    seed: int,
    cfg: BCDConfig,
    total_work: int,
    checkpoint_every: int,
    evaluate=None,
) -> dict:
    """Train from the seed's shared starting point for ``total_work``
    triplet-gradient units, checkpointing every ``checkpoint_every`` units.

    ``evaluate(params)`` may return a dict of extra metrics (for example
    ordinary-validation NDCG@10) to store with each checkpoint.

    Returns a log with one entry per gradient (``work``, ``block``,
    ``objective``, ``grad_norm``, ``step``, ``trials``), one entry per block
    visit in ``solves`` (outer iteration, block, gradients used, whether
    ``inner_tol`` was reached, final gradient norm), and a list of
    ``checkpoints``; ``status`` is "converged" if the outer stopping rule
    fired, "failed" if a value or parameter became non-finite, else "ok".
    """
    params = init_params(data, seed)
    n_triplets = len(data.train_triplets)
    log = {
        "optimizer": "bcd", "seed": seed, "config": asdict(cfg), "status": "ok",
        "work": [], "block": [], "objective": [], "grad_norm": [], "step": [], "trials": [],
        "solves": [], "checkpoints": [],
    }
    last_step = {block: cfg.init_step / 2 for block in BLOCKS}  # first trial = init_step
    work, updates, outer, train_time = 0, 0, 0, 0.0

    def checkpoint():
        entry = {
            "work": work,
            "updates": updates,
            "outer": outer,
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
        idle_blocks = 0
        for b, block in enumerate(BLOCKS):
            used, converged, gnorm = 0, False, np.nan
            while used < cfg.max_inner and work < total_work:
                started = time.perf_counter()
                value, grads = loss_and_grad(params, data, data.train_triplets,
                                             cfg.lambda_f, cfg.lambda_w)
                gnorm = _block_norm(grads, block)
                converged = gnorm <= cfg.inner_tol
                step, trials = 0.0, 0
                if not converged and np.isfinite(value):
                    first = min(2 * last_step[block], cfg.max_step)
                    params, _, step, trials = line_search(
                        params, grads, block, value, gnorm, first, data, cfg)
                    if step > 0:
                        last_step[block] = step
                        updates += 1
                train_time += time.perf_counter() - started

                used += 1
                work += n_triplets
                for key, entry in zip(("work", "block", "objective", "grad_norm", "step", "trials"),
                                      (work, b, value, gnorm, step, trials)):
                    log[key].append(entry)

                if not np.isfinite(value) or not all(np.isfinite(v).all() for v in params.values()):
                    log["status"] = "failed"
                    return log
                if work >= next_checkpoint:
                    checkpoint()
                    next_checkpoint += checkpoint_every
                if converged or step == 0.0:
                    break
            log["solves"].append({"outer": outer, "block": b, "gradients": used,
                                  "converged": bool(converged), "grad_norm": float(gnorm),
                                  "line_search_failed": bool(not converged and step == 0.0)})
            idle_blocks += converged and used == 1
            if work >= total_work:
                break
        outer += 1
        if idle_blocks == len(BLOCKS):
            log["status"] = "converged"
            if log["checkpoints"][-1]["work"] != work:
                checkpoint()  # the final point is always a candidate checkpoint
            break
    return log
