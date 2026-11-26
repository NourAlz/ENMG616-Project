"""BCD: block isolation, monotone objective, block convergence and work accounting."""

import numpy as np
import pytest

from recommender.bcd import BLOCKS, BCDConfig, _block_norm, line_search, objective, train_bcd
from recommender.data import load_dataset
from recommender.model import init_params, loss_and_grad

CFG = BCDConfig(lambda_f=1e-3, lambda_w=1e-3, inner_tol=1e-4, max_inner=5)


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_line_search_moves_only_the_block_and_decreases_f(ds):
    params = init_params(ds, 616)
    value, grads = loss_and_grad(params, ds, ds.train_triplets, CFG.lambda_f, CFG.lambda_w)
    for block in BLOCKS:
        gnorm = _block_norm(grads, block)
        new, new_value, step, trials = line_search(params, grads, block, value, gnorm, 1e3, ds, CFG)
        assert step > 0 and trials >= 1
        assert new_value < value
        assert new_value == pytest.approx(objective(new, ds, CFG))
        for name in params:
            moved = not np.array_equal(new[name], params[name])
            assert moved == (name in block)


def test_objective_never_increases(ds):
    log = train_bcd(ds, 616, CFG, total_work=60 * len(ds.train_triplets), checkpoint_every=10**9)
    assert log["status"] == "ok"
    assert np.all(np.diff(log["objective"]) <= 1e-12)
    assert log["objective"][-1] < log["objective"][0] - 0.05


def test_block_solve_reaches_tolerance(ds):
    cfg = BCDConfig(lambda_f=1e-2, lambda_w=1e-2, inner_tol=1e-3, max_inner=500)
    log = train_bcd(ds, 616, cfg, total_work=600 * len(ds.train_triplets), checkpoint_every=10**9)
    assert log["status"] in ("ok", "converged")
    first_p_solve = log["solves"][0]
    assert first_p_solve["block"] == 0 and first_p_solve["converged"]
    assert first_p_solve["grad_norm"] <= cfg.inner_tol


def test_work_accounting_and_block_order(ds):
    n = len(ds.train_triplets)
    cfg = BCDConfig(lambda_f=1e-3, lambda_w=1e-3, inner_tol=0.0, max_inner=3)
    log = train_bcd(ds, 616, cfg, total_work=13 * n, checkpoint_every=4 * n)
    assert log["work"] == [n * (i + 1) for i in range(13)]  # every gradient is full batch
    assert log["block"][:12] == [0, 0, 0, 1, 1, 1] * 2
    assert [s["gradients"] for s in log["solves"]] == [3, 3, 3, 3, 1]
    assert [c["work"] for c in log["checkpoints"]] == [0, 4 * n, 8 * n, 12 * n]


def test_training_is_reproducible_and_seed_sets_the_start(ds):
    kw = dict(total_work=20 * len(ds.train_triplets), checkpoint_every=10**9)
    first, again = train_bcd(ds, 616, CFG, **kw), train_bcd(ds, 616, CFG, **kw)
    other = train_bcd(ds, 617, CFG, **kw)
    assert first["objective"] == again["objective"]
    assert first["objective"] != other["objective"]
    assert first["checkpoints"][0]["train_loss"] == pytest.approx(np.log(2), abs=1e-3)


def test_evaluate_metrics_are_stored(ds):
    log = train_bcd(ds, 616, CFG, total_work=4 * len(ds.train_triplets),
                    checkpoint_every=2 * len(ds.train_triplets),
                    evaluate=lambda p: {"val_ndcg": 0.5})
    assert all(c["val_ndcg"] == 0.5 for c in log["checkpoints"])


def test_stops_at_a_blockwise_stationary_point(ds):
    cfg = BCDConfig(lambda_f=1e-2, lambda_w=1e-2, inner_tol=1e-3, max_inner=500)
    total = 5_000 * len(ds.train_triplets)
    log = train_bcd(ds, 616, cfg, total_work=total, checkpoint_every=10**9)
    assert log["status"] == "converged"
    assert log["work"][-1] < total
    assert [s["gradients"] for s in log["solves"][-2:]] == [1, 1]
    assert log["checkpoints"][-1]["work"] == log["work"][-1]
