"""SALR: sharpness estimate, learning-rate rule, work accounting and training."""

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.model import init_params, loss_and_grad
from recommender.salr import SALRConfig, _norm, next_learning_rate, sharpness, train_salr

CFG = SALRConfig(base_lr=0.5, probe_step=0.05, lambda_f=1e-3, lambda_w=1e-3)


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_one_step_sharpness_is_first_order(ds):
    """For a tiny probe step, S = F(theta + g*d) - F(theta - g*d) ~ 2 * step * |g|."""
    params, batch = init_params(ds, 616), ds.train_triplets[:256]
    _, grads = loss_and_grad(params, ds, batch, CFG.lambda_f, CFG.lambda_w)
    cfg = SALRConfig(base_lr=0.5, probe_step=1e-4, lambda_f=1e-3, lambda_w=1e-3)
    estimate, extra = sharpness(params, grads, ds, batch, cfg)
    assert estimate == pytest.approx(2 * cfg.probe_step * _norm(grads), rel=1e-3)
    assert extra == 0  # single-step probes reuse the update's gradient


def test_probes_do_not_move_the_iterate(ds):
    params, batch = init_params(ds, 616), ds.train_triplets[:256]
    before = {name: value.copy() for name, value in params.items()}
    _, grads = loss_and_grad(params, ds, batch, CFG.lambda_f, CFG.lambda_w)
    cfg = SALRConfig(0.5, 0.05, 1e-3, 1e-3, n_descent=3, n_ascent=2)
    estimate, extra = sharpness(params, grads, ds, batch, cfg)
    assert estimate > 0 and extra == 3  # (3 - 1) + (2 - 1)
    assert all(np.array_equal(params[name], before[name]) for name in params)


def test_learning_rate_rule():
    history = []
    assert next_learning_rate(2.0, history, 0.1, 0.1) == (0.1, True)  # S / median = 1
    assert next_learning_rate(6.0, history, 0.1, 0.1) == (pytest.approx(0.15), True)  # 6 / 4
    for invalid in (0.0, -1.0, np.nan, np.inf):
        assert next_learning_rate(invalid, history, 0.15, 0.1) == (0.15, False)
    assert history == [2.0, 6.0]  # invalid estimates stay out of the median


def test_work_accounting(ds):
    cfg = SALRConfig(0.5, 0.05, 1e-3, 1e-3, n_descent=3, n_ascent=3, probe_every=2)
    log = train_salr(ds, 616, cfg, total_work=5_000, checkpoint_every=2_000)
    per_update = np.diff([0] + log["work"])
    assert log["probed"][:4] == [True, False, True, False]
    assert list(per_update[:4]) == [5 * 256, 256, 5 * 256, 256]  # 1 + (3-1) + (3-1) gradients
    assert log["lr"][0] == cfg.base_lr and log["lr"][1] == log["lr"][0]  # held between probes
    assert [c["work"] for c in log["checkpoints"]][0] == 0
    assert all(np.diff([c["work"] for c in log["checkpoints"]]) > 0)


def test_training_is_reproducible_and_reduces_loss(ds):
    first = train_salr(ds, 616, CFG, total_work=60_000, checkpoint_every=20_000)
    again = train_salr(ds, 616, CFG, total_work=60_000, checkpoint_every=20_000)
    assert first["status"] == "ok"
    assert first["lr"] == again["lr"]
    assert first["checkpoints"][-1]["train_loss"] == again["checkpoints"][-1]["train_loss"]
    assert first["checkpoints"][-1]["train_loss"] < first["checkpoints"][0]["train_loss"] - 0.01
    assert first["checkpoints"][0]["train_loss"] == pytest.approx(np.log(2), abs=1e-3)


def test_divergence_is_reported_as_failed(ds):
    cfg = SALRConfig(base_lr=1e6, probe_step=0.05, lambda_f=10.0, lambda_w=10.0)
    with np.errstate(all="ignore"):
        log = train_salr(ds, 616, cfg, total_work=200_000, checkpoint_every=50_000)
    assert log["status"] == "failed"
