"""SGD: the update rule, work accounting, shared batches and training."""

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.model import init_params, loss_and_grad
from recommender.salr import epoch_batches
from recommender.sgd import SGDConfig, train_sgd

CFG = SGDConfig(lr=0.5, lambda_f=1e-3, lambda_w=1e-3)
N = 5436


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_first_update_is_one_gradient_step(ds):
    """After one update, theta = theta_0 - lr * grad F_B(theta_0) on the first shuffled batch."""
    log = train_sgd(ds, 616, CFG, total_work=256, checkpoint_every=256)
    start = init_params(ds, 616)
    batch = next(epoch_batches(ds.train_triplets, 256, np.random.default_rng(616 + 1000)))
    loss, grads = loss_and_grad(start, ds, batch, CFG.lambda_f, CFG.lambda_w)
    after = log["checkpoints"][-1]["params"]
    assert log["batch_loss"] == [loss]
    for name in start:
        np.testing.assert_allclose(after[name], start[name] - CFG.lr * grads[name], atol=1e-15)


def test_work_accounting(ds):
    """One update costs |B| units; a pass is 21 batches of 256 and one of 60."""
    log = train_sgd(ds, 616, CFG, total_work=2 * N, checkpoint_every=2_000)
    per_update = np.diff([0] + log["work"])
    assert list(per_update[:22]) == [256] * 21 + [60]
    assert log["work"][21] == N and log["work"][-1] == 2 * N
    checkpoints = log["checkpoints"]
    # A checkpoint is taken at the first update that reaches each multiple of 2,000.
    assert [c["work"] for c in checkpoints] == [0, 2048, 4096, 6204, 8252, 10044]
    assert [c["updates"] for c in checkpoints] == [0, 8, 16, 25, 33, 40]
    assert all(np.diff([c["train_time"] for c in checkpoints]) > 0)


def test_checkpoints_are_snapshots(ds):
    log = train_sgd(ds, 616, CFG, total_work=2_000, checkpoint_every=500)
    first, last = log["checkpoints"][0]["params"], log["checkpoints"][-1]["params"]
    start = init_params(ds, 616)
    assert all(np.array_equal(first[name], start[name]) for name in start)
    assert not np.array_equal(first["P"], last["P"])


def test_training_is_reproducible_and_reduces_loss(ds):
    first = train_sgd(ds, 616, CFG, total_work=60_000, checkpoint_every=20_000)
    again = train_sgd(ds, 616, CFG, total_work=60_000, checkpoint_every=20_000)
    assert first["status"] == "ok"
    assert first["batch_loss"] == again["batch_loss"]
    assert first["checkpoints"][0]["train_loss"] == pytest.approx(np.log(2), abs=1e-3)
    assert first["checkpoints"][-1]["train_loss"] < first["checkpoints"][0]["train_loss"] - 0.01


def test_evaluate_metrics_are_stored(ds):
    log = train_sgd(ds, 616, CFG, 1_000, 500, evaluate=lambda p: {"norm_b": float(np.abs(p["b"]).sum())})
    assert log["checkpoints"][0]["norm_b"] == 0.0 and log["checkpoints"][-1]["norm_b"] > 0.0


def test_divergence_is_reported_as_failed(ds):
    cfg = SGDConfig(lr=1e6, lambda_f=10.0, lambda_w=10.0)
    with np.errstate(all="ignore"):
        log = train_sgd(ds, 616, cfg, total_work=200_000, checkpoint_every=50_000)
    assert log["status"] == "failed"
