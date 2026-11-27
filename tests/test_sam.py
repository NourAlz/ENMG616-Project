"""SAM: the two-pass update, work accounting, its relation to SGD, and training."""

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.model import init_params, loss_and_grad
from recommender.salr import epoch_batches
from recommender.sam import SAMConfig, train_sam
from recommender.sgd import SGDConfig, train_sgd

LAM = 1e-3
CFG = SAMConfig(lr=0.5, rho=0.05, lambda_f=LAM, lambda_w=LAM)
N = 5436


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_first_update_uses_the_gradient_at_the_perturbed_point(ds):
    """theta_1 = theta_0 - lr * grad F_B(theta_0 + rho * g / |g|), with g = grad F_B(theta_0)."""
    log = train_sam(ds, 616, CFG, total_work=512, checkpoint_every=512)
    start = init_params(ds, 616)
    batch = next(epoch_batches(ds.train_triplets, 256, np.random.default_rng(616 + 1000)))
    loss, g = loss_and_grad(start, ds, batch, LAM, LAM)
    g_norm = np.sqrt(sum(np.sum(v**2) for v in g.values()))
    perturbed = {name: start[name] + CFG.rho * g[name] / g_norm for name in start}
    shift = np.sqrt(sum(np.sum((perturbed[name] - start[name]) ** 2) for name in start))
    assert shift == pytest.approx(CFG.rho)  # the perturbation has length rho
    perturbed_loss, g_sam = loss_and_grad(perturbed, ds, batch, LAM, LAM)
    assert perturbed_loss > loss  # it is an ascent step

    assert log["batch_loss"] == [loss] and log["grad_norm"] == [pytest.approx(g_norm)]
    after = log["checkpoints"][-1]["params"]
    for name in start:
        np.testing.assert_allclose(after[name], start[name] - CFG.lr * g_sam[name], atol=1e-15)
        assert not np.allclose(after[name], start[name] - CFG.lr * g[name], atol=1e-12)


def test_both_passes_are_counted(ds):
    """One update costs 2|B| units; a pass is 21 batches of 256 and one of 60."""
    log = train_sam(ds, 616, CFG, total_work=4 * N, checkpoint_every=4_000)
    per_update = np.diff([0] + log["work"])
    assert list(per_update[:22]) == [512] * 21 + [120]
    assert log["work"][21] == 2 * N and log["work"][-1] == 4 * N
    assert [c["updates"] for c in log["checkpoints"]] == [0, 8, 16, 25, 33, 40]


def test_zero_radius_is_plain_sgd(ds):
    """With rho = 0 SAM follows SGD's trajectory exactly, at twice the gradient work."""
    sam = train_sam(ds, 616, SAMConfig(0.5, 0.0, LAM, LAM), total_work=20_000, checkpoint_every=20_000)
    sgd = train_sgd(ds, 616, SGDConfig(0.5, LAM, LAM), total_work=10_000, checkpoint_every=10_000)
    assert sam["checkpoints"][-1]["updates"] == sgd["checkpoints"][-1]["updates"]
    assert sam["checkpoints"][-1]["work"] == 2 * sgd["checkpoints"][-1]["work"]
    assert sam["batch_loss"] == sgd["batch_loss"]
    for name, value in sgd["checkpoints"][-1]["params"].items():
        assert np.array_equal(sam["checkpoints"][-1]["params"][name], value)


def test_training_is_reproducible_and_reduces_loss(ds):
    first = train_sam(ds, 616, CFG, total_work=120_000, checkpoint_every=40_000)
    again = train_sam(ds, 616, CFG, total_work=120_000, checkpoint_every=40_000)
    assert first["status"] == "ok"
    assert first["batch_loss"] == again["batch_loss"]
    assert first["checkpoints"][0]["train_loss"] == pytest.approx(np.log(2), abs=1e-3)
    assert first["checkpoints"][-1]["train_loss"] < first["checkpoints"][0]["train_loss"] - 0.01
    start = init_params(ds, 616)  # the same starting point as every other optimizer
    assert all(np.array_equal(first["checkpoints"][0]["params"][n], start[n]) for n in start)


def test_divergence_is_reported_as_failed(ds):
    cfg = SAMConfig(lr=1e6, rho=0.05, lambda_f=10.0, lambda_w=10.0)
    with np.errstate(all="ignore"):
        log = train_sam(ds, 616, cfg, total_work=200_000, checkpoint_every=50_000)
    assert log["status"] == "failed"
