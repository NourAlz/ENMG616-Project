"""Scores, loss and gradients of the hybrid model."""

from types import SimpleNamespace

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.model import data_loss, init_params, loss_and_grad, penalty, scores

LAMBDA_F, LAMBDA_W = 0.03, 0.007


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def random_params(ds, seed=0):
    """Large random parameters, so gradients are checked away from the flat start."""
    rng = np.random.default_rng(seed)
    return {name: rng.normal(0, 0.3, value.shape) for name, value in init_params(ds, 0).items()}


def test_initialization_follows_the_brief(ds):
    params = init_params(ds, seed=616)
    shapes = {name: value.shape for name, value in params.items()}
    assert shapes == {"P": (170, 32), "Q": (206, 32), "W": (32, 384), "b": (206,)}
    assert sum(value.size for value in params.values()) == 24_526
    assert (params["b"] == 0).all()
    assert abs(np.concatenate([params[n].ravel() for n in "PQW"]).std() - 0.01) < 5e-4
    again = init_params(ds, seed=616)
    assert all(np.array_equal(params[name], again[name]) for name in params)
    assert not np.array_equal(params["P"], init_params(ds, seed=617)["P"])


def test_worked_example_from_the_brief():
    """Section 5.1 and 5.2: tent scores 1.4, band 0.5, a cold product 1.1."""
    toy = SimpleNamespace(
        Z=np.array([[0.6, 0.8], [0.0, 1.0], [0.8, 0.6]]),  # tent, band, cold product
        q_index=np.array([0, 1, -1]),
    )
    params = {
        "P": np.array([[1.0, 0.5]]),
        "Q": np.array([[0.2, 0.0], [0.0, 0.2]]),
        "W": np.eye(2),
        "b": np.array([0.2, -0.1]),
    }
    assert np.allclose(scores(params, toy, 0, [0, 1, 2]), [1.4, 0.5, 1.1])
    tent_over_band = np.array([[0, 0, 1]])
    assert data_loss(params, toy, tent_over_band) == pytest.approx(np.log1p(np.exp(-0.9)))
    assert data_loss(params, toy, tent_over_band[:, [0, 2, 1]]) == pytest.approx(1.241, abs=1e-3)


def test_loss_matches_scores(ds):
    """The batched loss equals the definition built from individual scores."""
    params = random_params(ds)
    triplets = ds.train_triplets[:50]
    expected = np.mean(
        [
            np.log1p(np.exp(-(scores(params, ds, u, [i])[0] - scores(params, ds, u, [j])[0])))
            for u, i, j in triplets
        ]
    )
    assert data_loss(params, ds, triplets) == pytest.approx(expected)
    loss, _ = loss_and_grad(params, ds, triplets, LAMBDA_F, LAMBDA_W)
    assert loss == pytest.approx(expected + penalty(params, LAMBDA_F, LAMBDA_W))


def test_gradient_matches_finite_differences(ds):
    """Central differences on 60 entries of each of P, Q, W and b."""
    rng = np.random.default_rng(1)
    params = random_params(ds)
    triplets = ds.train_triplets[rng.choice(len(ds.train_triplets), 256, replace=False)]
    _, grads = loss_and_grad(params, ds, triplets, LAMBDA_F, LAMBDA_W)
    eps = 1e-6
    for name, value in params.items():
        for flat in rng.choice(value.size, 60, replace=False):
            index = np.unravel_index(flat, value.shape)
            original = value[index]
            value[index] = original + eps
            up, _ = loss_and_grad(params, ds, triplets, LAMBDA_F, LAMBDA_W)
            value[index] = original - eps
            down, _ = loss_and_grad(params, ds, triplets, LAMBDA_F, LAMBDA_W)
            value[index] = original
            assert grads[name][index] == pytest.approx((up - down) / (2 * eps), rel=1e-5, abs=1e-8)


def test_penalty_reaches_rows_outside_the_batch(ds):
    params = random_params(ds)
    triplets = ds.train_triplets[:3]  # all three belong to customer 0
    _, grads = loss_and_grad(params, ds, triplets, LAMBDA_F, LAMBDA_W)
    assert np.allclose(grads["P"][1:], LAMBDA_F * params["P"][1:])


def test_all_zero_start_moves_only_the_bias(ds):
    zeros = {name: np.zeros_like(value) for name, value in init_params(ds, 0).items()}
    loss, grads = loss_and_grad(zeros, ds, ds.train_triplets, LAMBDA_F, LAMBDA_W)
    assert loss == pytest.approx(np.log(2))
    assert not grads["P"].any() and not grads["Q"].any() and not grads["W"].any()
    assert grads["b"].any()


def test_large_scores_stay_finite(ds):
    params = {name: 1e3 * value for name, value in random_params(ds).items()}
    loss, grads = loss_and_grad(params, ds, ds.train_triplets, LAMBDA_F, LAMBDA_W)
    assert np.isfinite(loss) and all(np.isfinite(g).all() for g in grads.values())


def test_objective_is_not_convex_near_the_origin(ds):
    """Reproduces the report's thresholds: F has negative curvature at the origin
    when lambda_f < sigma_max(C) / 2N or sqrt(lambda_f lambda_w) < sigma_max(D) / 2N."""
    users, positives, comparisons = ds.train_triplets.T
    n = len(users)
    C = np.zeros((ds.n_users, ds.n_ordinary))
    np.add.at(C, (users, ds.q_index[positives]), 1.0)
    np.add.at(C, (users, ds.q_index[comparisons]), -1.0)
    D = np.zeros((ds.n_users, ds.Z.shape[1]))
    np.add.at(D, users, ds.Z[positives] - ds.Z[comparisons])
    (a_c, s_c, c_t), (a_d, s_d, w_t) = (np.linalg.svd(M, full_matrices=False) for M in (C, D))
    assert s_c[0] / (2 * n) == pytest.approx(2.79e-3, rel=5e-3)
    assert s_d[0] / (2 * n) == pytest.approx(4.63e-3, rel=5e-3)

    x = np.eye(32)[0]
    zero = {name: np.zeros_like(value) for name, value in init_params(ds, 0).items()}
    directions = [
        ({"P": np.outer(a_c[:, 0], x), "Q": np.outer(c_t[0], x)}, s_c[0] / (2 * n)),
        ({"P": np.outer(a_d[:, 0], x), "W": np.outer(x, w_t[0])}, s_d[0] / (2 * n)),
    ]
    for direction, threshold in directions:
        for lam, sign in ((0.5 * threshold, -1), (2.0 * threshold, 1)):
            values = [
                loss_and_grad(
                    {k: zero[k] + t * direction.get(k, 0.0) for k in zero}, ds, ds.train_triplets, lam, lam
                )[0]
                for t in (-1e-2, 0.0, 1e-2)
            ]
            assert np.sign(values[0] + values[2] - 2 * values[1]) == sign
