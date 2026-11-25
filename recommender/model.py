"""The hybrid recommendation model: parameters, scores, loss and gradients.

    score       s_ui = b_i + p_u . (q_i + W z_i)      cold product: q_i = b_i = 0
    triplet     D_uij = s_ui - s_uj,   l_uij = log(1 + exp(-D_uij))
    objective   F = mean(l) + lambda_f/2 (|P|^2 + |Q|^2 + |b|^2) + lambda_w/2 |W|^2

Parameters are a dict of arrays ``{"P", "Q", "W", "b"}``. ``data`` is a
``Dataset`` (only ``Z``, ``q_index``, ``n_users`` and ``n_ordinary`` are used),
and ``triplets`` is an (n, 3) array of user_id, positive_id, comparison_id.
"""

from __future__ import annotations

import numpy as np

K = 32  # learned coordinates per customer and product
INIT_STD = 0.01

Params = dict[str, np.ndarray]


def init_params(data, seed: int, k: int = K) -> Params:
    """P, Q, W ~ N(0, 0.01^2) and b = 0. The same seed always gives the same
    arrays, so every optimizer starts from an identical point."""
    rng = np.random.default_rng(seed)
    return {
        "P": rng.standard_normal((data.n_users, k)) * INIT_STD,
        "Q": rng.standard_normal((data.n_ordinary, k)) * INIT_STD,
        "W": rng.standard_normal((k, data.Z.shape[1])) * INIT_STD,
        "b": np.zeros(data.n_ordinary),
    }


def scores(params: Params, data, user_id: int, product_ids) -> np.ndarray:
    """Scores of one customer for the given products, ordinary or cold."""
    product_ids = np.asarray(product_ids)
    rows = data.q_index[product_ids]
    learned = rows >= 0  # cold products keep q_i = b_i = 0
    vectors = data.Z[product_ids] @ params["W"].T
    vectors[learned] += params["Q"][rows[learned]]
    result = vectors @ params["P"][user_id]
    result[learned] += params["b"][rows[learned]]
    return result


def _forward(params: Params, data, triplets: np.ndarray):
    """Score differences D for each triplet, plus the pieces the gradient reuses."""
    users, positives, comparisons = triplets.T
    qi, qj = data.q_index[positives], data.q_index[comparisons]
    p_u = params["P"][users]
    z_diff = data.Z[positives] - data.Z[comparisons]
    v_diff = params["Q"][qi] - params["Q"][qj] + z_diff @ params["W"].T
    margins = params["b"][qi] - params["b"][qj] + np.einsum("nk,nk->n", p_u, v_diff)
    return margins, (users, qi, qj, p_u, z_diff, v_diff)


def data_loss(params: Params, data, triplets: np.ndarray) -> float:
    """Mean pairwise logistic loss, without the regularization penalty."""
    margins, _ = _forward(params, data, triplets)
    return float(np.logaddexp(0.0, -margins).mean())


def penalty(params: Params, lambda_f: float, lambda_w: float) -> float:
    factors = sum(float(np.sum(params[name] ** 2)) for name in ("P", "Q", "b"))
    return 0.5 * lambda_f * factors + 0.5 * lambda_w * float(np.sum(params["W"] ** 2))


def loss_and_grad(
    params: Params, data, triplets: np.ndarray, lambda_f: float, lambda_w: float
) -> tuple[float, Params]:
    """Regularized objective on a batch of triplets and its gradient.

    The penalty covers every learned parameter, not only rows in the batch.
    One call costs ``len(triplets)`` triplet-gradient units.
    """
    margins, (users, qi, qj, p_u, z_diff, v_diff) = _forward(params, data, triplets)
    loss = float(np.logaddexp(0.0, -margins).mean()) + penalty(params, lambda_f, lambda_w)

    # d(mean loss)/dD = -sigmoid(-D) / n, computed without overflow.
    g = -np.exp(-np.logaddexp(0.0, margins)) / len(margins)
    g_p = g[:, None] * p_u

    grads = {name: np.zeros_like(value) for name, value in params.items()}
    np.add.at(grads["P"], users, g[:, None] * v_diff)
    np.add.at(grads["Q"], qi, g_p)
    np.add.at(grads["Q"], qj, -g_p)
    np.add.at(grads["b"], qi, g)
    np.add.at(grads["b"], qj, -g)
    grads["W"] = g_p.T @ z_diff

    for name in ("P", "Q", "b"):
        grads[name] += lambda_f * params[name]
    grads["W"] += lambda_w * params["W"]
    return loss, grads
