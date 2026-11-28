"""Reference scorers. Each returns ``score(user_id, product_ids) -> scores`` for
``evaluate.evaluate_ranking``; higher means recommended first, ties go to the
smaller product id.

* ``popularity``   positive training reviews per product (ordinary task only:
                   every cold product has zero and the ranking degenerates
                   to product-id order).
* ``random_scores`` i.i.d. uniform scores, a floor for any task.
* ``description_similarity``  the cold-start baseline. A customer's profile is
                   the mean text vector of the products they rated 4 or 5 in
                   training; a product's score is the inner product of its
                   text vector with that profile:

                       s_ui = z_i . (1/|R_u|) sum_{j in R_u} z_j

                   With unit-norm z this is the mean cosine similarity to the
                   customer's liked products. It ranks exactly like the cosine
                   between z_i and the profile, because per-customer scaling
                   does not change a ranking. It uses only training reviews
                   and the supplied text vectors, so it needs no fitting.
                   A customer with no positive training review would fall
                   back to every product they reviewed (none do in the
                   supplied data).
"""

from __future__ import annotations

import numpy as np

from recommender.data import POSITIVE_RATING


def popularity(data):
    counts = data.train_positive_counts().astype(float)
    return lambda user, ids: counts[np.asarray(ids)]


def random_scores(seed: int):
    rng = np.random.default_rng(seed)
    return lambda user, ids: rng.random(len(ids))


def user_profiles(data) -> np.ndarray:
    """(n_users, 384) mean text vector of each customer's positive training products."""
    positive = data.train[data.train["rating"] >= POSITIVE_RATING]
    liked = positive.groupby("user_id")["product_id"].apply(np.asarray).to_dict()
    profiles = np.empty((data.n_users, data.Z.shape[1]))
    for user in range(data.n_users):
        ids = liked.get(user, data.train_history[user])
        profiles[user] = data.Z[ids].mean(axis=0)
    return profiles


def description_similarity(data):
    profiles = user_profiles(data)
    return lambda user, ids: data.Z[np.asarray(ids)] @ profiles[user]
