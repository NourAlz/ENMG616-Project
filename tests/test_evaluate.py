"""Recall@10 / NDCG@10: hand-checked rankings, tie breaking and the optimizer hook."""

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.evaluate import (
    evaluate_model,
    evaluate_ranking,
    make_evaluator,
    ndcg_at_k,
    rank,
    recall_at_k,
    select_checkpoint,
)
from recommender.model import init_params


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_hit_at_rank_one_and_miss():
    ranked = rank(np.arange(20), np.arange(20, dtype=float))
    assert ranked[0] == 19
    assert recall_at_k(ranked, [19]) == 1.0 and ndcg_at_k(ranked, [19]) == 1.0
    assert recall_at_k(ranked, [0]) == 0.0 and ndcg_at_k(ranked, [0]) == 0.0  # rank 20


def test_ties_break_by_product_id():
    assert list(rank(np.array([5, 3, 7]), np.ones(3))) == [3, 5, 7]


def test_hand_computed_values():
    ranked = np.arange(10)
    # relevant {1, 4, 50}: hits at ranks 2 and 5, ideal fills ranks 1..3
    dcg = 1 / np.log2(3) + 1 / np.log2(6)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert ndcg_at_k(ranked, [1, 4, 50]) == pytest.approx(dcg / idcg)
    assert recall_at_k(ranked, [1, 4, 50]) == pytest.approx(2 / 3)


def test_higher_rank_gives_higher_ndcg():
    ids = np.array([10, 20, 30])
    assert ndcg_at_k(rank(ids, [3.0, 2.0, 1.0]), [10]) > ndcg_at_k(rank(ids, [2.0, 3.0, 1.0]), [10])


def test_oracle_scorer_is_perfect_on_every_task(ds):
    for task in ("ordinary", "cold"):
        for split in ("validation", "test"):
            truth = ds.relevant[(task, split)]
            oracle = lambda u, ids: np.isin(ids, truth[u]).astype(float)
            result = evaluate_ranking(oracle, ds, task, split)
            assert result["ndcg"] == pytest.approx(1.0)
            assert result["n_users"] == len(truth)


def test_model_is_evaluated_on_cold_products(ds):
    result = evaluate_model(init_params(ds, 616), ds, "cold", "validation")
    assert 0.0 <= result["ndcg"] <= 1.0 and 0.0 <= result["recall"] <= 1.0


def test_hook_and_checkpoint_selection(ds):
    metrics = make_evaluator(ds)(init_params(ds, 616))
    assert set(metrics) == {"val_recall", "val_ndcg"}
    log = {"checkpoints": [{"work": 0, "val_ndcg": 0.1}, {"work": 5, "val_ndcg": 0.2},
                           {"work": 9, "val_ndcg": 0.2}]}
    assert select_checkpoint(log)["work"] == 5  # tie goes to less work
