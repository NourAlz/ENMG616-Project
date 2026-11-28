"""Cold-start baseline, frozen checkpoints and their evaluation."""

import numpy as np
import pytest

from recommender.baselines import description_similarity, popularity, random_scores, user_profiles
from recommender.data import POSITIVE_RATING, load_dataset
from recommender.evaluate import evaluate_ranking
from recommender.frozen import evaluate_frozen, load_frozen, save_frozen
from recommender.model import init_params
from recommender.sgd import SGDConfig, train_sgd


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_profile_is_mean_of_liked_text_vectors(ds):
    profiles = user_profiles(ds)
    train = ds.train
    liked = train[(train.user_id == 0) & (train.rating >= POSITIVE_RATING)].product_id.to_numpy()
    np.testing.assert_allclose(profiles[0], ds.Z[liked].mean(axis=0))


def test_description_similarity_is_mean_cosine(ds):
    score = description_similarity(ds)
    train = ds.train
    liked = train[(train.user_id == 3) & (train.rating >= POSITIVE_RATING)].product_id.to_numpy()
    cold = ds.catalog_ids["cold_validation"][:5]
    expected = [np.mean([ds.Z[i] @ ds.Z[j] for j in liked]) for i in cold]
    np.testing.assert_allclose(score(3, cold), expected)


def test_description_similarity_uses_no_held_out_data(ds):
    """Changing validation/test reviews cannot change the baseline's scores."""
    cold = ds.catalog_ids["cold_test"]
    before = description_similarity(ds)(5, cold)
    shuffled = ds.__class__(**{**ds.__dict__, "validation": ds.validation.iloc[::-1],
                               "test": ds.test.iloc[:0]})
    np.testing.assert_array_equal(description_similarity(shuffled)(5, cold), before)


def test_baselines_on_cold_validation_are_valid(ds):
    for scorer in (description_similarity(ds), random_scores(616)):
        result = evaluate_ranking(scorer, ds, "cold", "validation")
        assert 0.0 <= result["ndcg"] <= 1.0 and result["n_users"] == 119


def test_popularity_matches_train_counts(ds):
    ids = ds.catalog_ids["ordinary"][:10]
    np.testing.assert_array_equal(popularity(ds)(0, ids), ds.train_positive_counts()[ids])


def test_freeze_round_trip_and_no_overwrite(ds, tmp_path):
    log = train_sgd(ds, 616, SGDConfig(0.5, 1e-3, 1e-3), total_work=2_000, checkpoint_every=1_000,
                    evaluate=lambda p: {"val_ndcg": float(np.sum(p["b"])), "val_recall": 0.0})
    name = save_frozen(log, tmp_path)
    params, meta = load_frozen(name, tmp_path)
    best = max(log["checkpoints"], key=lambda c: (c["val_ndcg"], -c["work"]))
    assert name == "sgd_seed616" and meta["work"] == best["work"]
    for k in params:
        np.testing.assert_array_equal(params[k], best["params"][k])
    with pytest.raises(FileExistsError):
        save_frozen(log, tmp_path)


def test_evaluate_frozen_reports_models_and_baselines(ds, tmp_path):
    log = {"optimizer": "demo", "seed": 616, "config": {}, "status": "ok",
           "checkpoints": [{"work": 0, "updates": 0, "train_time": 0.0, "train_loss": 0.69,
                            "val_loss": 0.69, "val_ndcg": 0.1, "val_recall": 0.1,
                            "params": init_params(ds, 616)}]}
    save_frozen(log, tmp_path)
    rows = evaluate_frozen(["demo_seed616"], ds, "cold", "validation", tmp_path)
    assert [r["model"] for r in rows] == ["demo_seed616", "description_similarity", "random"]
    ordinary = evaluate_frozen(["demo_seed616"], ds, "ordinary", "validation", tmp_path)
    assert [r["model"] for r in ordinary] == ["demo_seed616", "random"]
