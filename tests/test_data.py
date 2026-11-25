"""The loaded data must match the counts and rules stated in the project brief."""

import shutil

import numpy as np
import pandas as pd
import pytest

from recsys.data import DATA_DIR, DataError, load_dataset


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_counts_match_brief(ds):
    assert ds.n_users == 170
    assert ds.n_products == 286
    assert ds.n_ordinary == 206
    assert len(ds.catalog_ids["cold_validation"]) == 40
    assert len(ds.catalog_ids["cold_test"]) == 40
    assert len(ds.train) == 1897
    assert len(ds.validation) == 358
    assert len(ds.test) == 362
    assert ds.train_triplets.shape == (5436, 3)
    assert ds.validation_triplets.shape == (121, 3)


def test_evaluation_customer_counts_match_brief(ds):
    assert len(ds.eval_users("ordinary", "validation")) == 121
    assert len(ds.eval_users("ordinary", "test")) == 110
    assert len(ds.eval_users("cold", "validation")) == 119
    assert len(ds.eval_users("cold", "test")) == 114


def test_features(ds):
    assert ds.Z.shape == (286, 384)
    assert ds.Z.dtype == np.float64
    with np.load(DATA_DIR / "product_features.npz") as archive:
        assert np.array_equal(ds.Z, archive["Z"])  # values unchanged


def test_q_index(ds):
    ordinary = ds.catalog_ids["ordinary"]
    assert np.array_equal(np.sort(ds.q_index[ordinary]), np.arange(206))
    cold = np.concatenate([ds.catalog_ids["cold_validation"], ds.catalog_ids["cold_test"]])
    assert (ds.q_index[cold] == -1).all()


def test_three_comparisons_per_positive_training_review(ds):
    triplets = pd.DataFrame(ds.train_triplets, columns=["u", "i", "j"])
    per_positive = triplets.groupby(["u", "i"])["j"].agg(["size", "nunique"])
    assert (per_positive["size"] == 3).all()
    assert (per_positive["nunique"] == 3).all()
    assert len(per_positive) == (ds.train["rating"] >= 4).sum()


def test_cold_products_never_appear_in_training(ds):
    assert (ds.q_index[ds.train["product_id"]] >= 0).all()
    assert (ds.q_index[ds.train_triplets[:, 1:]] >= 0).all()


def test_candidates_exclude_training_history(ds):
    for task, split in [("ordinary", "validation"), ("ordinary", "test")]:
        for user in ds.eval_users(task, split):
            candidates = ds.candidates(task, split, user)
            assert not np.isin(candidates, ds.train_history[user]).any()
            assert len(candidates) == 206 - len(ds.train_history[user])
            assert np.isin(ds.relevant[(task, split)][user], candidates).all()
            assert (np.diff(candidates) > 0).all()  # sorted by product id


def test_cold_candidates_are_the_whole_cold_catalog(ds):
    for split, catalog in [("validation", "cold_validation"), ("test", "cold_test")]:
        for user in ds.eval_users("cold", split):
            assert np.array_equal(ds.candidates("cold", split, user), ds.catalog_ids[catalog])


def test_popularity_counts(ds):
    counts = ds.train_positive_counts()
    assert counts.shape == (286,)
    assert counts.sum() == (ds.train["rating"] >= 4).sum()
    assert (counts[ds.q_index == -1] == 0).all()


def test_arrays_are_read_only(ds):
    for array in (ds.Z, ds.q_index, ds.train_triplets, ds.validation_triplets):
        with pytest.raises(ValueError):
            array[0] = 0


def test_corrupted_file_is_rejected(tmp_path):
    for path in DATA_DIR.iterdir():
        shutil.copy(path, tmp_path / path.name)
    triplets = pd.read_csv(tmp_path / "train_triplets.csv")
    cold_product = 1  # catalog cold_validation
    triplets.loc[0, "comparison_id"] = cold_product
    triplets.to_csv(tmp_path / "train_triplets.csv", index=False)
    with pytest.raises(DataError, match="ordinary"):
        load_dataset(tmp_path)
