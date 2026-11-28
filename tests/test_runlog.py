"""JSON run logs: everything but the parameters, readable back."""

import numpy as np
import pytest

from recommender.data import load_dataset
from recommender.runlog import load_log, save_log
from recommender.salr import SALRConfig, train_salr


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def test_round_trip_without_params(ds, tmp_path):
    metric = iter(np.linspace(0.1, 0.5, 10))
    log = train_salr(ds, 616, SALRConfig(0.5, 0.05, 1e-3, 1e-3), total_work=3_000,
                     checkpoint_every=1_000, evaluate=lambda p: {"val_ndcg": next(metric)})
    path = save_log(log, "salr_demo", tmp_path, experiment="test")
    saved = load_log(path)
    assert saved["optimizer"] == "salr" and saved["seed"] == 616 and saved["experiment"] == "test"
    assert saved["config"]["base_lr"] == 0.5
    assert saved["lr"] == pytest.approx(log["lr"]) and saved["probed"] == log["probed"]
    assert all("params" not in c for c in saved["checkpoints"])
    assert [c["work"] for c in saved["checkpoints"]] == [c["work"] for c in log["checkpoints"]]
    assert saved["selected_work"] == log["checkpoints"][-1]["work"]  # the metric increases


def test_non_finite_values_become_null(tmp_path):
    log = {"optimizer": "x", "seed": 1, "config": {}, "status": "failed", "lr": [np.inf, np.nan, 1.0],
           "checkpoints": [{"work": 0, "params": {"P": np.zeros(2)}}]}
    saved = load_log(save_log(log, "bad", tmp_path))
    assert saved["lr"] == [None, None, 1.0] and "selected_work" not in saved
