"""Load and validate the supplied project data.

Files are read as they are, checked against the rules in the project brief,
and exposed through one read-only ``Dataset``. A file that breaks a rule raises
``DataError``; rows are never dropped.

* ``user_id`` (0..n_users-1) is the row of ``P``.
* ``product_id`` (0..n_products-1) is the row of ``Z``.
* ``q_index[product_id]`` is the row of ``Q`` and ``b`` for an ordinary product
  and -1 for a cold product (``q_i = b_i = 0``).

Run ``python -m recommender.data`` to print a summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSITIVE_RATING = 4  # a rating of 4 or 5 is a positive interaction
TEXT_DIM = 384
CATALOGS = ("ordinary", "cold_validation", "cold_test")
TASKS = ("ordinary", "cold")
SPLITS = ("validation", "test")


class DataError(ValueError):
    """The supplied data do not satisfy a rule stated in the project brief."""


def _check(condition, message: str) -> None:
    if not condition:
        raise DataError(message)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def candidate_catalog(task: str, split: str) -> str:
    """Name of the catalog whose products are ranked for a task and split."""
    _check(task in TASKS and split in SPLITS, f"unknown task/split: {task}/{split}")
    return "ordinary" if task == "ordinary" else f"cold_{split}"


@dataclass(frozen=True)
class Dataset:
    """Everything the model, optimizers and evaluators read. Arrays are read-only."""

    products: pd.DataFrame  # row r describes product_id r
    Z: np.ndarray  # (n_products, 384) float64 text vectors, unit norm
    q_index: np.ndarray  # (n_products,) row in Q and b, -1 for cold products
    catalog_ids: dict[str, np.ndarray]  # catalog name -> sorted product ids
    train: pd.DataFrame  # user_id, product_id, rating, timestamp
    validation: pd.DataFrame
    test: pd.DataFrame
    train_triplets: np.ndarray  # (N, 3): user_id, positive_id, comparison_id
    validation_triplets: np.ndarray  # same columns; never used for training
    relevant: dict[tuple[str, str], dict[int, np.ndarray]]  # (task, split) -> user -> ids
    train_history: dict[int, np.ndarray]  # user -> products reviewed in training, any rating

    @property
    def n_users(self) -> int:
        return len(self.train_history)

    @property
    def n_products(self) -> int:
        return len(self.products)

    @property
    def n_ordinary(self) -> int:
        return len(self.catalog_ids["ordinary"])

    def eval_users(self, task: str, split: str) -> list[int]:
        """Customers evaluated for a task and split, in increasing id order."""
        return sorted(self.relevant[(task, split)])

    def candidates(self, task: str, split: str, user_id: int) -> np.ndarray:
        """Products to rank for one customer: the task's whole catalog minus the
        products the customer reviewed in training, sorted by product id."""
        catalog = self.catalog_ids[candidate_catalog(task, split)]
        return np.setdiff1d(catalog, self.train_history[user_id], assume_unique=True)

    def train_positive_counts(self) -> np.ndarray:
        """Positive training reviews per product id (the popularity reference)."""
        positive = self.train.loc[self.train["rating"] >= POSITIVE_RATING, "product_id"]
        return np.bincount(positive.to_numpy(), minlength=self.n_products)

    def summary(self) -> str:
        sizes = "  ".join(f"{name}={len(ids)}" for name, ids in self.catalog_ids.items())
        lines = [
            f"customers            {self.n_users}",
            f"products             {self.n_products}  {sizes}",
            f"text vectors Z       {self.Z.shape} {self.Z.dtype}",
            f"train reviews        {len(self.train)}",
            f"validation reviews   {len(self.validation)}",
            f"test reviews         {len(self.test)}",
            f"train triplets       {len(self.train_triplets)}",
            f"validation triplets  {len(self.validation_triplets)}",
        ]
        for (task, split), users in self.relevant.items():
            n_relevant = sum(len(ids) for ids in users.values())
            lines.append(
                f"eval {task:8s} {split:10s}  {len(users)} customers, {n_relevant} relevant products"
            )
        return "\n".join(lines)


def _read_csv(path: Path, columns: list[str], may_be_empty: str | None = None) -> pd.DataFrame:
    _check(path.exists(), f"missing data file: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    missing = set(columns) - set(frame.columns)
    _check(not missing, f"{path.name}: missing columns {sorted(missing)}")
    frame = frame[columns]
    if may_be_empty:
        frame[may_be_empty] = frame[may_be_empty].fillna("")
    _check(not frame.isna().any().any(), f"{path.name}: contains empty values")
    return frame


def _pairs(frame: pd.DataFrame, product_column: str = "product_id") -> set[tuple[int, int]]:
    return set(zip(frame["user_id"].tolist(), frame[product_column].tolist()))


def _positive_pairs(reviews: pd.DataFrame) -> set[tuple[int, int]]:
    return _pairs(reviews[reviews["rating"] >= POSITIVE_RATING])


def _read_products(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    # Some products have a title but no description; their text vector still exists.
    columns = ["product_id", "parent_asin", "title", "description", "catalog", "q_index"]
    products = _read_csv(path, columns, may_be_empty="description")
    _check(
        np.array_equal(products["product_id"], np.arange(len(products))),
        "products.csv: product_id must run 0..n-1 in file order",
    )
    _check(products["catalog"].isin(CATALOGS).all(), "products.csv: unknown catalog value")
    q_index = products["q_index"].to_numpy(dtype=np.int64)
    is_ordinary = (products["catalog"] == "ordinary").to_numpy()
    _check(
        np.array_equal(np.sort(q_index[is_ordinary]), np.arange(is_ordinary.sum())),
        "products.csv: ordinary q_index values must be exactly 0..n_ordinary-1",
    )
    _check((q_index[~is_ordinary] == -1).all(), "products.csv: cold q_index must be -1")
    catalog_ids = {
        name: _frozen(np.flatnonzero((products["catalog"] == name).to_numpy()))
        for name in CATALOGS
    }
    return products, _frozen(q_index), catalog_ids


def _read_features(path: Path, n_products: int) -> np.ndarray:
    _check(path.exists(), f"missing data file: {path}")
    with np.load(path) as archive:
        _check({"Z", "product_ids"} <= set(archive.files), f"{path.name}: missing arrays")
        Z, product_ids = archive["Z"], archive["product_ids"]
    _check(Z.shape == (n_products, TEXT_DIM), f"{path.name}: Z has shape {Z.shape}")
    _check(
        np.array_equal(product_ids, np.arange(n_products)),
        f"{path.name}: row r of Z must belong to product_id r",
    )
    Z = Z.astype(np.float64)  # exact conversion: the supplied values are unchanged
    _check(
        np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-4),
        f"{path.name}: text vectors must be finite with unit norm",
    )
    return _frozen(Z)


def _read_reviews(path: Path, allowed: list[np.ndarray], allowed_name: str) -> pd.DataFrame:
    reviews = _read_csv(path, ["user_id", "product_id", "rating", "timestamp"])
    _check(reviews["rating"].isin(range(1, 6)).all(), f"{path.name}: ratings must be 1 to 5")
    _check(
        reviews["product_id"].isin(np.concatenate(allowed)).all(),
        f"{path.name}: contains products outside {allowed_name}",
    )
    _check(
        not reviews.duplicated(["user_id", "product_id"]).any(),
        f"{path.name}: a customer reviews the same product twice",
    )
    return reviews


def _read_triplets(
    path: Path,
    positives: set[tuple[int, int]],
    train_pairs: set[tuple[int, int]],
    ordinary: np.ndarray,
) -> np.ndarray:
    """A triplet's positive must be rated 4-5 by that customer in the matching
    split; its comparison must be an ordinary product they did not review in training."""
    frame = _read_csv(path, ["user_id", "positive_id", "comparison_id"])
    _check(
        frame[["positive_id", "comparison_id"]].isin(ordinary).all().all(),
        f"{path.name}: positive_id and comparison_id must be ordinary products",
    )
    _check(
        _pairs(frame, "positive_id") <= positives,
        f"{path.name}: a positive_id is not rated 4 or 5 by that customer",
    )
    _check(
        not (_pairs(frame, "comparison_id") & train_pairs),
        f"{path.name}: a comparison_id was reviewed by that customer in training",
    )
    return _frozen(frame.to_numpy(dtype=np.int64))


def _read_relevant(
    path: Path,
    held_out: dict[str, pd.DataFrame],
    catalog_ids: dict[str, np.ndarray],
    train_history: dict[int, np.ndarray],
) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    """Relevant sets per (task, split) and customer. A relevant product must be
    a candidate (in the task's catalog, not reviewed in training) rated 4-5 in
    that split, and every customer must have at least ten candidates."""
    evaluation = _read_csv(path, ["task", "split", "user_id", "relevant_product_id"])
    _check(not evaluation.duplicated().any(), f"{path.name}: duplicated rows")
    relevant = {}
    for task in TASKS:
        for split in SPLITS:
            where = f"{path.name} [{task}/{split}]"
            rows = evaluation[(evaluation["task"] == task) & (evaluation["split"] == split)]
            _check(
                _pairs(rows, "relevant_product_id") <= _positive_pairs(held_out[split]),
                f"{where}: relevant product not rated 4 or 5 by that customer in {split}.csv",
            )
            _check(
                set(rows["user_id"]) <= set(train_history),
                f"{where}: an evaluated customer has no training history",
            )
            catalog = catalog_ids[candidate_catalog(task, split)]
            by_user = {}
            for user, ids in rows.groupby("user_id")["relevant_product_id"]:
                candidates = np.setdiff1d(catalog, train_history[user], assume_unique=True)
                _check(len(candidates) >= 10, f"{where}: customer {user} has under ten candidates")
                _check(
                    ids.isin(candidates).all(),
                    f"{where}: customer {user} has a relevant product that is not a candidate",
                )
                by_user[int(user)] = _frozen(np.sort(ids.to_numpy(dtype=np.int64)))
            relevant[(task, split)] = by_user
    _check(len(evaluation) == sum(len(ids) for users in relevant.values() for ids in users.values()),
           f"{path.name}: unknown task or split")
    return relevant


def load_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    """Read the eight supplied files, check them, and return a ``Dataset``."""
    data_dir = Path(data_dir)
    products, q_index, catalog_ids = _read_products(data_dir / "products.csv")
    Z = _read_features(data_dir / "product_features.npz", len(products))

    ordinary = catalog_ids["ordinary"]
    train = _read_reviews(data_dir / "train.csv", [ordinary], "the ordinary catalog")
    held_out = {
        split: _read_reviews(
            data_dir / f"{split}.csv",
            [ordinary, catalog_ids[f"cold_{split}"]],
            f"the ordinary and cold-{split} catalogs",
        )
        for split in SPLITS
    }
    train_history = {
        int(user): _frozen(np.sort(ids.to_numpy(dtype=np.int64)))
        for user, ids in train.groupby("user_id")["product_id"]
    }
    _check(
        list(train_history) == list(range(len(train_history))),
        "train.csv: user_id must cover 0..n_users-1 with no gaps",
    )
    for split, reviews in held_out.items():
        _check(
            reviews["user_id"].isin(list(train_history)).all(),
            f"{split}.csv: customer without training history",
        )

    train_pairs = _pairs(train)
    return Dataset(
        products=products,
        Z=Z,
        q_index=q_index,
        catalog_ids=catalog_ids,
        train=train,
        validation=held_out["validation"],
        test=held_out["test"],
        train_triplets=_read_triplets(
            data_dir / "train_triplets.csv", _positive_pairs(train), train_pairs, ordinary
        ),
        validation_triplets=_read_triplets(
            data_dir / "validation_triplets.csv",
            _positive_pairs(held_out["validation"]),
            train_pairs,
            ordinary,
        ),
        relevant=_read_relevant(data_dir / "evaluation.csv", held_out, catalog_ids, train_history),
        train_history=train_history,
    )


if __name__ == "__main__":
    print(load_dataset().summary())
