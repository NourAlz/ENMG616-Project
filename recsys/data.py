"""Load and validate the supplied project data.

Nothing here changes the supplied values: files are read as they are, checked
against the rules in the project brief, and exposed through one read-only
``Dataset``. A file that breaks a rule raises ``DataError`` instead of having
rows dropped silently.

Identifiers
-----------
* ``user_id`` runs from 0 to ``n_users - 1`` and is the row of ``P``.
* ``product_id`` runs from 0 to ``n_products - 1`` and is the row of ``Z``.
* ``q_index[product_id]`` is the row of ``Q`` and ``b`` for an ordinary
  product, and -1 for a cold product (meaning ``q_i = b_i = 0``).

Run ``python -m recsys.data`` to print a summary of the loaded data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

POSITIVE_RATING = 4  # a rating of 4 or 5 is a positive interaction
TEXT_DIM = 384

ORDINARY = "ordinary"
COLD_VALIDATION = "cold_validation"
COLD_TEST = "cold_test"
CATALOGS = (ORDINARY, COLD_VALIDATION, COLD_TEST)

TASKS = ("ordinary", "cold")
SPLITS = ("validation", "test")

REVIEW_COLUMNS = ["user_id", "product_id", "rating", "timestamp"]
TRIPLET_COLUMNS = ["user_id", "positive_id", "comparison_id"]
EVALUATION_COLUMNS = ["task", "split", "user_id", "relevant_product_id"]
PRODUCT_COLUMNS = ["product_id", "parent_asin", "title", "description", "catalog", "q_index"]


class DataError(ValueError):
    """The supplied data do not satisfy a rule stated in the project brief."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DataError(message)


def _frozen(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def candidate_catalog(task: str, split: str) -> str:
    """Name of the catalog whose products are ranked for a task and split."""
    _check(task in TASKS, f"unknown task {task!r}; expected one of {TASKS}")
    _check(split in SPLITS, f"unknown split {split!r}; expected one of {SPLITS}")
    if task == "ordinary":
        return ORDINARY
    return COLD_VALIDATION if split == "validation" else COLD_TEST


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

    train_triplets: np.ndarray  # (N, 3) columns: user_id, positive_id, comparison_id
    validation_triplets: np.ndarray  # same columns; never used for training

    # (task, split) -> user_id -> sorted relevant product ids
    relevant: dict[tuple[str, str], dict[int, np.ndarray]]
    # user_id -> sorted product ids the user reviewed in training, any rating
    train_history: dict[int, np.ndarray]

    n_users: int

    @property
    def n_products(self) -> int:
        return len(self.products)

    @property
    def n_ordinary(self) -> int:
        return len(self.catalog_ids[ORDINARY])

    def eval_users(self, task: str, split: str) -> list[int]:
        """Customers evaluated for a task and split, in increasing id order."""
        candidate_catalog(task, split)
        return sorted(self.relevant[(task, split)])

    def candidates(self, task: str, split: str, user_id: int) -> np.ndarray:
        """Products to rank for one customer: the task's whole catalog, minus
        products the customer reviewed in training. Sorted by product id."""
        catalog = self.catalog_ids[candidate_catalog(task, split)]
        return np.setdiff1d(catalog, self.train_history[user_id], assume_unique=True)

    def train_positive_counts(self) -> np.ndarray:
        """Number of positive training reviews per product, indexed by product id.
        This is the score used by the popularity reference ranking."""
        positive = self.train.loc[self.train["rating"] >= POSITIVE_RATING, "product_id"]
        return np.bincount(positive.to_numpy(), minlength=self.n_products)

    def summary(self) -> str:
        lines = [
            f"customers            {self.n_users}",
            f"products             {self.n_products}  "
            + "  ".join(f"{name}={len(ids)}" for name, ids in self.catalog_ids.items()),
            f"text vectors Z       {self.Z.shape} {self.Z.dtype}",
            f"train reviews        {len(self.train)}",
            f"validation reviews   {len(self.validation)}",
            f"test reviews         {len(self.test)}",
            f"train triplets       {len(self.train_triplets)}",
            f"validation triplets  {len(self.validation_triplets)}",
        ]
        for task in TASKS:
            for split in SPLITS:
                users = self.relevant[(task, split)]
                n_relevant = sum(len(ids) for ids in users.values())
                lines.append(
                    f"eval {task:8s} {split:10s}  {len(users)} customers, "
                    f"{n_relevant} relevant products"
                )
        return "\n".join(lines)


# ── Readers ────────────────────────────────────────────────────────────────


def _read_csv(path: Path, columns: list[str], may_be_empty: tuple[str, ...] = ()) -> pd.DataFrame:
    _check(path.exists(), f"missing data file: {path}")
    frame = pd.read_csv(path, encoding="utf-8")
    missing = set(columns) - set(frame.columns)
    _check(not missing, f"{path.name}: missing columns {sorted(missing)}")
    frame = frame[columns]
    required = [column for column in columns if column not in may_be_empty]
    empty = frame[required].isna().any()
    _check(not empty.any(), f"{path.name}: empty values in {sorted(empty[empty].index)}")
    for column in may_be_empty:
        frame[column] = frame[column].fillna("")
    return frame


def _read_products(path: Path) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    # Some products have a title but no description; their text vector still exists.
    products = _read_csv(path, PRODUCT_COLUMNS, may_be_empty=("description",))
    n = len(products)
    _check(
        np.array_equal(products["product_id"].to_numpy(), np.arange(n)),
        "products.csv: product_id must run 0..n-1 in file order",
    )
    unknown = set(products["catalog"]) - set(CATALOGS)
    _check(not unknown, f"products.csv: unknown catalog values {sorted(unknown)}")

    q_index = products["q_index"].to_numpy(dtype=np.int64)
    is_ordinary = (products["catalog"] == ORDINARY).to_numpy()
    _check(
        np.array_equal(np.sort(q_index[is_ordinary]), np.arange(is_ordinary.sum())),
        "products.csv: ordinary q_index values must be exactly 0..n_ordinary-1",
    )
    _check(
        bool(np.all(q_index[~is_ordinary] == -1)),
        "products.csv: cold products must have q_index = -1",
    )

    catalog_ids = {
        name: _frozen(np.flatnonzero((products["catalog"] == name).to_numpy()))
        for name in CATALOGS
    }
    return products, _frozen(q_index), catalog_ids


def _read_features(path: Path, n_products: int) -> np.ndarray:
    _check(path.exists(), f"missing data file: {path}")
    with np.load(path) as archive:
        missing = {"Z", "product_ids"} - set(archive.files)
        _check(not missing, f"{path.name}: missing arrays {sorted(missing)}")
        Z = archive["Z"]
        product_ids = archive["product_ids"]
    _check(Z.shape == (n_products, TEXT_DIM), f"{path.name}: Z has shape {Z.shape}")
    _check(
        np.array_equal(product_ids, np.arange(n_products)),
        f"{path.name}: row r of Z must belong to product_id r",
    )
    _check(bool(np.isfinite(Z).all()), f"{path.name}: Z contains non-finite values")
    norms = np.linalg.norm(Z.astype(np.float64), axis=1)
    _check(
        bool(np.allclose(norms, 1.0, atol=1e-4)),
        f"{path.name}: text vectors must have unit norm "
        f"(found {norms.min():.6f} to {norms.max():.6f})",
    )
    # float32 -> float64 is exact, so the supplied values are unchanged.
    return _frozen(Z.astype(np.float64))


def _read_reviews(
    path: Path, n_products: int, allowed: np.ndarray, allowed_name: str
) -> pd.DataFrame:
    reviews = _read_csv(path, REVIEW_COLUMNS)
    _check(
        bool(reviews["product_id"].between(0, n_products - 1).all()),
        f"{path.name}: product_id outside the catalog",
    )
    _check(
        bool(reviews["rating"].isin([1, 2, 3, 4, 5]).all()),
        f"{path.name}: ratings must be integers 1 to 5",
    )
    _check(
        bool(reviews["product_id"].isin(allowed).all()),
        f"{path.name}: contains products outside {allowed_name}",
    )
    _check(
        not reviews.duplicated(["user_id", "product_id"]).any(),
        f"{path.name}: a customer reviews the same product twice",
    )
    return reviews


def _read_triplets(path: Path) -> np.ndarray:
    return _frozen(_read_csv(path, TRIPLET_COLUMNS).to_numpy(dtype=np.int64))


# ── Cross-file checks ──────────────────────────────────────────────────────


def _pairs(frame: pd.DataFrame, product_column: str = "product_id") -> set[tuple[int, int]]:
    return set(zip(frame["user_id"].tolist(), frame[product_column].tolist()))


def _check_triplets(
    name: str,
    triplets: np.ndarray,
    positives: set[tuple[int, int]],
    train_pairs: set[tuple[int, int]],
    is_ordinary: np.ndarray,
    n_users: int,
) -> None:
    users, positive_ids, comparison_ids = triplets.T
    _check(bool(((users >= 0) & (users < n_users)).all()), f"{name}: user_id out of range")
    n_products = len(is_ordinary)
    for column, ids in (("positive_id", positive_ids), ("comparison_id", comparison_ids)):
        _check(
            bool(((ids >= 0) & (ids < n_products)).all()),
            f"{name}: {column} outside the catalog",
        )
        _check(bool(is_ordinary[ids].all()), f"{name}: {column} must be an ordinary product")
    _check(
        bool((positive_ids != comparison_ids).all()),
        f"{name}: a product is compared with itself",
    )
    user_positive = set(zip(users.tolist(), positive_ids.tolist()))
    _check(
        user_positive <= positives,
        f"{name}: a positive_id is not rated 4 or 5 by that customer in the matching split",
    )
    user_comparison = set(zip(users.tolist(), comparison_ids.tolist()))
    _check(
        not (user_comparison & train_pairs),
        f"{name}: a comparison_id was reviewed by that customer in training",
    )


def _read_relevant(
    path: Path,
    held_out: dict[str, pd.DataFrame],
    catalog_ids: dict[str, np.ndarray],
    train_pairs: set[tuple[int, int]],
    train_history: dict[int, np.ndarray],
) -> dict[tuple[str, str], dict[int, np.ndarray]]:
    evaluation = _read_csv(path, EVALUATION_COLUMNS)
    _check(bool(evaluation["task"].isin(TASKS).all()), "evaluation.csv: unknown task")
    _check(bool(evaluation["split"].isin(SPLITS).all()), "evaluation.csv: unknown split")
    _check(not evaluation.duplicated().any(), "evaluation.csv: duplicated rows")

    relevant: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    for task in TASKS:
        for split in SPLITS:
            where = f"evaluation.csv [{task}/{split}]"
            rows = evaluation[(evaluation["task"] == task) & (evaluation["split"] == split)]
            _check(len(rows) > 0, f"{where}: no rows")
            catalog = catalog_ids[candidate_catalog(task, split)]
            _check(
                bool(rows["relevant_product_id"].isin(catalog).all()),
                f"{where}: relevant product outside the candidate catalog",
            )
            reviews = held_out[split]
            positives = _pairs(reviews[reviews["rating"] >= POSITIVE_RATING])
            row_pairs = _pairs(rows, "relevant_product_id")
            _check(
                row_pairs <= positives,
                f"{where}: relevant product not rated 4 or 5 by that customer in {split}.csv",
            )
            _check(
                not (row_pairs & train_pairs),
                f"{where}: relevant product was reviewed by that customer in training",
            )
            by_user = {
                int(user): _frozen(np.sort(group.to_numpy(dtype=np.int64)))
                for user, group in rows.groupby("user_id")["relevant_product_id"]
            }
            _check(
                set(by_user) <= set(train_history),
                f"{where}: an evaluated customer has no training history",
            )
            for user in by_user:
                n_candidates = len(catalog) - np.isin(train_history[user], catalog).sum()
                _check(n_candidates >= 10, f"{where}: customer {user} has under ten candidates")
            relevant[(task, split)] = by_user
    return relevant


# ── Entry point ────────────────────────────────────────────────────────────


def load_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    """Read the eight supplied files, check them, and return a ``Dataset``."""
    data_dir = Path(data_dir)

    products, q_index, catalog_ids = _read_products(data_dir / "products.csv")
    n_products = len(products)
    Z = _read_features(data_dir / "product_features.npz", n_products)

    ordinary = catalog_ids[ORDINARY]
    train = _read_reviews(data_dir / "train.csv", n_products, ordinary, "the ordinary catalog")
    validation = _read_reviews(
        data_dir / "validation.csv",
        n_products,
        np.concatenate([ordinary, catalog_ids[COLD_VALIDATION]]),
        "the ordinary and cold-validation catalogs",
    )
    test = _read_reviews(
        data_dir / "test.csv",
        n_products,
        np.concatenate([ordinary, catalog_ids[COLD_TEST]]),
        "the ordinary and cold-test catalogs",
    )

    train_users = np.sort(train["user_id"].unique())
    n_users = len(train_users)
    _check(
        np.array_equal(train_users, np.arange(n_users)),
        "train.csv: user_id must cover 0..n_users-1 with no gaps",
    )
    for name, reviews in (("validation.csv", validation), ("test.csv", test)):
        _check(
            bool(reviews["user_id"].between(0, n_users - 1).all()),
            f"{name}: customer without training history",
        )

    train_history = {
        int(user): _frozen(np.sort(group.to_numpy(dtype=np.int64)))
        for user, group in train.groupby("user_id")["product_id"]
    }
    train_pairs = _pairs(train)
    is_ordinary = q_index >= 0

    train_triplets = _read_triplets(data_dir / "train_triplets.csv")
    _check_triplets(
        "train_triplets.csv",
        train_triplets,
        _pairs(train[train["rating"] >= POSITIVE_RATING]),
        train_pairs,
        is_ordinary,
        n_users,
    )
    validation_triplets = _read_triplets(data_dir / "validation_triplets.csv")
    _check_triplets(
        "validation_triplets.csv",
        validation_triplets,
        _pairs(validation[validation["rating"] >= POSITIVE_RATING]),
        train_pairs,
        is_ordinary,
        n_users,
    )

    relevant = _read_relevant(
        data_dir / "evaluation.csv",
        {"validation": validation, "test": test},
        catalog_ids,
        train_pairs,
        train_history,
    )

    return Dataset(
        products=products,
        Z=Z,
        q_index=q_index,
        catalog_ids=catalog_ids,
        train=train,
        validation=validation,
        test=test,
        train_triplets=train_triplets,
        validation_triplets=validation_triplets,
        relevant=relevant,
        train_history=train_history,
        n_users=n_users,
    )


if __name__ == "__main__":
    print(load_dataset().summary())
