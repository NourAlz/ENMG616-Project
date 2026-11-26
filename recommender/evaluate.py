"""Recall@10 and NDCG@10 over full candidate sets.

For a customer u in a (task, split), every candidate in ``Dataset.candidates``
is scored with ``model.scores`` and ranked by score, highest first. Ties are
broken by product id, smallest first, so a ranking never depends on array
order.

    Recall@k = |top-k ∩ R_u| / |R_u|
    NDCG@k   = DCG@k / IDCG@k,  DCG@k = sum over hits at rank r of 1 / log2(r + 1),
               IDCG@k = sum_{r=1}^{min(|R_u|, k)} 1 / log2(r + 1)

Both are averaged over the customers evaluated for that (task, split).
"""

from __future__ import annotations

import numpy as np

from recommender.model import Params, scores

K = 10


def rank(candidate_ids: np.ndarray, candidate_scores: np.ndarray) -> np.ndarray:
    """Candidate ids by score descending, ties by product id ascending."""
    candidate_ids = np.asarray(candidate_ids)
    return candidate_ids[np.lexsort((candidate_ids, -np.asarray(candidate_scores)))]


def recall_at_k(ranked: np.ndarray, relevant, k: int = K) -> float:
    relevant = set(np.asarray(relevant).tolist())
    if not relevant:
        return 0.0
    return len(relevant.intersection(np.asarray(ranked[:k]).tolist())) / len(relevant)


def ndcg_at_k(ranked: np.ndarray, relevant, k: int = K) -> float:
    relevant = set(np.asarray(relevant).tolist())
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 2) for r, pid in enumerate(ranked[:k]) if int(pid) in relevant)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relevant), k)))
    return dcg / idcg


def evaluate_ranking(score_fn, data, task: str, split: str, k: int = K) -> dict:
    """Mean Recall@k and NDCG@k for any scorer ``score_fn(user_id, product_ids)``.
    Baselines (popularity, random, description similarity) use this directly."""
    recalls, ndcgs = [], []
    for user in data.eval_users(task, split):
        candidates = data.candidates(task, split, user)
        ranked = rank(candidates, score_fn(user, candidates))
        relevant = data.relevant[(task, split)][user]
        recalls.append(recall_at_k(ranked, relevant, k))
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
    return {"recall": float(np.mean(recalls)), "ndcg": float(np.mean(ndcgs)), "n_users": len(ndcgs)}


def evaluate_model(params: Params, data, task: str, split: str, k: int = K) -> dict:
    """Mean Recall@k and NDCG@k of the hybrid model."""
    return evaluate_ranking(lambda u, ids: scores(params, data, u, ids), data, task, split, k)


def make_evaluator(data, k: int = K):
    """The ``evaluate`` hook for the optimizers: ordinary-validation metrics only,
    the one signal allowed for checkpoint selection."""

    def evaluate(params: Params) -> dict:
        result = evaluate_model(params, data, "ordinary", "validation", k)
        return {"val_recall": result["recall"], "val_ndcg": result["ndcg"]}

    return evaluate


def select_checkpoint(log: dict) -> dict:
    """Checkpoint with the highest ordinary-validation NDCG@10; ties go to the
    earlier checkpoint (less gradient work)."""
    return max(log["checkpoints"], key=lambda c: (c["val_ndcg"], -c["work"]))
