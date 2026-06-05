"""Compute a battery of diagnostics for a single fitted ALS model.

Each metric illuminates a different axis of recommender behavior:

- ``ndcg`` / ``recall`` / ``map`` — held-out ranking quality (the *outcome* metrics).
- ``train_test_ndcg_gap`` — memorization vs. generalization (overfitting axis).
- ``catalog_coverage`` — share of the catalog recommended across the *sampled* eval users
  (bounded above by ``EVAL_SAMPLE_USERS * k / n_items``); a relative diversity signal compared
  like-for-like across configs, not an absolute whole-catalog coverage.
- ``avg_rec_popularity`` — mean popularity percentile of recommendations (popularity-bias axis).
- ``novelty`` — mean self-information of recommendations (long-tail exposure axis).
- ``tail_recall`` — recall restricted to users whose held-out item is unpopular (tail-serving axis).
- ``mean_factor_norm`` — embedding magnitude (regularization axis), as the geometric mean of the
  mean user and item factor norms. The geometric mean is used because matrix factorization has a
  scaling gauge freedom (``x_u . y_i`` is invariant under ``x_u -> c x_u, y_i -> y_i / c``), so the
  user and item norms individually are not meaningful — their product is. It decreases monotonically
  as regularization increases.
- ``train_recon_error`` — confidence-weighted fit error on observed entries (convergence/fit axis).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from holmes.config import EVAL_SAMPLE_USERS

if TYPE_CHECKING:
    import scipy.sparse as sp

    from holmes.als.model import ALSRecommender
    from holmes.data.dataset import Dataset

_RECON_SAMPLE = 50_000
"""Cap on observed entries sampled when estimating reconstruction error."""


def _discounts(k: int) -> np.ndarray:
    """Return the DCG positional discounts ``1 / log2(rank + 2)`` for ranks ``0..k-1``."""
    return 1.0 / np.log2(np.arange(2, k + 2))


def _single_relevant_metrics(rec_ids: np.ndarray, relevant: np.ndarray, k: int) -> dict[str, float]:
    """Ranking metrics when each user has exactly one relevant (held-out) item.

    Args:
        rec_ids: Recommended item indices, shape ``(n_users, k)``.
        relevant: The single relevant item per user, shape ``(n_users,)``.
        k: Ranking cut-off.

    Returns:
        dict[str, float]: ``ndcg``, ``recall``, and ``map`` averaged over users.
    """
    hits = rec_ids == relevant[:, None]  # (n_users, k)
    hit_any = hits.any(axis=1)
    # Rank (0-indexed) of the hit, where it exists.
    first_hit_rank = np.argmax(hits, axis=1)
    discounts = _discounts(k)
    ndcg = np.where(hit_any, discounts[first_hit_rank], 0.0)  # ideal DCG == 1 for one relevant
    average_precision = np.where(hit_any, 1.0 / (first_hit_rank + 1.0), 0.0)
    return {
        "ndcg": float(ndcg.mean()),
        "recall": float(hit_any.mean()),
        "map": float(average_precision.mean()),
    }


def _train_memorization_ndcg(rec_train: np.ndarray, train_ui: sp.csr_matrix, users: np.ndarray, k: int) -> float:
    """Mean NDCG@k of the unfiltered recommendations against each user's training history.

    A high value (relative to held-out NDCG) signals the model memorizes training interactions.
    Vectorized via sparse fancy indexing: membership of every recommended item in a user's training
    row is looked up in one gather rather than building a Python set per user.

    Args:
        rec_train: Unfiltered top-k recommendations per user, shape ``(n_users, k)``.
        train_ui: The training matrix supplying each user's relevant (interacted) items.
        users: The user indices that ``rec_train`` rows correspond to, shape ``(n_users,)``.
        k: Ranking cut-off.

    Returns:
        float: Mean NDCG@k across the supplied users.
    """
    n_users = len(users)
    gathered = np.asarray(train_ui[np.repeat(users, k), rec_train.ravel()]).reshape(n_users, k)
    hits = gathered > 0

    discounts = _discounts(k)
    dcg = (hits * discounts).sum(axis=1)
    # Relevant items per user = nonzeros in its training row, read straight from indptr (no copy).
    relevant_counts = train_ui.indptr[users + 1] - train_ui.indptr[users]
    ideal_positions = np.clip(np.minimum(relevant_counts, k), 1, k)
    ideal_dcg = np.cumsum(discounts)[ideal_positions - 1]
    return float(np.mean(dcg / ideal_dcg))


def _reconstruction_error(model: ALSRecommender, dataset: Dataset, seed: int) -> float:
    """Mean squared preference error ``(1 - xu . yi)^2`` on sampled observed interactions.

    A high or seed-unstable value indicates the factorization has not fit the observed
    positives — typically too few iterations or factors. Samples observed entries directly from
    the CSR ``indptr``/``indices`` rather than expanding the whole matrix to COO, so cost is
    ``O(_RECON_SAMPLE)`` regardless of how many millions of nonzeros ``train_ui`` holds.

    Args:
        model: A fitted recommender.
        dataset: The dataset whose training matrix supplies observed entries.
        seed: Seed for sampling observed entries.

    Returns:
        float: Mean squared error against the implicit target preference of 1.
    """
    train_ui = dataset.train_ui
    n_obs = train_ui.nnz
    rng = np.random.default_rng(seed)
    flat = rng.choice(n_obs, size=_RECON_SAMPLE, replace=False) if n_obs > _RECON_SAMPLE else np.arange(n_obs)
    cols = train_ui.indices[flat]
    rows = np.searchsorted(train_ui.indptr, flat, side="right") - 1
    preds = np.einsum("ij,ij->i", model.user_factors[rows], model.item_factors[cols])
    return float(np.mean((1.0 - preds) ** 2))


def compute_diagnostics(
    model: ALSRecommender,
    dataset: Dataset,
    k: int,
    *,
    split: str = "test",
    seed: int = 0,
) -> dict[str, float]:
    """Compute the full diagnostic battery for one fitted model.

    Args:
        model: A recommender already fit on ``dataset.train_ui``.
        dataset: The dataset providing held-out positives and popularity.
        k: Ranking cut-off for all top-k metrics.
        split: Either ``"val"`` or ``"test"``; selects the held-out positives to score.
        seed: Seed for evaluation-user and reconstruction sampling.

    Returns:
        dict[str, float]: One scalar per diagnostic (see module docstring).

    Raises:
        ValueError: If ``split`` is not ``"val"`` or ``"test"``.
    """
    if split == "test":
        held_users, held_items = dataset.test_users, dataset.test_items
    elif split == "val":
        held_users, held_items = dataset.val_users, dataset.val_items
    else:
        msg = f"split must be 'val' or 'test', got {split!r}."
        raise ValueError(msg)

    # Sample positions into the parallel held-out arrays — keeps users and their relevant items
    # aligned with no per-call array sized to all n_users.
    n_held = len(held_users)
    if n_held > EVAL_SAMPLE_USERS:
        selection = np.random.default_rng(seed).choice(n_held, size=EVAL_SAMPLE_USERS, replace=False)
        users, relevant = held_users[selection], held_items[selection]
    else:
        users, relevant = held_users, held_items

    rec = model.recommend(users, dataset.train_ui, k, filter_seen=True)
    outcome = _single_relevant_metrics(rec, relevant, k)

    # Memorization: re-rank without filtering seen items; relevance = the training history.
    rec_train = model.recommend(users, dataset.train_ui, k, filter_seen=False)
    train_ndcg = _train_memorization_ndcg(rec_train, dataset.train_ui, users, k)

    # Diversity and popularity bias (percentiles are cached on the dataset, computed once).
    coverage = len(np.unique(rec)) / dataset.n_items
    avg_rec_popularity = float(dataset.popularity_percentile[rec].mean())
    pop_safe = np.maximum(dataset.item_popularity, 1) / dataset.n_users
    novelty = float((-np.log2(pop_safe))[rec].mean())

    # Tail serving: recall over users whose held-out item is below the head-popularity cutoff.
    # No tail users in the sample folds to 0.0 (an empty-denominator case that, with thousands of
    # sampled users and a tail spanning most of the catalog, effectively never occurs on real data).
    tail_mask = dataset.item_popularity[relevant] < dataset.head_item_threshold
    if tail_mask.any():
        tail_hits = (rec[tail_mask] == relevant[tail_mask, None]).any(axis=1)
        tail_recall = float(tail_hits.mean())
    else:
        tail_recall = 0.0

    # Gauge-invariant embedding magnitude: geometric mean of the user and item factor norms.
    mean_user_norm = float(np.linalg.norm(model.user_factors, axis=1).mean())
    mean_item_norm = float(np.linalg.norm(model.item_factors, axis=1).mean())
    mean_factor_norm = float(np.sqrt(mean_user_norm * mean_item_norm))

    return {
        "ndcg": outcome["ndcg"],
        "recall": outcome["recall"],
        "map": outcome["map"],
        "train_ndcg": train_ndcg,
        "train_test_ndcg_gap": train_ndcg - outcome["ndcg"],
        "catalog_coverage": coverage,
        "avg_rec_popularity": avg_rec_popularity,
        "novelty": novelty,
        "tail_recall": tail_recall,
        "mean_factor_norm": mean_factor_norm,
        "train_recon_error": _reconstruction_error(model, dataset, seed),
    }
