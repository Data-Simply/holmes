"""Build a sparse user-item interaction matrix from Amazon Reviews 2023 (Books).

The raw reviews ship as one JSONL file per category (~20GB for Books) on Hugging Face; this dataset
has a loader script, so HF never auto-converted it to Parquet. On first run ``hf_hub_download`` fetches
the JSONL (resumable, cached by the HF hub) and Polars projects it to just the four columns the
pipeline needs, persisting a local Parquet **cache** — so subsequent runs ``scan_parquet`` the cache
and skip the multi-GB re-read. Filtering, deduplication, and aggregation then run through Polars' streaming
engine (vectorized in Rust, spilling larger-than-memory) rather than iterating tens of millions of
rows in Python. Each kept review is one positive interaction stored at its raw star rating
(``c_ui = 1 + α·rating`` in implicit ALS, so 5-star reviews carry stronger confidence than 4-star);
the matrix is k-core filtered and split leave-last-out by timestamp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import scipy.sparse as sp
from huggingface_hub import hf_hub_download

from holmes.config import RAW_CACHE_DIR
from holmes.data.dataset import Dataset

if TYPE_CHECKING:
    from pathlib import Path

HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
REVIEW_FILENAME_TEMPLATE = "raw/review_categories/{category}.jsonl"
"""In-repo path to a category's raw reviews JSONL (e.g. ``Books`` -> ``raw/review_categories/Books.jsonl``).

The dataset ships a loader script, so Hugging Face never auto-converted it to Parquet; the JSONL is
fetched from ``main`` via the HF hub cache (resumable, integrity-checked) and read by Polars."""

_USER = "user_id"
_ITEM = "parent_asin"
_RATING = "rating"
_TIME = "timestamp"
_USER_IDX = "user_idx"
_ITEM_IDX = "item_idx"
_POS = "_pos"
_GROUP_SIZE = "_group_size"
_ROLE = "_role"

_REVIEW_COLUMNS = (_USER, _ITEM, _RATING, _TIME)
"""The only columns the pipeline consumes; the rest of each review is dropped at cache build."""

_MIN_INTERACTIONS_FOR_VAL = 3
"""A user needs train + val + test (>=3 interactions) to contribute a validation positive."""


def review_filename(category: str) -> str:
    """Return the in-repo path to a category's raw reviews JSONL.

    Args:
        category: Amazon category name as it appears in the dataset (e.g. ``"Books"``).

    Returns:
        str: The repo-relative path, e.g. ``raw/review_categories/Books.jsonl``.
    """
    return REVIEW_FILENAME_TEMPLATE.format(category=category)


def _download_reviews(category: str) -> str:
    """Fetch a category's raw reviews JSONL via the HF hub cache, returning the local path.

    ``hf_hub_download`` transfers the file compressed, verifies it, and stores it in the HF hub
    cache, so a re-run reuses the local copy instead of re-downloading.

    Args:
        category: Amazon category name (e.g. ``"Books"``).

    Returns:
        str: Local filesystem path to the downloaded JSONL.
    """
    return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=review_filename(category))


def _build_review_cache(source: str, cache_path: Path) -> None:
    """Project the raw reviews JSONL to the needed columns and persist as a Parquet cache.

    Polars streams the JSONL and ``sink_parquet`` runs through the streaming engine, so the ~20GB of
    reviews is never materialized in memory; only :data:`_REVIEW_COLUMNS` are kept, making the cache
    far smaller than the source.

    Args:
        source: A Polars-readable NDJSON path or URI (e.g. the ``hf://`` reviews file).
        cache_path: Destination Parquet path for the projected reviews.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Projecting {list(_REVIEW_COLUMNS)} from {source}\n  -> {cache_path}")
    pl.scan_ndjson(source).select(_REVIEW_COLUMNS).sink_parquet(cache_path)


def _deduplicate_interactions(reviews: pl.LazyFrame, min_rating: float) -> pl.LazyFrame:
    """Filter to positive reviews and collapse repeats to one interaction per user-item pair.

    Args:
        reviews: Lazy frame of raw reviews with ``user_id``, ``parent_asin``, ``rating``,
            ``timestamp`` columns.
        min_rating: Minimum star rating for a review to count as a positive interaction.

    Returns:
        pl.LazyFrame: One row per ``(user_id, parent_asin)`` with the most recent timestamp.
    """
    return (
        reviews.select(_USER, _ITEM, _RATING, _TIME)
        .filter(pl.col(_RATING).is_not_null() & (pl.col(_RATING) >= min_rating))
        .filter(
            pl.col(_USER).is_not_null()
            & pl.col(_ITEM).is_not_null()
            & (pl.col(_USER) != "")
            & (pl.col(_ITEM) != ""),
        )
        .group_by(_USER, _ITEM)
        .agg(pl.col(_TIME).max(), pl.col(_RATING).max())
    )


def _k_core_filter(interactions: pl.DataFrame, min_user: int, min_item: int) -> pl.DataFrame:
    """Iteratively drop users and items below the interaction thresholds.

    Args:
        interactions: One row per interaction with ``user_id`` and ``parent_asin`` columns.
        min_user: Minimum interactions required to keep a user.
        min_item: Minimum interactions required to keep an item.

    Returns:
        pl.DataFrame: The densified frame; may be empty if thresholds are too strict.
    """
    while True:
        before = interactions.height
        interactions = interactions.filter(pl.len().over(_ITEM) >= min_item)
        interactions = interactions.filter(pl.len().over(_USER) >= min_user)
        if interactions.height in (before, 0):
            return interactions


def _assign_indices_and_split(interactions: pl.DataFrame) -> Dataset:
    """Map ids to contiguous indices and split leave-last-out per user by timestamp.

    Indices span the *pre-split* universe (every user/item in ``interactions``), so ``n_users`` /
    ``n_items`` / ``density`` describe that universe rather than the evaluable population. A user
    with a single interaction becomes a test positive with no training history; the semi-joins below
    drop it from val/test scoring but it keeps an all-zero ``train_ui`` row. With the default
    ``min_user``/``min_item`` of 5 every surviving user has >=3 interactions, so no user is stranded
    in production; the all-zero-row case only arises under a relaxed (sub-3) k-core.

    Args:
        interactions: Deduplicated, k-core-filtered interactions with ``user_id``,
            ``parent_asin``, and ``timestamp`` columns.

    Returns:
        Dataset: Training matrix plus validation and test held-out positives.
    """
    # Assign contiguous integer indices, then drop the wide string id columns immediately so every
    # downstream copy (sort, windows, filters, joins) carries only narrow integer columns.
    indexed = interactions.with_columns(
        (pl.col(_USER).rank("dense") - 1).cast(pl.Int32).alias(_USER_IDX),
        (pl.col(_ITEM).rank("dense") - 1).cast(pl.Int32).alias(_ITEM_IDX),
    ).select(_USER_IDX, _ITEM_IDX, _TIME, _RATING)
    # Sort by (user, time, item) — the _ITEM_IDX tiebreaker makes this a total, value-based order
    # so tied timestamps split deterministically regardless of the (unordered) group_by row order.
    indexed = indexed.sort(_USER_IDX, _TIME, _ITEM_IDX).with_columns(
        pl.int_range(pl.len()).over(_USER_IDX).alias(_POS),
        pl.len().over(_USER_IDX).alias(_GROUP_SIZE),
    )
    # Most recent interaction is the test positive; the second most recent (with enough history)
    # is the validation positive; the rest are training.
    role = (
        pl.when(pl.col(_POS) == pl.col(_GROUP_SIZE) - 1)
        .then(pl.lit("test"))
        .when((pl.col(_POS) == pl.col(_GROUP_SIZE) - 2) & (pl.col(_GROUP_SIZE) >= _MIN_INTERACTIONS_FOR_VAL))
        .then(pl.lit("val"))
        .otherwise(pl.lit("train"))
        .alias(_ROLE)
    )
    indexed = indexed.with_columns(role)

    n_users = int(indexed[_USER_IDX].max()) + 1
    n_items = int(indexed[_ITEM_IDX].max()) + 1

    train = indexed.filter(pl.col(_ROLE) == "train")
    train_users = train.select(_USER_IDX).unique()
    # Only keep held-out positives for users that still have training history.
    val = indexed.filter(pl.col(_ROLE) == "val").join(train_users, on=_USER_IDX, how="semi")
    test = indexed.filter(pl.col(_ROLE) == "test").join(train_users, on=_USER_IDX, how="semi")

    # Use the raw rating (post-min_rating filter, so 4 or 5) as the matrix value, not a binary 1.
    # implicit.als computes ``c_ui = 1 + alpha * matrix_value``, so passing the rating lets a
    # 5-star carry ~25% more confidence than a 4-star (e.g. c=201 vs c=161 at alpha=40) instead
    # of being indistinguishable. This is exactly the role of ``r_ui`` in the Hu et al. 2008
    # formulation — the original paper used TV watch-time; here we use review rating.
    train_ui = sp.csr_matrix(
        (
            train[_RATING].to_numpy().astype(np.float32),
            (train[_USER_IDX].to_numpy(), train[_ITEM_IDX].to_numpy()),
        ),
        shape=(n_users, n_items),
    )
    return Dataset(
        train_ui=train_ui,
        val_users=val[_USER_IDX].to_numpy(),
        val_items=val[_ITEM_IDX].to_numpy(),
        test_users=test[_USER_IDX].to_numpy(),
        test_items=test[_ITEM_IDX].to_numpy(),
    )


def build_dataset(
    category: str = "Books",
    *,
    cache_dir: Path = RAW_CACHE_DIR,
    source: str | None = None,
    max_interactions: int | None = None,
    min_user: int = 5,
    min_item: int = 5,
    min_rating: float = 4.0,
) -> Dataset:
    """Load (caching once), filter, and split the reviews into a :class:`Dataset`.

    On first call the reviews JSONL is read and cached as ``{cache_dir}/{category}.parquet`` (just
    the four columns the pipeline needs); later calls reuse that Parquet cache and skip the multi-GB
    re-read. The cache is unfiltered, so changing ``min_rating``/``min_user``/``min_item`` reuses it.

    Args:
        category: Amazon category name (e.g. ``"Books"``).
        cache_dir: Directory holding the per-category Parquet review cache.
        source: Override with a local path to a reviews JSONL; defaults to downloading ``category``
            from the HF hub.
        max_interactions: Optional cap on the number of cached review rows scanned, for a tractable
            development matrix. Applied to the lazy scan *before* the deduplicating ``group_by``,
            so it genuinely bounds memory and is deterministic (a row prefix, not a post-aggregation
            sample). ``None`` (the default) processes every cached row.
        min_user: k-core threshold on interactions per user.
        min_item: k-core threshold on interactions per item.
        min_rating: Minimum star rating for an interaction to count as positive.

    Returns:
        Dataset: The preprocessed interaction matrix with leave-last-out splits.

    Raises:
        ValueError: If filtering removes all interactions.
    """
    cache_path = cache_dir / f"{category}.parquet"
    if cache_path.exists():
        print(f"Using cached reviews at {cache_path}")
    else:
        jsonl_path = source if source is not None else _download_reviews(category)
        print(f"No cache at {cache_path}; building it from {jsonl_path} (one-time download).")
        _build_review_cache(jsonl_path, cache_path)

    reviews = pl.scan_parquet(cache_path)
    if max_interactions is not None:
        reviews = reviews.head(max_interactions)
    interactions = _deduplicate_interactions(reviews, min_rating).collect(engine="streaming")
    print(f"  {interactions.height:,} unique interactions after rating filter and dedup")

    interactions = _k_core_filter(interactions, min_user, min_item)
    if interactions.height == 0:
        msg = "k-core filtering removed all interactions; relax min_user/min_item."
        raise ValueError(msg)
    print(f"  after {min_user}/{min_item}-core: {interactions.height:,} interactions")

    dataset = _assign_indices_and_split(interactions)
    print(
        f"  matrix: {dataset.n_users:,} users x {dataset.n_items:,} items, "
        f"{dataset.n_interactions:,} train interactions, density {dataset.density:.2e}",
    )
    return dataset
