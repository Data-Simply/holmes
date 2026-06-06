"""On-disk container for a preprocessed interaction matrix with leave-last-out splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import scipy.stats

from holmes.config import HEAD_ITEM_FRACTION

if TYPE_CHECKING:
    from pathlib import Path

_TRAIN_FILE = "train_ui.npz"
_SPLITS_FILE = "splits.npz"
_META_FILE = "meta.json"


@dataclass
class Dataset:
    """A user-item interaction matrix with held-out validation and test positives.

    The matrix uses leave-last-out splitting: per user, the most recent interaction is the
    test positive, the second most recent is the validation positive, and the remainder form
    the training matrix. Validation and test are stored as parallel arrays where entry ``i``
    means user ``val_users[i]`` has held-out item ``val_items[i]``.

    Attributes:
        train_ui: Rating-weighted CSR matrix of shape ``(n_users, n_items)`` of training
            interactions. Stored values are the raw star ratings (4 or 5 under the default
            ``min_rating``); ``implicit.als`` consumes them as ``r_ui`` in ``c_ui = 1 + alpha * r_ui``.
        val_users: User indices with a validation positive.
        val_items: Item indices held out for validation, aligned with ``val_users``.
        test_users: User indices with a test positive.
        test_items: Item indices held out for test, aligned with ``test_users``.
    """

    train_ui: sp.csr_matrix
    val_users: np.ndarray
    val_items: np.ndarray
    test_users: np.ndarray
    test_items: np.ndarray

    @cached_property
    def item_popularity(self) -> np.ndarray:
        """Per-item training interaction counts, shape ``(n_items,)``.

        Derived from ``train_ui`` (the per-column nonzero counts) rather than stored, so the
        training matrix remains the single source of truth. ``getnnz`` gives integer counts
        directly, avoiding a dense float64 column-sum and a dtype round-trip.
        """
        return self.train_ui.getnnz(axis=0).astype(np.int64)

    @cached_property
    def popularity_percentile(self) -> np.ndarray:
        """Per-item popularity percentile in ``[0, 1]`` (1 == most popular), shape ``(n_items,)``.

        Uses tie-aware average ranks so that equally-popular items receive the *same* percentile —
        important on the long tail where huge numbers of items share a popularity of 1 and would
        otherwise be spread across a wide percentile band purely by index order. Cached because it
        depends only on ``train_ui`` and is consulted by every diagnostics call.
        """
        ranks = scipy.stats.rankdata(self.item_popularity, method="average")
        return (ranks - 1) / max(len(ranks) - 1, 1)

    @cached_property
    def head_item_threshold(self) -> int:
        """Popularity count at or above which an item is in the popular 'head'.

        Items with ``item_popularity >= head_item_threshold`` form the top
        :data:`~holmes.config.HEAD_ITEM_FRACTION` of the catalog. Cached and expressed as a
        scalar cutoff so the tail test is an ``O(len(relevant))`` compare rather than a repeated
        full ``argsort`` plus ``isin`` against a large head-item array.

        Ties at the cutoff resolve *into* the head: on the long tail (huge numbers of items share a
        popularity of 1) the cutoff value is itself tied, so the head can be larger than
        ``HEAD_ITEM_FRACTION`` implies. The cutoff is constant for a given dataset, so ``tail_recall``
        stays comparable like-for-like across configs — it is a relative signal, not an exact 20% split.
        """
        n_items = len(self.item_popularity)
        n_head = max(int(n_items * HEAD_ITEM_FRACTION), 1)
        return int(np.partition(self.item_popularity, n_items - n_head)[n_items - n_head])

    @property
    def n_users(self) -> int:
        """Number of users in the training matrix."""
        return self.train_ui.shape[0]

    @property
    def n_items(self) -> int:
        """Number of items in the training matrix."""
        return self.train_ui.shape[1]

    @property
    def n_interactions(self) -> int:
        """Number of stored training interactions (nonzeros)."""
        return int(self.train_ui.nnz)

    @property
    def density(self) -> float:
        """Fraction of the user-item matrix that is observed."""
        return self.n_interactions / (self.n_users * self.n_items)

    def save(self, directory: Path) -> None:
        """Persist the dataset to ``directory`` as sparse matrices plus a metadata file.

        Args:
            directory: Destination directory; created if it does not exist.
        """
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / _TRAIN_FILE, self.train_ui)
        np.savez(
            directory / _SPLITS_FILE,
            val_users=self.val_users,
            val_items=self.val_items,
            test_users=self.test_users,
            test_items=self.test_items,
        )
        meta = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": self.n_interactions,
            "density": self.density,
        }
        (directory / _META_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        """Load a dataset previously written by :meth:`save`.

        Args:
            directory: Directory containing the saved dataset files.

        Returns:
            Dataset: The reconstructed dataset.

        Raises:
            FileNotFoundError: If the expected files are missing from ``directory``.
        """
        train_path = directory / _TRAIN_FILE
        if not train_path.exists():
            msg = f"No preprocessed dataset found at {directory} (missing {_TRAIN_FILE})."
            raise FileNotFoundError(msg)
        train_ui = sp.load_npz(train_path).tocsr()
        splits = np.load(directory / _SPLITS_FILE)
        return cls(
            train_ui=train_ui,
            val_users=splits["val_users"],
            val_items=splits["val_items"],
            test_users=splits["test_users"],
            test_items=splits["test_items"],
        )
