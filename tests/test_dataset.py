"""Tests for the on-disk Dataset container."""

import json

import numpy as np
import pytest
import scipy.sparse as sp
from tests.conftest import BOOKS_PER_GROUP, INTERACTIONS_PER_CUSTOMER, N_CUSTOMERS, N_GROUPS

from holmes.data.dataset import Dataset

# Two interactions per user are held out (val + test), the rest are training.
_TRAIN_INTERACTIONS = N_CUSTOMERS * (INTERACTIONS_PER_CUSTOMER - 2)
_N_ITEMS = N_GROUPS * BOOKS_PER_GROUP
_EXPECTED_DENSITY = _TRAIN_INTERACTIONS / (N_CUSTOMERS * _N_ITEMS)


class TestDatasetProperties:
    def test_dimensions_match_matrix(self, books_dataset):
        assert books_dataset.n_users == N_CUSTOMERS
        assert books_dataset.n_items == _N_ITEMS

    def test_n_interactions_counts_training_rows(self, books_dataset):
        assert books_dataset.n_interactions == _TRAIN_INTERACTIONS

    def test_density_matches_interactions_over_cells(self, books_dataset):
        assert books_dataset.density == _EXPECTED_DENSITY

    def test_item_popularity_sums_to_interactions(self, books_dataset):
        assert int(books_dataset.item_popularity.sum()) == _TRAIN_INTERACTIONS
        assert books_dataset.item_popularity.shape == (_N_ITEMS,)


class TestDatasetRoundTrip:
    def test_save_load_preserves_matrix_and_splits(self, books_dataset, tmp_path):
        books_dataset.save(tmp_path)
        loaded = Dataset.load(tmp_path)

        # strict=True enforces dtype equality on top of the value check, catching a regression
        # that silently widens/narrows the matrix or the index arrays during round-trip.
        np.testing.assert_array_equal(loaded.train_ui.toarray(), books_dataset.train_ui.toarray(), strict=True)
        np.testing.assert_array_equal(loaded.test_users, books_dataset.test_users, strict=True)
        np.testing.assert_array_equal(loaded.test_items, books_dataset.test_items, strict=True)
        np.testing.assert_array_equal(loaded.val_users, books_dataset.val_users, strict=True)
        np.testing.assert_array_equal(loaded.item_popularity, books_dataset.item_popularity, strict=True)

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Dataset.load(tmp_path / "does_not_exist")

    def test_save_writes_metadata_file(self, books_dataset, tmp_path):
        books_dataset.save(tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["n_users"] == N_CUSTOMERS
        assert meta["n_items"] == _N_ITEMS
        assert meta["n_interactions"] == _TRAIN_INTERACTIONS


class TestItemSelfInformation:
    def test_matches_negative_log2_of_popularity_share(self, books_dataset):
        """Cached on Dataset like the other matrix-wide arrays (item_popularity,
        popularity_percentile), so diagnostics don't rebuild two O(n_items) arrays per call.
        Zero-popularity items are floored at a count of 1 so the log stays finite."""
        expected = -np.log2(np.maximum(books_dataset.item_popularity, 1) / books_dataset.n_users)
        np.testing.assert_allclose(books_dataset.item_self_information, expected)
        assert books_dataset.item_self_information.shape == (_N_ITEMS,)


class TestPopularityPercentile:
    def test_tie_aware_and_monotonic(self, books_dataset):
        """Equally-popular books share a percentile; the most popular gets the maximum."""
        pop = books_dataset.item_popularity
        pct = books_dataset.popularity_percentile
        # Every set of equally-popular items maps to a single shared percentile.
        for value in np.unique(pop):
            shared = pct[pop == value]
            assert np.allclose(shared, shared[0])
        # The most popular item attains the highest percentile.
        assert pct[np.argmax(pop)] == pct.max()


def test_empty_matrix_density_is_zero():
    empty = sp.csr_matrix((5, 5), dtype=np.float32)
    dataset = Dataset(
        train_ui=empty,
        val_users=np.array([], dtype=int),
        val_items=np.array([], dtype=int),
        test_users=np.array([], dtype=int),
        test_items=np.array([], dtype=int),
    )
    assert dataset.density == 0.0
    assert dataset.n_interactions == 0
    assert dataset.item_popularity.sum() == 0
