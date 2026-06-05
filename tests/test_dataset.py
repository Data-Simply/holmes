"""Tests for the on-disk Dataset container."""

import json

import numpy as np
import pytest
import scipy.sparse as sp
from tests.conftest import BOOKS_PER_GROUP, N_CUSTOMERS, N_GROUPS

from holmes.data.dataset import Dataset


class TestDatasetProperties:
    def test_dimensions_match_matrix(self, books_dataset):
        assert books_dataset.n_users == N_CUSTOMERS
        assert books_dataset.n_items == N_GROUPS * BOOKS_PER_GROUP

    def test_n_interactions_equals_nnz(self, books_dataset):
        assert books_dataset.n_interactions == books_dataset.train_ui.nnz

    def test_density_is_interactions_over_cells(self, books_dataset):
        expected = books_dataset.n_interactions / (books_dataset.n_users * books_dataset.n_items)
        assert books_dataset.density == expected

    def test_item_popularity_sums_to_interactions(self, books_dataset):
        assert int(books_dataset.item_popularity.sum()) == books_dataset.n_interactions


class TestDatasetRoundTrip:
    def test_save_load_preserves_matrix_and_splits(self, books_dataset, tmp_path):
        books_dataset.save(tmp_path)
        loaded = Dataset.load(tmp_path)

        np.testing.assert_array_equal(loaded.train_ui.toarray(), books_dataset.train_ui.toarray())
        np.testing.assert_array_equal(loaded.test_users, books_dataset.test_users)
        np.testing.assert_array_equal(loaded.test_items, books_dataset.test_items)
        np.testing.assert_array_equal(loaded.val_users, books_dataset.val_users)
        np.testing.assert_array_equal(loaded.item_popularity, books_dataset.item_popularity)

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Dataset.load(tmp_path / "does_not_exist")

    def test_save_writes_metadata_file(self, books_dataset, tmp_path):
        books_dataset.save(tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text())
        assert meta["n_users"] == N_CUSTOMERS
        assert meta["n_interactions"] == books_dataset.n_interactions


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
