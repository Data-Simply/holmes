"""Tests for the ALSRecommender wrapper."""

import numpy as np
import pytest
from tests.conftest import book_group, customer_group

from holmes.als.model import ALSRecommender
from holmes.config import ALSParams


class TestRecommendContract:
    def test_recommend_before_fit_raises(self, books_dataset):
        model = ALSRecommender(ALSParams(factors=16), seed=0)
        with pytest.raises(RuntimeError):
            model.recommend(np.array([0, 1]), books_dataset.train_ui, k=5)

    def test_recommend_returns_requested_shape(self, books_dataset):
        model = ALSRecommender(ALSParams(factors=16, iterations=10), seed=0).fit(books_dataset.train_ui)
        users = np.arange(10)
        recs = model.recommend(users, books_dataset.train_ui, k=7)
        assert recs.shape == (10, 7)

    def test_filter_seen_excludes_training_items(self, books_dataset):
        model = ALSRecommender(ALSParams(factors=16, iterations=10), seed=0).fit(books_dataset.train_ui)
        users = np.arange(20)
        recs = model.recommend(users, books_dataset.train_ui, k=10, filter_seen=True)
        for user, row in zip(users, recs, strict=True):
            seen = set(books_dataset.train_ui[user].indices.tolist())
            assert seen.isdisjoint(set(row.tolist()))


class TestFactors:
    def test_factor_shapes_match_params(self, books_dataset):
        params = ALSParams(factors=24, iterations=10)
        model = ALSRecommender(params, seed=0).fit(books_dataset.train_ui)
        assert model.user_factors.shape == (books_dataset.n_users, 24)
        assert model.item_factors.shape == (books_dataset.n_items, 24)


class TestLearnsStructure:
    def test_recommends_mostly_in_group_books(self, books_dataset):
        """A well-fit model recommends books from each customer's own taste group."""
        model = ALSRecommender(ALSParams(factors=32, regularization=0.01, iterations=20, alpha=40.0), seed=0)
        model.fit(books_dataset.train_ui)
        users = np.arange(books_dataset.n_users)
        recs = model.recommend(users, books_dataset.train_ui, k=10)
        in_group = [
            book_group(book) == customer_group(user) for user, row in zip(users, recs, strict=True) for book in row
        ]
        # With clean group structure the vast majority of recommendations should be in-group.
        assert np.mean(in_group) > 0.8

    def test_seed_is_reproducible(self, books_dataset):
        params = ALSParams(factors=16, iterations=10)
        first = ALSRecommender(params, seed=3).fit(books_dataset.train_ui).item_factors
        second = ALSRecommender(params, seed=3).fit(books_dataset.train_ui).item_factors
        np.testing.assert_allclose(first, second, atol=1e-5)
