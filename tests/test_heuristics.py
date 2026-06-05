"""Tests for the heuristic initial-params function."""

import numpy as np
import pytest
import scipy.sparse as sp

from holmes.data.dataset import Dataset
from holmes.search.heuristics import initial_hypothesis, initial_params


def _dataset_with_interactions(n_users: int, n_items: int, n_interactions: int) -> Dataset:
    """Build a Dataset whose training matrix has roughly ``n_interactions`` nonzeros."""
    rng = np.random.default_rng(0)
    rows = rng.integers(0, n_users, size=n_interactions)
    cols = rng.integers(0, n_items, size=n_interactions)
    train_ui = sp.csr_matrix((np.ones(n_interactions, dtype=np.float32), (rows, cols)), shape=(n_users, n_items))
    train_ui.sum_duplicates()
    empty = np.array([], dtype=int)
    return Dataset(train_ui, empty, empty, empty, empty)


class TestFactorsHeuristic:
    @pytest.mark.parametrize(
        ("n_users", "n_items", "n_interactions", "expected_factors"),
        [
            (50_000, 20_000, 1_500_000, 128),  # large -> more factors
            (5_000, 2_000, 50_000, 64),  # small -> the searchable floor
            (5_000, 2_000, 300_000, 64),  # moderate -> balanced
        ],
    )
    def test_factors_scale_with_interaction_count(self, n_users, n_items, n_interactions, expected_factors):
        dataset = _dataset_with_interactions(n_users, n_items, n_interactions)
        params, _ = initial_params(dataset)
        assert params.factors == expected_factors


class TestAlphaHeuristic:
    def test_sparser_matrix_gets_higher_alpha(self):
        sparse = _dataset_with_interactions(50_000, 50_000, 200_000)  # density well below 1e-3
        denser = _dataset_with_interactions(500, 500, 200_000)  # density well above 1e-3
        sparse_params, _ = initial_params(sparse)
        dense_params, _ = initial_params(denser)
        assert sparse_params.alpha > dense_params.alpha


def test_rationale_covers_every_hyperparameter():
    dataset = _dataset_with_interactions(5_000, 2_000, 300_000)
    _, rationale = initial_params(dataset)
    assert set(rationale.keys()) == {"factors", "regularization", "iterations", "alpha"}
    assert all(len(reason) > 0 for reason in rationale.values())


class TestInitialHypothesis:
    def test_returns_all_three_falsifiable_fields(self):
        dataset = _dataset_with_interactions(5_000, 2_000, 300_000)
        params, rationale = initial_params(dataset)
        hypothesis = initial_hypothesis(params, rationale)
        assert set(hypothesis.keys()) == {"mechanism", "outcome", "falsifiers"}
        assert all(len(value) > 0 for value in hypothesis.values())

    def test_mechanism_grounds_in_per_hyperparameter_rationales(self):
        """Mechanism should reference each HP's rationale so the iter-1 hypothesis is concrete."""
        dataset = _dataset_with_interactions(5_000, 2_000, 300_000)
        params, rationale = initial_params(dataset)
        mechanism = initial_hypothesis(params, rationale)["mechanism"]
        for hp_rationale in rationale.values():
            assert hp_rationale in mechanism

    def test_falsifiers_name_diagnostics_that_would_invalidate_each_rule(self):
        """Falsifiers must name the diagnostic metrics that would refute the heuristic, so the LLM
        knows what observation to check after the fit."""
        dataset = _dataset_with_interactions(5_000, 2_000, 300_000)
        params, rationale = initial_params(dataset)
        falsifiers = initial_hypothesis(params, rationale)["falsifiers"]
        for metric in ("train_recon_error", "train_test_ndcg_gap", "tail_recall"):
            assert metric in falsifiers
