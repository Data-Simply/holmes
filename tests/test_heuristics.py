"""Tests for the heuristic initial-params function."""

import math

import numpy as np
import pytest
import scipy.sparse as sp

from holmes.config import HOLMES_SPACE
from holmes.data.dataset import Dataset
from holmes.search.heuristics import initial_hypothesis, initial_params


def _dataset_with_nnz(nnz: int, rating: float = 1.0) -> Dataset:
    """A diagonal training matrix with exactly ``nnz`` interactions stored at ``rating``.

    The heuristic reads only the interaction count and the stored rating values, so a diagonal
    matrix exercises every rule without materializing a realistic interaction pattern.
    """
    data = np.full(nnz, rating, dtype=np.float32)
    indices = np.arange(nnz, dtype=np.int32)
    indptr = np.arange(nnz + 1, dtype=np.int32)
    train_ui = sp.csr_matrix((data, indices, indptr), shape=(nnz, nnz))
    empty = np.array([], dtype=int)
    return Dataset(train_ui, empty, empty, empty, empty)


class TestFactorsCapacityRule:
    def test_sits_at_the_searchable_floor_at_the_anchor_count(self):
        floor = HOLMES_SPACE["factors"][0]
        params, _ = initial_params(_dataset_with_nnz(100_000))
        assert params.factors == floor

    def test_doubles_per_decade_of_interactions(self):
        floor = HOLMES_SPACE["factors"][0]
        params, _ = initial_params(_dataset_with_nnz(1_000_000))
        assert params.factors == 2 * floor

    def test_small_datasets_clamp_to_the_floor(self):
        params, _ = initial_params(_dataset_with_nnz(1_000))
        assert params.factors == HOLMES_SPACE["factors"][0]

    def test_clamps_to_the_upper_bound(self, monkeypatch):
        tightened = {**HOLMES_SPACE, "factors": (64, 100)}
        monkeypatch.setattr("holmes.search.heuristics.HOLMES_SPACE", tightened)
        params, _ = initial_params(_dataset_with_nnz(1_000_000))  # raw rule says 128
        assert params.factors == 100


class TestAlphaConfidenceRule:
    @pytest.mark.parametrize(
        ("rating", "expected_alpha"),
        [
            (4.0, 24.75),  # c = 1 + 24.75 * 4.0 = 100, the Hu et al. operating point
            (5.0, 19.8),  # c = 1 + 19.8 * 5.0 = 100
        ],
    )
    def test_targets_the_confidence_operating_point(self, rating, expected_alpha):
        params, _ = initial_params(_dataset_with_nnz(100_000, rating=rating))
        assert params.alpha == pytest.approx(expected_alpha)

    def test_clamps_to_the_search_bounds(self):
        # A unit-rating matrix implies alpha = 99, far above the hull ceiling.
        params, _ = initial_params(_dataset_with_nnz(100_000, rating=1.0))
        assert params.alpha == HOLMES_SPACE["alpha"][1]


class TestMidpointRules:
    def test_regularization_has_equal_log_headroom_in_both_directions(self):
        """The starting L2 must leave the same multiplicative room to move up or down, so the
        loop's bold 10x moves are possible in either direction from iteration 2."""
        low, high = HOLMES_SPACE["regularization"]
        params, _ = initial_params(_dataset_with_nnz(100_000))
        assert math.isclose(params.regularization / low, high / params.regularization)

    def test_iterations_is_the_midpoint_of_the_sweep_range(self):
        low, high = HOLMES_SPACE["iterations"]
        params, _ = initial_params(_dataset_with_nnz(100_000))
        assert abs((high - params.iterations) - (params.iterations - low)) <= 1  # integer rounding


@pytest.mark.parametrize("nnz", [1_000, 100_000, 1_000_000])
def test_heuristic_params_fall_within_holmes_space(nnz):
    """Iter-1 starts from the heuristic, so its params must be inside the search space the rest of
    the loop optimizes over — otherwise the heuristic seeds the trajectory at a point no subsequent
    iteration is allowed to revisit."""
    params, _ = initial_params(_dataset_with_nnz(nnz, rating=4.5))
    values = params.to_dict()
    for name, (low, high) in HOLMES_SPACE.items():
        assert low <= values[name] <= high, f"heuristic {name}={values[name]} is outside HOLMES_SPACE [{low}, {high}]"


def test_heuristic_stays_in_bounds_when_the_space_is_retuned(monkeypatch):
    """In-bounds must hold by construction (clamping against the live HOLMES_SPACE), not because
    today's literals happen to fall inside today's grid — a GRID_SPACE retune must not break the
    HOLMES loop's mandated entry point."""
    tightened = {
        "factors": (128, 256),
        "regularization": (0.5, 1.0),
        "iterations": (25, 30),
        "alpha": (5.0, 10.0),
    }
    monkeypatch.setattr("holmes.search.heuristics.HOLMES_SPACE", tightened)
    params, _ = initial_params(_dataset_with_nnz(50_000, rating=4.0))
    values = params.to_dict()
    for name, (low, high) in tightened.items():
        assert low <= values[name] <= high, f"heuristic {name}={values[name]} escaped tightened [{low}, {high}]"


def test_rationale_grounds_each_hyperparameter_in_its_signal():
    """Each rationale should reference the signal it depends on (interaction count, mean rating,
    the searchable range) so the LLM can read the trajectory and see WHY the heuristic picked
    that value, not just that it did."""
    dataset = _dataset_with_nnz(300_000, rating=4.0)
    params, rationale = initial_params(dataset)
    assert set(rationale.keys()) == {"factors", "regularization", "iterations", "alpha"}
    assert f"{dataset.n_interactions:,}" in rationale["factors"]
    assert "4.00" in rationale["alpha"]  # the mean stored rating driving the confidence target
    assert str(params.regularization) in rationale["regularization"]


class TestInitialHypothesis:
    def test_mechanism_grounds_in_per_hyperparameter_rationales(self):
        """Mechanism should reference each HP's rationale so the iter-1 hypothesis is concrete."""
        dataset = _dataset_with_nnz(300_000, rating=4.0)
        params, rationale = initial_params(dataset)
        mechanism = initial_hypothesis(params, rationale)["mechanism"]
        for hp_rationale in rationale.values():
            assert hp_rationale in mechanism

    def test_falsifiers_name_diagnostics_that_would_invalidate_each_rule(self):
        """Falsifiers must name the diagnostic metrics that would refute the heuristic, so the LLM
        knows what observation to check after the fit."""
        dataset = _dataset_with_nnz(300_000, rating=4.0)
        params, rationale = initial_params(dataset)
        falsifiers = initial_hypothesis(params, rationale)["falsifiers"]
        for metric in ("train_recon_error", "train_test_ndcg_gap", "tail_recall"):
            assert metric in falsifiers
