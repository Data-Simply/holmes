"""Tests for the heuristic initial-params function."""

import numpy as np
import pytest
import scipy.sparse as sp

from holmes.config import HOLMES_SPACE
from holmes.data.dataset import Dataset
from holmes.search.heuristics import initial_hypothesis, initial_params

_ZIPF_EXPONENT = 1.05


def _power_law_indices(rng: np.random.Generator, n: int, max_index: int) -> np.ndarray:
    """Sample ``n`` indices in ``[0, max_index)`` from a Zipf-style power-law distribution.

    Real Books data has a heavy long tail: a few power readers/blockbusters carry most of the
    mass and most users/items have very few interactions. Uniform sampling would smear density
    evenly and give the heuristic a matrix that no realistic Books slice would produce. The
    exponent is mild (just above 1) so the head doesn't collapse the unique-nnz count too far
    below the sample count — heavier skew would make the heuristic's branch-threshold tests
    impossible to hit without dataset sizes far above what a unit-test fixture can build.
    """
    weights = 1.0 / np.power(np.arange(1, max_index + 1), _ZIPF_EXPONENT)
    weights /= weights.sum()
    return rng.choice(max_index, size=n, p=weights)


def _dataset_with_interactions(n_users: int, n_items: int, n_interactions: int) -> Dataset:
    """Build a Dataset whose training matrix has roughly ``n_interactions`` nonzeros, with
    user/item degree following a Books-realistic power law (heavy head, long tail)."""
    rng = np.random.default_rng(0)
    rows = _power_law_indices(rng, n_interactions, n_users)
    cols = _power_law_indices(rng, n_interactions, n_items)
    train_ui = sp.csr_matrix((np.ones(n_interactions, dtype=np.float32), (rows, cols)), shape=(n_users, n_items))
    train_ui.sum_duplicates()
    empty = np.array([], dtype=int)
    return Dataset(train_ui, empty, empty, empty, empty)


class TestFactorsHeuristic:
    @pytest.mark.parametrize(
        ("n_users", "n_items", "n_samples", "expected_factors"),
        [
            # n_samples > the heuristic's _LARGE_INTERACTIONS threshold AFTER power-law dedup
            # collapses ~half of the samples — needs to exceed ~2.1x the threshold raw.
            (50_000, 20_000, 3_000_000, 128),  # large -> more factors
            (5_000, 2_000, 50_000, 64),  # small -> the searchable floor
            (5_000, 2_000, 300_000, 64),  # moderate -> balanced
        ],
    )
    def test_factors_scale_with_interaction_count(self, n_users, n_items, n_samples, expected_factors):
        dataset = _dataset_with_interactions(n_users, n_items, n_samples)
        params, _ = initial_params(dataset)
        assert params.factors == expected_factors


class TestAlphaHeuristic:
    @pytest.mark.parametrize(
        ("n_users", "n_items", "n_samples", "expected_alpha"),
        [
            (50_000, 50_000, 200_000, 40.0),  # density well below 1e-3 → sparse branch
            (500, 500, 200_000, 15.0),  # density well above 1e-3 → dense branch
        ],
    )
    def test_alpha_pins_to_density_branch(self, n_users, n_items, n_samples, expected_alpha):
        dataset = _dataset_with_interactions(n_users, n_items, n_samples)
        params, _ = initial_params(dataset)
        assert params.alpha == expected_alpha


@pytest.mark.parametrize(
    ("n_users", "n_items", "n_samples"),
    [
        (50_000, 20_000, 3_000_000),  # large branch (post-dedup nnz > 1M)
        (5_000, 2_000, 50_000),  # small branch
        (5_000, 2_000, 300_000),  # moderate branch
        (50_000, 50_000, 200_000),  # sparse branch (alpha=40)
        (500, 500, 200_000),  # dense branch (alpha=15)
    ],
)
def test_heuristic_params_fall_within_holmes_space(n_users, n_items, n_samples):
    """Iter-1 starts from the heuristic, so its params must be inside the search space the rest of
    the loop optimizes over — otherwise the heuristic seeds the trajectory at a point no subsequent
    iteration is allowed to revisit, and (worse) the per-HP falsifier "this value is too low" can't
    be acted on because pushing it lower is out of bounds."""
    dataset = _dataset_with_interactions(n_users, n_items, n_samples)
    params, _ = initial_params(dataset)
    values = params.to_dict()
    for name, (low, high) in HOLMES_SPACE.items():
        assert low <= values[name] <= high, f"heuristic {name}={values[name]} is outside HOLMES_SPACE [{low}, {high}]"


def test_rationale_grounds_each_hyperparameter_in_dataset_signals():
    """Each rationale should reference the dataset signal it depends on (n_interactions, density)
    so the LLM can read the trajectory and see WHY the heuristic picked that value, not just that
    it did. Empty or generic strings would defeat the point."""
    n_users, n_items, n_interactions = 5_000, 2_000, 300_000
    dataset = _dataset_with_interactions(n_users, n_items, n_interactions)
    params, rationale = initial_params(dataset)
    assert set(rationale.keys()) == {"factors", "regularization", "iterations", "alpha"}
    # factors and alpha branch on dataset signals (n_interactions, density); the rationale must
    # surface those signals, not just state the picked value.
    assert f"{dataset.n_interactions:,}" in rationale["factors"]
    assert f"{dataset.density:.1e}" in rationale["alpha"]
    # regularization is anchored to GRID_SPACE's lower bound — the rationale must name the value.
    assert str(params.regularization) in rationale["regularization"]


class TestInitialHypothesis:
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
