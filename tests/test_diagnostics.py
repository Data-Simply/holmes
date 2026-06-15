"""Tests for the diagnostic battery."""

import inspect

import numpy as np
import pytest

from holmes.als.model import ALSRecommender
from holmes.config import ALSParams
from holmes.data.dataset import Dataset
from holmes.metrics.diagnostics import compute_diagnostics

K = 10


@pytest.fixture
def fitted_diagnostics(books_dataset):
    model = ALSRecommender(ALSParams(factors=32, regularization=0.01, iterations=20, alpha=40.0), seed=0)
    model.fit(books_dataset.train_ui)
    return compute_diagnostics(model, books_dataset, k=K, split="test")


class TestEvalSamplingIsDecoupledFromTheFitSeed:
    def test_compute_diagnostics_takes_no_seed(self):
        """Evaluation sampling uses the fixed EVAL_SAMPLE_SEED, never the fit seed: re-running a
        config with another --seed must change only the model, not the evaluated population —
        otherwise eval-sampling noise is indistinguishable from model instability. Locked
        structurally: no seed can flow in at all."""
        assert "seed" not in inspect.signature(compute_diagnostics).parameters

    def test_sampled_eval_population_is_deterministic(self, books_dataset, monkeypatch):
        """With more held-out users than the cap, two calls must sample the identical users
        (identical metrics), so scores are comparable across runs."""
        monkeypatch.setattr("holmes.metrics.diagnostics.EVAL_SAMPLE_USERS", 50)
        model = ALSRecommender(ALSParams(factors=16, iterations=10), seed=0).fit(books_dataset.train_ui)
        first = compute_diagnostics(model, books_dataset, k=K, split="test")
        second = compute_diagnostics(model, books_dataset, k=K, split="test")
        assert first == second


def test_empty_held_out_split_raises(books_dataset):
    """An empty split must fail loudly here, not propagate NaN into every trial score (where
    select_best's max() would silently crown the first trial)."""
    empty = np.array([], dtype=int)
    dataset = Dataset(books_dataset.train_ui, empty, empty, empty, empty)
    model = ALSRecommender(ALSParams(factors=16, iterations=5), seed=0).fit(dataset.train_ui)
    with pytest.raises(ValueError, match="no held-out users"):
        compute_diagnostics(model, dataset, k=K, split="val")


class TestMetricRelationships:
    def test_gap_equals_train_minus_test_ndcg(self, fitted_diagnostics):
        expected = fitted_diagnostics["train_ndcg"] - fitted_diagnostics["ndcg"]
        assert fitted_diagnostics["train_test_ndcg_gap"] == pytest.approx(expected)

    def test_structured_data_metrics_are_reproducible(self, fitted_diagnostics):
        """The deterministic fixture (fixed data + ALS seed) yields stable, pinned metrics.

        Pinning the values (rather than a loose ``> 0.1`` bound) catches a regression that merely
        degrades quality — e.g. NDCG dropping from ~0.19 to ~0.05 — which a bound would miss.
        """
        assert fitted_diagnostics["catalog_coverage"] == pytest.approx(1.0)
        assert fitted_diagnostics["recall"] == pytest.approx(0.375, abs=0.03)
        assert fitted_diagnostics["ndcg"] == pytest.approx(0.19, abs=0.03)

    def test_invalid_split_raises(self, books_dataset):
        model = ALSRecommender(ALSParams(factors=16, iterations=5), seed=0).fit(books_dataset.train_ui)
        with pytest.raises(ValueError, match="split"):
            compute_diagnostics(model, books_dataset, k=K, split="train")


class TestOverfittingSignal:
    def test_low_regularization_widens_train_test_gap(self, books_dataset):
        """Less regularization should memorize training history more, widening the gap."""
        params_weak = ALSParams(factors=64, regularization=1e-4, iterations=20, alpha=40.0)
        params_strong = ALSParams(factors=64, regularization=5.0, iterations=20, alpha=40.0)
        weak = compute_diagnostics(ALSRecommender(params_weak, seed=0).fit(books_dataset.train_ui), books_dataset, k=K)
        strong = compute_diagnostics(
            ALSRecommender(params_strong, seed=0).fit(books_dataset.train_ui), books_dataset, k=K
        )
        assert weak["train_test_ndcg_gap"] > strong["train_test_ndcg_gap"]


class TestRegularizationSignal:
    def test_higher_regularization_shrinks_mean_factor_norm(self, books_dataset):
        """The gauge-invariant geometric-mean factor norm decreases as regularization rises.

        The per-side user/item norms move in opposite directions (a scaling-gauge artifact), so
        only their geometric mean is a meaningful shrinkage signal.
        """
        weak = compute_diagnostics(
            ALSRecommender(ALSParams(factors=32, regularization=0.001, iterations=15), seed=0).fit(
                books_dataset.train_ui
            ),
            books_dataset,
            k=K,
        )
        strong = compute_diagnostics(
            ALSRecommender(ALSParams(factors=32, regularization=30.0, iterations=15), seed=0).fit(
                books_dataset.train_ui
            ),
            books_dataset,
            k=K,
        )
        assert strong["mean_factor_norm"] < weak["mean_factor_norm"]
