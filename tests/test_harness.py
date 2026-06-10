"""Tests for the shared evaluation harness."""

import pytest

from holmes.config import ALSParams
from holmes.search.harness import PRIMARY_METRIC, evaluate_config, select_best


class TestSelectBest:
    def test_picks_highest_scoring_trial(self):
        # select_best keys only on "score"; minimal dicts (with a "tag" to identify the winner)
        # isolate that argmax logic. Full EvalResult-shaped trials are exercised by TestEvaluateConfig.
        trials = [{"score": 0.1, "tag": "a"}, {"score": 0.3, "tag": "b"}, {"score": 0.2, "tag": "c"}]
        assert select_best(trials)["tag"] == "b"

    def test_empty_trials_raises(self):
        with pytest.raises(ValueError, match="No trials"):
            select_best([])

    def test_non_finite_scores_are_rejected(self):
        """A NaN score must fail loudly: max() over NaN keys silently returns an arbitrary trial
        (every comparison is False), crowning a garbage winner with no error."""
        trials = [{"score": float("nan"), "tag": "a"}, {"score": 0.2, "tag": "b"}]
        with pytest.raises(ValueError, match="non-finite"):
            select_best(trials)


class TestEvaluateConfig:
    def test_score_equals_primary_metric_and_records_seed(self, books_dataset):
        result = evaluate_config(ALSParams(factors=16, iterations=10), books_dataset, seed=1, k=10)
        assert result["score"] == result["metrics"][PRIMARY_METRIC]
        assert result["seed"] == 1

    def test_same_seed_is_reproducible(self, books_dataset):
        params = ALSParams(factors=24, regularization=0.01, iterations=15, alpha=40.0)
        first = evaluate_config(params, books_dataset, seed=0, k=10)
        second = evaluate_config(params, books_dataset, seed=0, k=10)
        assert first["score"] == pytest.approx(second["score"], abs=1e-6)

    def test_records_fit_and_eval_timing_in_metrics(self, books_dataset):
        """Per-iteration wall time is part of the diagnostic battery so the trajectory log shows
        how long each ALS fit + diagnostic pass took. Useful for comparing grid vs. bayes vs.
        HOLMES on wall-time efficiency, not just final ndcg."""
        result = evaluate_config(ALSParams(factors=16, iterations=10), books_dataset, seed=0, k=10)
        assert result["metrics"]["fit_time_seconds"] > 0
        assert result["metrics"]["eval_time_seconds"] > 0

    def test_more_factors_improves_or_matches_structured_score(self, books_dataset):
        """On clean structure, a reasonable model should beat a degenerate one-factor model."""
        tiny = evaluate_config(ALSParams(factors=1, regularization=0.01, iterations=15), books_dataset, seed=0, k=10)
        rich = evaluate_config(ALSParams(factors=32, regularization=0.01, iterations=15), books_dataset, seed=0, k=10)
        assert rich["score"] > tiny["score"]

    def test_split_choice_is_recorded(self, books_dataset):
        result = evaluate_config(ALSParams(factors=16, iterations=10), books_dataset, seed=0, k=10, split="test")
        assert result["split"] == "test"
