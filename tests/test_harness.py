"""Tests for the shared evaluation harness."""

import json

import pytest

from holmes.config import ALSParams
from holmes.search.harness import PRIMARY_METRIC, evaluate_config, log_trial, select_best, write_search_output


def _trial(score: float, factors: int = 64) -> dict:
    """A minimal EvalResult-shaped trial for the output-writing tests."""
    return {
        "params": {"factors": factors},
        "seed": 0,
        "k": 10,
        "split": "val",
        "score": score,
        "metrics": {"ndcg": score, "fit_time_seconds": 1.0, "eval_time_seconds": 0.5},
    }


class TestWriteSearchOutput:
    """The single exit point for grid/random/bayes, so the results-file schema and console
    summary cannot drift between strategies (previously three hand-kept copies)."""

    def test_writes_results_file_and_returns_output(self, tmp_path, capsys):
        trials = [_trial(0.1, factors=64), _trial(0.3, factors=128)]
        out = tmp_path / "nested" / "grid.json"

        output = write_search_output("grid", trials, out)

        assert output == {"strategy": "grid", "n_trials": 2, "best": trials[1], "trials": trials}
        assert json.loads(out.read_text()) == output
        printed = capsys.readouterr().out
        assert "Wrote 2 grid trials" in printed
        assert "Best grid config" in printed

    def test_no_out_path_skips_the_write_but_still_reports_best(self, capsys):
        output = write_search_output("random", [_trial(0.2)], None)
        assert output["n_trials"] == 1
        assert "Best random config" in capsys.readouterr().out


def test_log_trial_prints_the_shared_progress_line(capsys):
    log_trial("grid", 3, 72, _trial(0.1234))
    printed = capsys.readouterr().out
    assert "[grid 3/72]" in printed
    assert "val ndcg=0.1234" in printed
    assert "fit=1.00s" in printed


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
