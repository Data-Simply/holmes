"""Tests for the grid, Bayesian, and HOLMES search drivers."""

import json
import math

import pytest

from holmes.config import GRID_SPACE
from holmes.search import holmes as holmes_module
from holmes.search.bayes import run_bayes
from holmes.search.grid import _grid_configs, run_grid
from holmes.search.holmes import annotate_iteration, load_trajectory, run_iteration

SEED = 0


class TestGrid:
    def test_grid_enumerates_full_cartesian_product(self):
        expected = math.prod(len(values) for values in GRID_SPACE.values())
        assert len(_grid_configs()) == expected

    def test_run_grid_best_has_highest_score(self, books_dataset, tmp_path):
        # Use a reduced grid via monkeypatch-free direct evaluation: run the real (small) grid.
        output = run_grid(books_dataset, seed=SEED, k=10, out_path=tmp_path / "grid.json")
        best_score = output["best"]["score"]
        assert best_score == max(trial["score"] for trial in output["trials"])

    def test_run_grid_writes_results_file(self, books_dataset, tmp_path):
        out = tmp_path / "grid.json"
        run_grid(books_dataset, seed=SEED, k=10, out_path=out)
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "grid"
        assert saved["n_trials"] == len(_grid_configs())


class TestBayes:
    def test_run_bayes_runs_requested_trials(self, books_dataset, tmp_path):
        output = run_bayes(books_dataset, seed=SEED, n_trials=4, k=10, sampler_seed=0, out_path=tmp_path / "bayes.json")
        assert output["n_trials"] == 4
        assert output["best"]["score"] == max(trial["score"] for trial in output["trials"])

    def test_zero_trials_raises_descriptive_error(self, books_dataset):
        with pytest.raises(ValueError, match="No trials"):
            run_bayes(books_dataset, seed=SEED, n_trials=0, k=10, sampler_seed=0)


class TestHolmesIteration:
    def test_iteration_appends_entry_with_metrics_and_blank_interpretation(self, books_dataset, tmp_path):
        hypothesis = {"mechanism": "more factors raise ndcg", "outcome": "ndcg up", "falsifiers": "ndcg flat"}
        spec = {"params": {"factors": 64, "iterations": 15}, "hypothesis": hypothesis}
        trajectory_path = tmp_path / "trajectory.json"

        entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)

        assert entry["iteration"] == 1
        assert entry["hypothesis"] == hypothesis
        assert "ndcg" in entry["metrics"]
        assert entry["validation_status"] is None
        assert entry["interpretation"] is None

    def test_iterations_accumulate_in_trajectory(self, books_dataset, tmp_path):
        trajectory_path = tmp_path / "trajectory.json"
        first = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        run_iteration(books_dataset, first, trajectory_path, seed=SEED, k=10)
        second = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        run_iteration(books_dataset, second, trajectory_path, seed=SEED, k=10)

        trajectory = load_trajectory(trajectory_path)
        assert [entry["iteration"] for entry in trajectory] == [1, 2]
        assert trajectory[1]["params"]["factors"] == 64

    def test_missing_params_raises(self, books_dataset, tmp_path):
        with pytest.raises(KeyError):
            run_iteration(books_dataset, {"hypothesis": {}}, tmp_path / "trajectory.json", seed=SEED, k=10)

    @pytest.mark.parametrize(
        ("params", "violated_name"),
        [
            ({"factors": 16}, "factors"),
            ({"factors": 2048}, "factors"),
            ({"regularization": 1e-5}, "regularization"),
            ({"regularization": 5.0}, "regularization"),
            ({"iterations": 5}, "iterations"),
            ({"iterations": 60}, "iterations"),
            ({"alpha": 0.1}, "alpha"),
            ({"alpha": 100.0}, "alpha"),
        ],
    )
    def test_out_of_bounds_params_rejected(self, books_dataset, tmp_path, params, violated_name):
        spec = {"params": params, "hypothesis": {}}
        with pytest.raises(ValueError, match=violated_name):
            run_iteration(books_dataset, spec, tmp_path / "trajectory.json", seed=SEED, k=10)

    def test_bounds_check_reads_holmes_space_not_bayes_space(self, books_dataset, tmp_path, monkeypatch):
        # Tighten HOLMES_SPACE only (factors lower bound 128) — if the loop reads HOLMES_SPACE,
        # factors=64 must be rejected; if it accidentally reads BAYES_SPACE (still derived from
        # the unpatched GRID_SPACE, lower=64), the value is in bounds and the rejection won't fire.
        tightened = {
            "factors": (128, 1024),
            "regularization": (1e-3, 1.0),
            "iterations": (15, 30),
            "alpha": (10.0, 40.0),
        }
        monkeypatch.setattr(holmes_module, "HOLMES_SPACE", tightened)
        spec = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        with pytest.raises(ValueError, match="factors"):
            run_iteration(books_dataset, spec, tmp_path / "trajectory.json", seed=SEED, k=10)


def test_load_trajectory_missing_file_returns_empty(tmp_path):
    assert load_trajectory(tmp_path / "absent.json") == []


def test_run_iteration_prints_entry_as_json(books_dataset, tmp_path, capsys):
    """The appended entry is echoed to stdout as JSON so the agent reads it without a second command."""
    spec = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
    entry = run_iteration(books_dataset, spec, tmp_path / "trajectory.json", seed=SEED, k=10)
    printed = json.loads(capsys.readouterr().out)
    assert printed == entry


class TestMaxIterationsCap:
    def test_at_cap_refuses_to_run(self, books_dataset, tmp_path):
        trajectory_path = tmp_path / "trajectory.json"
        spec = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        # Fill the trajectory to the cap.
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)
        # Third call must refuse rather than burn another fit.
        with pytest.raises(RuntimeError, match="Budget exhausted"):
            run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)

    def test_below_cap_proceeds(self, books_dataset, tmp_path):
        """Sanity: a high cap doesn't change behavior."""
        trajectory_path = tmp_path / "trajectory.json"
        spec = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=10)
        assert entry["iteration"] == 1


class TestAnnotateIteration:
    def _seed_two_iterations(self, books_dataset, trajectory_path):
        first = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        run_iteration(books_dataset, first, trajectory_path, seed=SEED, k=10)
        second = {"params": {"factors": 64, "iterations": 15}, "hypothesis": {}}
        run_iteration(books_dataset, second, trajectory_path, seed=SEED, k=10)

    def test_sets_status_and_interpretation_on_named_iteration(self, books_dataset, tmp_path):
        trajectory_path = tmp_path / "trajectory.json"
        self._seed_two_iterations(books_dataset, trajectory_path)

        entry = annotate_iteration(
            trajectory_path,
            iteration=2,
            validation_status="validated",
            interpretation="gap closed",
        )

        assert entry["validation_status"] == "validated"
        assert entry["interpretation"] == "gap closed"
        persisted = load_trajectory(trajectory_path)
        assert persisted[1]["validation_status"] == "validated"
        assert persisted[1]["interpretation"] == "gap closed"
        assert persisted[0]["validation_status"] is None  # other iteration untouched

    def test_invalid_status_rejected_with_allowed_set_in_message(self, books_dataset, tmp_path):
        trajectory_path = tmp_path / "trajectory.json"
        self._seed_two_iterations(books_dataset, trajectory_path)
        with pytest.raises(ValueError, match="validated"):
            annotate_iteration(trajectory_path, iteration=1, validation_status="winner", interpretation="x")

    def test_unknown_iteration_rejected(self, books_dataset, tmp_path):
        trajectory_path = tmp_path / "trajectory.json"
        self._seed_two_iterations(books_dataset, trajectory_path)
        with pytest.raises(ValueError, match="iteration 99"):
            annotate_iteration(trajectory_path, iteration=99, validation_status="null", interpretation="x")

    def test_missing_trajectory_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            annotate_iteration(
                tmp_path / "absent.json",
                iteration=1,
                validation_status="validated",
                interpretation="x",
            )
