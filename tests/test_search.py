"""Tests for the grid, Bayesian, and HOLMES search drivers."""

import json
import math
from pathlib import Path

import pytest

from holmes.config import GRID_SPACE
from holmes.search import holmes as holmes_module
from holmes.search.bayes import run_bayes
from holmes.search.grid import _grid_configs, run_grid
from holmes.search.holmes import annotate_iteration, load_trajectory, run_iteration

SEED = 0


@pytest.fixture
def in_bounds_params() -> dict[str, float]:
    """GRID_SPACE's lower corner — a stable in-bounds params dict. Reading the grid keeps
    the values aligned with bounds changes instead of hardcoding constants that can drift."""
    return {name: values[0] for name, values in GRID_SPACE.items()}


@pytest.fixture
def trajectory_path(tmp_path) -> Path:
    """The append-only trajectory log location every HOLMES iteration writes to."""
    return tmp_path / "trajectory.json"


def _just_out_of_bounds(name: str, direction: str) -> float:
    """Return a value just outside GRID_SPACE's extreme for ``name``. Integer HPs step by one
    (the integer-coercion check fires before the bounds check on non-integral floats); float
    HPs step by a factor of ten."""
    low, high = GRID_SPACE[name][0], GRID_SPACE[name][-1]
    if isinstance(low, int):
        return low - 1 if direction == "low" else high + 1
    return low / 10 if direction == "low" else high * 10


class TestGrid:
    def test_grid_enumerates_full_cartesian_product(self):
        expected = math.prod(len(values) for values in GRID_SPACE.values())
        assert len(_grid_configs()) == expected

    def test_run_grid_writes_results_file(self, books_dataset, tmp_path):
        out = tmp_path / "grid.json"
        run_grid(books_dataset, seed=SEED, k=10, out_path=out)
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "grid"
        assert saved["n_trials"] == len(_grid_configs())
        assert len(saved["trials"]) == len(_grid_configs())


class TestBayes:
    def test_run_bayes_runs_requested_trials(self, books_dataset, tmp_path):
        n_trials = 4
        output = run_bayes(
            books_dataset,
            seed=SEED,
            n_trials=n_trials,
            k=10,
            sampler_seed=0,
            out_path=tmp_path / "bayes.json",
        )
        assert output["n_trials"] == n_trials
        assert len(output["trials"]) == n_trials

    def test_zero_trials_raises_descriptive_error(self, books_dataset):
        with pytest.raises(ValueError, match="No trials"):
            run_bayes(books_dataset, seed=SEED, n_trials=0, k=10, sampler_seed=0)


class TestHolmesIteration:
    def test_iteration_appends_entry_with_metrics_and_blank_interpretation(
        self,
        books_dataset,
        trajectory_path,
        in_bounds_params,
    ):
        hypothesis = {"mechanism": "more factors raise ndcg", "outcome": "ndcg up", "falsifiers": "ndcg flat"}
        spec = {"params": in_bounds_params, "hypothesis": hypothesis}

        entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)

        assert entry["iteration"] == 1
        assert entry["hypothesis"] == hypothesis
        assert "ndcg" in entry["metrics"]
        assert entry["validation_status"] is None
        assert entry["interpretation"] is None

    def test_iterations_accumulate_in_trajectory(self, books_dataset, trajectory_path, in_bounds_params):
        spec = {"params": in_bounds_params, "hypothesis": {}}
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)

        trajectory = load_trajectory(trajectory_path)
        assert [entry["iteration"] for entry in trajectory] == [1, 2]
        assert trajectory[1]["params"] == in_bounds_params

    def test_missing_params_raises(self, books_dataset, trajectory_path):
        with pytest.raises(KeyError, match="params"):
            run_iteration(books_dataset, {"hypothesis": {}}, trajectory_path, seed=SEED, k=10)

    @pytest.mark.parametrize("name", list(GRID_SPACE.keys()))
    @pytest.mark.parametrize("direction", ["low", "high"])
    def test_out_of_bounds_params_rejected(
        self,
        books_dataset,
        trajectory_path,
        in_bounds_params,
        name,
        direction,
    ):
        # In-bounds base so the override is the only violation — ALSParams' defaults
        # (regularization=0.01) aren't all inside HOLMES_SPACE.
        override = {name: _just_out_of_bounds(name, direction)}
        spec = {"params": {**in_bounds_params, **override}, "hypothesis": {}}
        with pytest.raises(ValueError, match=name):
            run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)

    def test_bounds_check_reads_holmes_space_not_bayes_space(
        self,
        books_dataset,
        trajectory_path,
        in_bounds_params,
        monkeypatch,
    ):
        # Tighten HOLMES_SPACE only (factors lower bound 128). If the loop reads HOLMES_SPACE,
        # the unpatched lower-corner factors must be rejected; if it accidentally reads
        # BAYES_SPACE (still derived from the unpatched GRID_SPACE), the value is in bounds.
        tightened = {
            "factors": (128, 1024),
            "regularization": (1e-3, 1.0),
            "iterations": (15, 30),
            "alpha": (10.0, 40.0),
        }
        monkeypatch.setattr(holmes_module, "HOLMES_SPACE", tightened)
        spec = {"params": in_bounds_params, "hypothesis": {}}
        with pytest.raises(ValueError, match="factors"):
            run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)


def test_load_trajectory_missing_file_returns_empty(tmp_path):
    assert load_trajectory(tmp_path / "absent.json") == []


def test_run_iteration_prints_entry_as_json(books_dataset, trajectory_path, in_bounds_params, capsys):
    """The appended entry is echoed to stdout as JSON so the agent reads it without a second command."""
    spec = {"params": in_bounds_params, "hypothesis": {}}
    entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
    printed = json.loads(capsys.readouterr().out)
    assert printed == entry


class TestMaxIterationsCap:
    def test_at_cap_refuses_to_run(self, books_dataset, trajectory_path, in_bounds_params):
        spec = {"params": in_bounds_params, "hypothesis": {}}
        # Fill the trajectory to the cap.
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)
        # Third call must refuse rather than burn another fit.
        with pytest.raises(RuntimeError, match="Budget exhausted"):
            run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=2)

    def test_below_cap_proceeds(self, books_dataset, trajectory_path, in_bounds_params):
        """Sanity: a high cap doesn't change behavior."""
        spec = {"params": in_bounds_params, "hypothesis": {}}
        entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10, max_iterations=10)
        assert entry["iteration"] == 1


@pytest.fixture
def seeded_trajectory(books_dataset, trajectory_path, in_bounds_params):
    """Two recorded iterations on the shared trajectory_path, ready for annotation tests."""
    spec = {"params": in_bounds_params, "hypothesis": {}}
    run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
    run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
    return trajectory_path


class TestAnnotateIteration:
    def test_sets_status_and_interpretation_on_named_iteration(self, seeded_trajectory):
        entry = annotate_iteration(
            seeded_trajectory,
            iteration=2,
            validation_status="validated",
            interpretation="gap closed",
        )

        assert entry["validation_status"] == "validated"
        assert entry["interpretation"] == "gap closed"
        persisted = load_trajectory(seeded_trajectory)
        assert persisted[1]["validation_status"] == "validated"
        assert persisted[1]["interpretation"] == "gap closed"
        assert persisted[0]["validation_status"] is None  # other iteration untouched

    def test_invalid_status_rejected_with_allowed_set_in_message(self, seeded_trajectory):
        with pytest.raises(ValueError, match="validated"):
            annotate_iteration(seeded_trajectory, iteration=1, validation_status="winner", interpretation="x")

    def test_unknown_iteration_rejected(self, seeded_trajectory):
        with pytest.raises(ValueError, match="iteration 99"):
            annotate_iteration(seeded_trajectory, iteration=99, validation_status="null", interpretation="x")

    def test_missing_trajectory_file_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            annotate_iteration(
                tmp_path / "absent.json",
                iteration=1,
                validation_status="validated",
                interpretation="x",
            )
