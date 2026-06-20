"""Tests for the grid, random, Bayesian, rule-engine, and HOLMES search drivers."""

import dataclasses
import functools
import inspect
import json
from pathlib import Path

import pytest

from holmes.config import (
    BAYES_SPACE,
    GRID_SPACE,
    HOLMES_SPACE,
    MAX_ITERATIONS,
    PARAM_SCALES,
    RANDOM_SPACE,
    RULE_SPACE,
    ALSParams,
    _grid_hull,
)
from holmes.search import holmes as holmes_module
from holmes.search import rule_engine
from holmes.search.bayes import run_bayes
from holmes.search.grid import _grid_configs, run_grid
from holmes.search.holmes import annotate_iteration, load_trajectory, run_iteration
from holmes.search.random_search import run_random
from holmes.search.rule_engine import next_params, run_rule_engine

SEED = 0


# The diagnostic keys the rule engine reads from a trajectory entry; the stub must supply them all
# or run_rule_engine (which feeds prior metrics into next_params) would KeyError. Derived nowhere
# authoritative, so kept explicit and matched to holmes/metrics/diagnostics.py.
_FULL_METRICS = {
    "ndcg": 0.5,
    "recall": 0.5,
    "map": 0.5,
    "train_ndcg": 0.5,
    "train_test_ndcg_gap": 0.5,
    "catalog_coverage": 0.5,
    "avg_rec_popularity": 0.5,
    "novelty": 0.5,
    "tail_recall": 0.5,
    "mean_factor_norm": 0.5,
    "train_recon_error": 0.5,
    "fit_time_seconds": 0.0,
    "eval_time_seconds": 0.0,
}


def _stub_eval(calls: list[str], params: ALSParams, dataset, *, seed=0, k=10, split=None, show_progress=False):
    """evaluate_config stand-in that records the requested split without fitting a model."""
    calls.append(split)
    return {
        "params": params.to_dict(),
        "seed": seed,
        "k": k,
        "split": split,
        "score": 0.5,
        "metrics": dict(_FULL_METRICS),
    }


# The single source of truth for which hyperparameters every strategy tunes; derived from ALSParams
# so adding a field can't silently leave a space behind.
_ALS_HYPERPARAMETERS = {field.name for field in dataclasses.fields(ALSParams)}


class TestComparabilityInvariants:
    """Lock the by-construction guarantees that make grid/random/bayes/HOLMES a fair comparison —
    same region, same budget, same hyperparameters. A future edit that breaks one fails here rather
    than silently skewing the benchmark."""

    def test_all_continuous_spaces_equal_the_grid_hull(self):
        # grid is discrete; random/bayes/HOLMES optimize its continuous hull. The three hulls are
        # deliberately distinct objects (so monkeypatching one doesn't bind the others) but MUST
        # stay equal by value, or the comparison measures search-space coverage, not optimizer skill.
        expected = _grid_hull(GRID_SPACE)
        assert RANDOM_SPACE == BAYES_SPACE == HOLMES_SPACE == RULE_SPACE == expected

    def test_every_space_covers_exactly_the_als_hyperparameters(self):
        for space in (GRID_SPACE, RANDOM_SPACE, BAYES_SPACE, HOLMES_SPACE, RULE_SPACE):
            assert set(space.keys()) == _ALS_HYPERPARAMETERS

    def test_grid_enumerates_exactly_the_shared_budget(self):
        # random/bayes read MAX_ITERATIONS directly and HOLMES caps at it; grid must enumerate the
        # same count or the fixed-budget comparison is off by construction.
        assert len(_grid_configs()) == MAX_ITERATIONS

    def test_spaces_are_distinct_objects(self):
        # The continuous spaces are deliberately distinct objects so an in-test monkeypatch on one
        # cannot silently rebind the others' search regions (see config.py and CLAUDE.md).
        spaces = (RANDOM_SPACE, BAYES_SPACE, HOLMES_SPACE, RULE_SPACE)
        assert len({id(space) for space in spaces}) == len(spaces)

    def test_rule_engine_default_start_lies_in_the_search_region(self):
        # The rule engine is the only strategy seeded from a fixed external config (the ALSParams
        # defaults) rather than sampled from / moved within the hull. Lock that this start is inside
        # RULE_SPACE so its first fit is in-region like every other strategy; derive both sides from
        # their sources so a GRID_SPACE retune that excludes the defaults fails loudly here.
        defaults = ALSParams().to_dict()
        for name, (low, high) in RULE_SPACE.items():
            assert low <= defaults[name] <= high

    def test_sampling_scales_cover_exactly_the_als_hyperparameters(self):
        # Random and bayes both read PARAM_SCALES, so they sample the same measure over the hull
        # by construction; this locks the table itself to the hyperparameter set.
        assert set(PARAM_SCALES) == _ALS_HYPERPARAMETERS
        assert set(PARAM_SCALES.values()) <= {"log", "linear"}

    def test_holmes_budget_is_not_overridable_per_call(self):
        # CLAUDE.md: the budget is the single MAX_ITERATIONS with no per-call override. HOLMES is
        # the only strategy whose driver runs one iteration at a time, so it is the only one where
        # an override parameter could creep in.
        assert "max_iterations" not in inspect.signature(run_iteration).parameters

    def test_every_strategy_scores_the_validation_split(
        self,
        books_dataset,
        trajectory_path,
        in_bounds_params,
        monkeypatch,
    ):
        """All five strategies must request the same held-out split from the shared harness —
        one strategy drifting to split="test" would optimize against the reporting split
        (test-set leakage) with nothing else failing. "val" is the convention being locked; it
        has no derivable source, which is exactly why CLAUDE.md requires a guardrail test."""
        calls: list[str] = []
        stub = functools.partial(_stub_eval, calls)
        for module in ("grid", "random_search", "bayes", "rule_engine", "holmes"):
            monkeypatch.setattr(f"holmes.search.{module}.evaluate_config", stub)
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 3)
        monkeypatch.setattr("holmes.search.bayes.MAX_ITERATIONS", 3)
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 3)

        run_grid(books_dataset, seed=SEED, k=10)
        run_random(books_dataset, seed=SEED, search_seed=0, k=10)
        run_bayes(books_dataset, seed=SEED, sampler_seed=0, k=10)
        run_rule_engine(books_dataset, seed=SEED, k=10)
        run_iteration(books_dataset, {"params": in_bounds_params}, trajectory_path, seed=SEED, k=10)

        assert len(calls) == len(_grid_configs()) + 3 + 3 + 3 + 1
        assert set(calls) == {"val"}


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
    def test_run_grid_writes_results_file(self, books_dataset, tmp_path):
        out = tmp_path / "grid.json"
        run_grid(books_dataset, seed=SEED, k=10, out_path=out)
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "grid"
        assert saved["n_trials"] == len(_grid_configs())
        assert len(saved["trials"]) == len(_grid_configs())


class TestRandom:
    def test_runs_the_fixed_max_iterations_budget(self, books_dataset, tmp_path, monkeypatch):
        # The budget is the shared MAX_ITERATIONS, not a per-call arg; shrink it so the test
        # fits a handful of models instead of the full budget.
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 4)
        out = tmp_path / "random.json"
        output = run_random(books_dataset, seed=SEED, search_seed=0, k=10, out_path=out)
        assert output["n_trials"] == 4
        assert len(output["trials"]) == 4
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "random"
        assert len(saved["trials"]) == 4

    def test_sampled_configs_stay_within_bounds(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 8)
        output = run_random(books_dataset, seed=SEED, search_seed=0, k=10)
        for trial in output["trials"]:
            for name, (low, high) in RANDOM_SPACE.items():
                assert low <= trial["params"][name] <= high

    def test_search_seed_makes_the_trajectory_reproducible(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 5)
        first = run_random(books_dataset, seed=SEED, search_seed=7, k=10)
        second = run_random(books_dataset, seed=SEED, search_seed=7, k=10)
        assert [t["params"] for t in first["trials"]] == [t["params"] for t in second["trials"]]

    def test_different_search_seeds_draw_different_configs(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 5)
        first = run_random(books_dataset, seed=SEED, search_seed=1, k=10)
        second = run_random(books_dataset, seed=SEED, search_seed=2, k=10)
        assert [t["params"] for t in first["trials"]] != [t["params"] for t in second["trials"]]

    def test_zero_budget_raises_descriptive_error(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.random_search.MAX_ITERATIONS", 0)
        with pytest.raises(ValueError, match="No trials"):
            run_random(books_dataset, seed=SEED, search_seed=0, k=10)


class TestBayes:
    def test_runs_the_fixed_max_iterations_budget(self, books_dataset, tmp_path, monkeypatch):
        monkeypatch.setattr("holmes.search.bayes.MAX_ITERATIONS", 4)
        out = tmp_path / "bayes.json"
        output = run_bayes(books_dataset, seed=SEED, k=10, sampler_seed=0, out_path=out)
        assert output["n_trials"] == 4
        assert len(output["trials"]) == 4
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "bayes"
        assert len(saved["trials"]) == 4

    def test_zero_budget_raises_descriptive_error(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.bayes.MAX_ITERATIONS", 0)
        with pytest.raises(ValueError, match="No trials"):
            run_bayes(books_dataset, seed=SEED, k=10, sampler_seed=0)


def _rule_entry(params: ALSParams, **metric_overrides: float) -> dict:
    """Build a trajectory entry for next_params: the params plus a full metric battery."""
    metrics = dict(_FULL_METRICS)
    metrics.update(metric_overrides)
    return {
        "params": params.to_dict(),
        "seed": SEED,
        "k": 10,
        "split": "val",
        "score": metrics["ndcg"],
        "metrics": metrics,
    }


class TestRuleEngine:
    def test_runs_the_fixed_max_iterations_budget(self, books_dataset, tmp_path, monkeypatch):
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 4)
        out = tmp_path / "rule.json"
        output = run_rule_engine(books_dataset, seed=SEED, k=10, out_path=out)
        assert output["n_trials"] == 4
        assert len(output["trials"]) == 4
        saved = json.loads(out.read_text())
        assert saved["strategy"] == "rule"
        assert len(saved["trials"]) == 4

    def test_starts_from_the_als_defaults_not_the_heuristic(self, books_dataset, monkeypatch):
        # Guide-only: the guide prescribes moves, not a starting config, so iteration 1 is the
        # dataset-independent ALSParams defaults rather than the heuristic. Locks that the ablation
        # shares no initialization rule with HOLMES.
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 1)
        output = run_rule_engine(books_dataset, seed=SEED, k=10)
        assert output["trials"][0]["params"] == ALSParams().to_dict()

    def test_proposed_configs_stay_within_bounds(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 8)
        output = run_rule_engine(books_dataset, seed=SEED, k=10)
        for trial in output["trials"]:
            for name, (low, high) in RULE_SPACE.items():
                assert low <= trial["params"][name] <= high

    def test_run_is_deterministic(self, books_dataset, monkeypatch):
        # No LLM and no sampler: identical inputs must yield an identical trajectory. That
        # determinism is the ablation's selling point (and CLAUDE.md's correctness property), so
        # lock it rather than leave it to convention.
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 6)
        first = run_rule_engine(books_dataset, seed=SEED, k=10)
        second = run_rule_engine(books_dataset, seed=SEED, k=10)
        assert [t["params"] for t in first["trials"]] == [t["params"] for t in second["trials"]]

    def test_zero_budget_raises_descriptive_error(self, books_dataset, monkeypatch):
        monkeypatch.setattr("holmes.search.rule_engine.MAX_ITERATIONS", 0)
        with pytest.raises(ValueError, match="No trials"):
            run_rule_engine(books_dataset, seed=SEED, k=10)

    def test_tail_starvation_lowers_alpha_and_raises_factors(self):
        # Pattern 7: acceptable ndcg, tail_recall far below overall recall, head-dominated recs ->
        # the prescribed two-HP move (alpha /4, factors *2). An absolute/cross-metric signal, so it
        # fires from a single entry without needing a trajectory history to calibrate against.
        params = ALSParams(factors=128, regularization=0.1, iterations=20, alpha=40.0)
        trajectory = [_rule_entry(params, recall=0.4, tail_recall=0.0, avg_rec_popularity=0.95)]
        proposed = next_params(trajectory)
        assert proposed.alpha == 10.0
        assert proposed.factors == 256

    def test_no_pattern_at_the_capacity_bound_re_fits_the_current_config(self):
        # Full-budget contract: when nothing fires and factors is already at the bound, the engine
        # returns the current config unchanged (a no-op re-fit) rather than stopping.
        _, fac_hi = RULE_SPACE["factors"]
        params = ALSParams(factors=int(fac_hi), regularization=0.1, iterations=20, alpha=10.0)
        assert next_params([_rule_entry(params)]) == params

    def test_move_rounds_a_fractional_integer_field(self):
        # Halving an odd `factors` yields a non-integral value that ALSParams.from_dict would
        # reject; _move must round integer fields so the move resolves deterministically. 127.5
        # rounds to the even 128 and stays in bounds.
        base = ALSParams(factors=255, regularization=0.1, iterations=20, alpha=10.0)
        assert rule_engine._move(base, factors=base.factors / 2).factors == 128


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

    def test_iteration_records_the_ranking_cutoff_and_split(self, books_dataset, trajectory_path, in_bounds_params):
        """k and split are persisted per entry (as grid/random/bayes trials already do), so a
        cut-off or split drift between iterations is visible in the trajectory rather than
        silently inflating scores."""
        spec = {"params": in_bounds_params, "hypothesis": {}}
        entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
        assert entry["k"] == 10
        assert entry["split"] == "val"

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


def test_load_trajectory_corrupted_file_raises_naming_the_path(tmp_path):
    """A truncated/corrupt trajectory must surface a descriptive error, not a bare JSONDecodeError
    traceback that leaves the agent guessing which file broke."""
    path = tmp_path / "trajectory.json"
    path.write_text('[{"iteration": 1, "par')  # a write cut short
    with pytest.raises(ValueError, match=r"trajectory\.json"):
        load_trajectory(path)


def test_trajectory_writes_are_atomic_and_leave_no_temp_file(books_dataset, trajectory_path, in_bounds_params):
    """Persistence goes through a sibling temp file + os.replace, so a kill mid-write can never
    truncate the trajectory; after a successful write the temp file must be gone."""
    spec = {"params": in_bounds_params, "hypothesis": {}}
    run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
    leftovers = [p for p in trajectory_path.parent.iterdir() if p.name != trajectory_path.name]
    assert leftovers == []
    assert len(load_trajectory(trajectory_path)) == 1


def test_run_iteration_prints_entry_as_json(books_dataset, trajectory_path, in_bounds_params, capsys):
    """The appended entry is echoed to stdout as JSON so the agent reads it without a second command."""
    spec = {"params": in_bounds_params, "hypothesis": {}}
    entry = run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
    printed = json.loads(capsys.readouterr().out)
    assert printed == entry


class TestMaxIterationsCap:
    def test_at_cap_refuses_to_run(self, books_dataset, trajectory_path, in_bounds_params, monkeypatch):
        # The budget is the shared module-level MAX_ITERATIONS (no per-call override); shrink it
        # the same way the random/bayes budget tests do.
        monkeypatch.setattr(holmes_module, "MAX_ITERATIONS", 2)
        spec = {"params": in_bounds_params, "hypothesis": {}}
        # Fill the trajectory to the cap.
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
        run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)
        # Third call must refuse rather than burn another fit.
        with pytest.raises(RuntimeError, match="Budget exhausted"):
            run_iteration(books_dataset, spec, trajectory_path, seed=SEED, k=10)


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
