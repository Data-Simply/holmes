"""Tests for the CLI argument parser."""

import argparse
import json
from pathlib import Path

import pytest

from holmes.cli import _build_iter_spec, _build_parser, _cmd_eval, _cmd_heuristic, _cmd_ranges
from holmes.config import HOLMES_SPACE, MAX_ITERATIONS


def _iter_namespace(**overrides):
    """Return an argparse-style Namespace for holmes-iter with all CLI fields defaulted to None."""
    defaults = {
        "factors": None,
        "regularization": None,
        "iterations": None,
        "alpha": None,
        "mechanism": None,
        "outcome": None,
        "falsifiers": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


class TestParser:
    def test_grid_defaults_to_canonical_seed(self):
        args = _build_parser().parse_args(["grid", "--data", "x"])
        assert args.command == "grid"
        assert args.seed == 0
        assert isinstance(args.seed, int)

    def test_explicit_seed_is_parsed_as_int(self):
        args = _build_parser().parse_args(["eval", "--params", "{}", "--seed", "5"])
        assert args.seed == 5

    def test_bayes_has_distinct_fit_and_sampler_seeds(self):
        args = _build_parser().parse_args(["bayes", "--seed", "3", "--sampler-seed", "7"])
        assert args.seed == 3
        assert args.sampler_seed == 7

    def test_preprocess_max_interactions_defaults_to_no_limit(self):
        args = _build_parser().parse_args(["preprocess"])
        assert args.max_interactions is None

    def test_missing_subcommand_errors(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])

    def test_heuristic_trajectory_defaults_to_none(self):
        args = _build_parser().parse_args(["heuristic", "--data", "x"])
        assert args.trajectory is None

    def test_heuristic_trajectory_is_parsed_as_path(self):
        args = _build_parser().parse_args(
            ["heuristic", "--data", "x", "--trajectory", "results/trajectory.json", "--seed", "3"],
        )
        assert args.trajectory == Path("results/trajectory.json")
        assert args.seed == 3

    def test_holmes_iter_param_flags_parsed_with_correct_types(self):
        args = _build_parser().parse_args(
            [
                "holmes-iter",
                "--factors",
                "128",
                "--regularization",
                "0.05",
                "--iterations",
                "25",
                "--alpha",
                "12.5",
                "--mechanism",
                "m",
                "--outcome",
                "o",
                "--falsifiers",
                "f",
            ],
        )
        assert args.factors == 128
        assert args.regularization == 0.05
        assert args.iterations == 25
        assert args.alpha == 12.5

    def test_holmes_iter_hypothesis_flags_parsed(self):
        args = _build_parser().parse_args(
            [
                "holmes-iter",
                "--factors",
                "64",
                "--regularization",
                "0.01",
                "--iterations",
                "20",
                "--alpha",
                "40.0",
                "--mechanism",
                "raises regularization",
                "--outcome",
                "ndcg up",
                "--falsifiers",
                "no shift",
            ],
        )
        assert args.mechanism == "raises regularization"
        assert args.outcome == "ndcg up"
        assert args.falsifiers == "no shift"


class TestCmdRanges:
    def test_prints_hp_bounds_and_max_iterations(self, capsys):
        """The agent calls this to discover BOTH the supported HP bounds and the iteration budget,
        so SKILL.md never has to hardcode numbers that drift from the config."""
        _cmd_ranges(argparse.Namespace())

        printed = json.loads(capsys.readouterr().out)
        assert printed["max_iterations"] == MAX_ITERATIONS
        assert set(printed["ranges"].keys()) == set(HOLMES_SPACE.keys())
        for name, (low, high) in HOLMES_SPACE.items():
            assert printed["ranges"][name] == [low, high]


class TestCmdEval:
    def test_overlong_inline_json_reports_friendly_error(self):
        """An inline ``--params`` string longer than the OS filename limit must not crash.

        ``args.params`` is a ``Path``; calling ``.exists()`` on a >255-char string raises
        ``OSError: File name too long``. That must be caught so a malformed inline JSON string
        surfaces the friendly ``SystemExit`` message rather than a raw traceback.
        """
        overlong_invalid_json = "{not valid json" + "x" * 300
        args = argparse.Namespace(
            params=Path(overlong_invalid_json),
            data=Path("does-not-exist"),
            seed=0,
            k=10,
            split="test",
        )
        with pytest.raises(SystemExit, match="--params must be a JSON file or JSON string"):
            _cmd_eval(args)


class TestCmdHeuristic:
    def test_trajectory_appends_iter_one_with_derived_hypothesis(self, books_dataset, tmp_path):
        """With --trajectory, heuristic fits and appends iter 1 directly — no iter.json hop."""
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)
        trajectory_path = tmp_path / "trajectory.json"
        args = argparse.Namespace(data=data_dir, trajectory=trajectory_path, seed=0, k=10, max_iterations=10)

        _cmd_heuristic(args)

        trajectory = json.loads(trajectory_path.read_text())
        assert len(trajectory) == 1
        entry = trajectory[0]
        assert entry["iteration"] == 1
        assert set(entry["params"].keys()) == {"factors", "regularization", "iterations", "alpha"}
        assert set(entry["hypothesis"].keys()) == {"mechanism", "outcome", "falsifiers"}
        assert all(len(value) > 0 for value in entry["hypothesis"].values())
        assert "ndcg" in entry["metrics"]

    def test_no_trajectory_prints_to_stdout(self, books_dataset, tmp_path, capsys):
        """Without --trajectory, the inspection form is preserved (params + rationale)."""
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)
        args = argparse.Namespace(data=data_dir, trajectory=None, seed=0, k=10)

        _cmd_heuristic(args)

        printed = json.loads(capsys.readouterr().out)
        assert set(printed.keys()) == {"params", "rationale"}


class TestBuildIterSpec:
    def test_full_flag_set_builds_spec(self):
        spec = _build_iter_spec(
            _iter_namespace(
                factors=64,
                regularization=0.01,
                iterations=20,
                alpha=40.0,
                mechanism="m",
                outcome="o",
                falsifiers="f",
            )
        )
        assert spec == {
            "params": {"factors": 64, "regularization": 0.01, "iterations": 20, "alpha": 40.0},
            "hypothesis": {"mechanism": "m", "outcome": "o", "falsifiers": "f"},
        }

    def test_no_flags_lists_every_missing_flag(self):
        """All seven flags are required; the error must enumerate every missing one, not just one."""
        with pytest.raises(SystemExit) as excinfo:
            _build_iter_spec(_iter_namespace())
        message = str(excinfo.value)
        for flag in (
            "--factors",
            "--regularization",
            "--iterations",
            "--alpha",
            "--mechanism",
            "--outcome",
            "--falsifiers",
        ):
            assert flag in message

    @pytest.mark.parametrize(
        ("missing_flag", "expected_in_message"),
        [
            ("factors", "--factors"),
            ("regularization", "--regularization"),
            ("iterations", "--iterations"),
            ("alpha", "--alpha"),
            ("mechanism", "--mechanism"),
            ("outcome", "--outcome"),
            ("falsifiers", "--falsifiers"),
        ],
    )
    def test_partial_flag_set_lists_missing_flags(self, missing_flag, expected_in_message):
        full = {
            "factors": 64,
            "regularization": 0.01,
            "iterations": 20,
            "alpha": 40.0,
            "mechanism": "m",
            "outcome": "o",
            "falsifiers": "f",
        }
        flags = {name: (None if name == missing_flag else value) for name, value in full.items()}
        with pytest.raises(SystemExit, match=expected_in_message):
            _build_iter_spec(_iter_namespace(**flags))
