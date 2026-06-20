"""Tests for the CLI argument parser."""

import argparse
import functools
import json

import pytest

from holmes.cli import (
    _build_iter_spec,
    _build_parser,
    _cmd_eval,
    _cmd_heuristic,
    _cmd_preprocess,
    _cmd_ranges,
    _cmd_rule,
)
from holmes.config import DEFAULT_SEED, HOLMES_SPACE, MAX_ITERATIONS, TOP_K
from holmes.data.preprocess import AMAZON_CATEGORIES
from holmes.search.heuristics import initial_hypothesis, initial_params


def _capture_rule(captured: dict, dataset, *, seed, k, out_path):
    """run_rule_engine stand-in that records the forwarded arguments without fitting any models."""
    captured.update(seed=seed, k=k, out_path=out_path)
    return {}


def _build_or_fail(*, category, fail, dataset, **_):
    """build_dataset stand-in that raises for the ``fail`` category and returns ``dataset`` otherwise."""
    if category == fail:
        msg = f"k-core emptied the {category} matrix"
        raise ValueError(msg)
    return dataset


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


@pytest.mark.parametrize("command", ["holmes-iter", "heuristic"])
def test_budget_is_not_overridable_per_call(command):
    """CLAUDE.md: the fit budget is the single MAX_ITERATIONS with no per-call override — a
    --max-iterations flag would let HOLMES quietly run on a larger budget than grid/random/bayes."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args([command, "--data", "x", "--max-iterations", "5"])


class TestCmdRule:
    def test_parser_defaults(self):
        args = _build_parser().parse_args(["rule", "--data", "d"])
        assert args.command == "rule"
        assert args.k == TOP_K
        assert args.seed == DEFAULT_SEED
        assert args.out.name == "rule.json"

    def test_dispatch_forwards_seed_k_and_out(self, books_dataset, tmp_path, monkeypatch):
        """_cmd_rule loads the dataset and forwards the parsed flags to run_rule_engine unchanged.
        The driver is stubbed so the dispatch is checked without fitting the full budget."""
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)
        out = tmp_path / "rule.json"
        captured: dict = {}
        monkeypatch.setattr("holmes.cli.run_rule_engine", functools.partial(_capture_rule, captured))
        args = _build_parser().parse_args(
            ["rule", "--data", str(data_dir), "--seed", "3", "--k", "5", "--out", str(out)],
        )
        _cmd_rule(args)
        assert captured == {"seed": 3, "k": 5, "out_path": out}


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
        args = _build_parser().parse_args(["eval", "--data", "does-not-exist", "--params", overlong_invalid_json])
        with pytest.raises(SystemExit, match="--params must be a JSON file or JSON string"):
            _cmd_eval(args)

    def test_wrong_shaped_params_json_reports_friendly_error(self, tmp_path):
        """Feeding eval a results-file-shaped JSON (nested 'params') must error, not silently
        evaluate the all-default config and report it as the final score."""
        spec = tmp_path / "grid.json"
        spec.write_text(json.dumps({"params": {"factors": 64}, "seed": 0, "score": 0.1}))
        args = _build_parser().parse_args(["eval", "--data", "does-not-exist", "--params", str(spec)])
        with pytest.raises(SystemExit, match="params"):
            _cmd_eval(args)


class TestCmdPreprocess:
    def test_default_out_namespaces_by_category(self, books_dataset, tmp_path, monkeypatch):
        """Without --out the dataset lands in data/processed/<category>, not a shared directory."""
        monkeypatch.setattr("holmes.cli.PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("holmes.cli.build_dataset", lambda **_: books_dataset)
        args = _build_parser().parse_args(["preprocess", "--category", "Electronics"])

        _cmd_preprocess(args)

        assert (tmp_path / "Electronics" / "meta.json").exists()

    def test_explicit_out_overrides_the_default(self, books_dataset, tmp_path, monkeypatch):
        monkeypatch.setattr("holmes.cli.build_dataset", lambda **_: books_dataset)
        out = tmp_path / "custom"
        args = _build_parser().parse_args(["preprocess", "--out", str(out)])

        _cmd_preprocess(args)

        assert (out / "meta.json").exists()

    def test_all_preprocesses_every_category_into_its_own_dir(self, books_dataset, tmp_path, monkeypatch):
        monkeypatch.setattr("holmes.cli.PROCESSED_DIR", tmp_path)
        monkeypatch.setattr("holmes.cli.build_dataset", lambda **_: books_dataset)
        args = _build_parser().parse_args(["preprocess", "--all"])

        _cmd_preprocess(args)

        # Every category produced its own dataset directory.
        produced = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())
        assert produced == sorted(AMAZON_CATEGORIES)
        assert (tmp_path / "Books" / "meta.json").exists()

    def test_all_continues_past_a_failing_category(self, books_dataset, tmp_path, monkeypatch):
        """One category raising must not abort the batch: the rest still build, then a summary exits non-zero."""
        monkeypatch.setattr("holmes.cli.PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(
            "holmes.cli.build_dataset",
            functools.partial(_build_or_fail, fail="Books", dataset=books_dataset),
        )
        args = _build_parser().parse_args(["preprocess", "--all"])

        with pytest.raises(SystemExit, match="1 of 34 categories failed: Books"):
            _cmd_preprocess(args)

        produced = {p.name for p in tmp_path.iterdir() if p.is_dir()}
        assert "Books" not in produced  # the failing category wrote nothing
        assert produced == set(AMAZON_CATEGORIES) - {"Books"}  # all 33 others still built

    def test_all_rejects_out(self):
        args = _build_parser().parse_args(["preprocess", "--all", "--out", "x"])
        with pytest.raises(SystemExit, match="--out cannot be combined with --all"):
            _cmd_preprocess(args)

    def test_all_rejects_source(self):
        args = _build_parser().parse_args(["preprocess", "--all", "--source", "reviews.jsonl"])
        with pytest.raises(SystemExit, match="--source"):
            _cmd_preprocess(args)


class TestCmdHeuristic:
    def test_trajectory_appends_iter_one_with_derived_hypothesis(self, books_dataset, tmp_path):
        """With --trajectory, heuristic fits and appends iter 1 directly — no iter.json hop.

        The recorded params and hypothesis are pinned to the deterministic heuristic output
        (derived from its source functions, not asserted as merely non-empty), so a regression
        that records a stub or padded hypothesis fails here.
        """
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)
        trajectory_path = tmp_path / "trajectory.json"
        args = _build_parser().parse_args(
            ["heuristic", "--data", str(data_dir), "--trajectory", str(trajectory_path)],
        )

        _cmd_heuristic(args)

        trajectory = json.loads(trajectory_path.read_text())
        assert len(trajectory) == 1
        entry = trajectory[0]
        expected_params, expected_rationale = initial_params(books_dataset)
        assert entry["iteration"] == 1
        assert entry["params"] == expected_params.to_dict()
        assert entry["hypothesis"] == initial_hypothesis(expected_params, expected_rationale)
        assert "ndcg" in entry["metrics"]

    def test_no_trajectory_prints_to_stdout(self, books_dataset, tmp_path, capsys):
        """Without --trajectory, the inspection form is preserved (params + rationale)."""
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)
        args = _build_parser().parse_args(["heuristic", "--data", str(data_dir)])

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
