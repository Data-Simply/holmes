"""Tests for the CLI argument parser."""

import argparse
import functools
import json

import pytest

from holmes.cli import _build_iter_spec, _build_parser, _cmd_eval, _cmd_preprocess, _cmd_ranges
from holmes.config import HOLMES_SPACE, MAX_ITERATIONS
from holmes.data.preprocess import AMAZON_CATEGORIES


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


def test_budget_is_not_overridable_per_call():
    """CLAUDE.md: the fit budget is the single MAX_ITERATIONS with no per-call override — a
    --max-iterations flag would let HOLMES quietly run on a larger budget than grid/random/bayes.
    holmes-iter is the only per-call strategy, so it is the only place the flag could leak in."""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["holmes-iter", "--data", "x", "--max-iterations", "5"])


class TestCmdRanges:
    def test_prints_hp_bounds_budget_and_dataset_signal(self, books_dataset, tmp_path, capsys):
        """The agent calls this to discover the supported HP bounds, the iteration budget, and the
        dataset signal it reasons from to choose iteration 1 — so SKILL.md never hardcodes numbers
        that drift from the config, and the starting point is chosen from live data, not a formula."""
        data_dir = tmp_path / "data"
        books_dataset.save(data_dir)

        _cmd_ranges(argparse.Namespace(data=data_dir))

        printed = json.loads(capsys.readouterr().out)
        assert printed["max_iterations"] == MAX_ITERATIONS
        assert set(printed["ranges"].keys()) == set(HOLMES_SPACE.keys())
        for name, (low, high) in HOLMES_SPACE.items():
            assert printed["ranges"][name] == [low, high]
        assert printed["dataset"] == books_dataset.describe()


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
