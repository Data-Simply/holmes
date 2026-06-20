"""Command-line entry point: ``holmes preprocess|grid|random|bayes|rule|holmes-iter|heuristic|ranges|annotate|eval``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holmes.config import (
    DEFAULT_SEED,
    HOLMES_SPACE,
    MAX_ITERATIONS,
    PROCESSED_DIR,
    RAW_CACHE_DIR,
    RESULTS_DIR,
    TOP_K,
    ALSParams,
)
from holmes.data.dataset import Dataset
from holmes.data.preprocess import AMAZON_CATEGORIES, build_dataset
from holmes.search.bayes import run_bayes
from holmes.search.grid import run_grid
from holmes.search.harness import evaluate_config
from holmes.search.heuristics import initial_hypothesis, initial_params
from holmes.search.holmes import VALIDATION_STATUSES, annotate_iteration, run_iteration
from holmes.search.random_search import run_random
from holmes.search.rule_engine import run_rule_engine


def _add_common_data_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the shared, required ``--data`` argument to a subparser."""
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Directory holding the preprocessed dataset, e.g. data/processed/Books (required).",
    )


def _add_seed_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--seed`` argument (one fit per run; repeat runs for stability)."""
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for the fit.")


def _add_progress_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--progress`` flag so a long backgrounded fit emits a per-sweep ETA."""
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Stream a per-sweep tqdm progress bar from the ALS fit to stderr. "
        "Off by default (silent); tail the command's output file to watch sweeps tick.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with one subparser per strategy."""
    parser = argparse.ArgumentParser(prog="holmes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("preprocess", help="Build an Amazon Reviews interaction matrix for a category.")
    pre.add_argument(
        "--category",
        default="Books",
        help="Amazon category to download (e.g. Books, Electronics, Video_Games). See `holmes preprocess --all`.",
    )
    pre.add_argument(
        "--all",
        action="store_true",
        help="Preprocess every Amazon category, each into its own data/processed/<category> directory.",
    )
    pre.add_argument(
        "--cache-dir",
        type=Path,
        default=RAW_CACHE_DIR,
        help="Directory for the downloaded per-category Parquet review cache.",
    )
    pre.add_argument(
        "--max-interactions",
        type=int,
        default=None,
        help="Optional cap on cached rows scanned (development); default is no limit (full dataset).",
    )
    pre.add_argument("--min-user", type=int, default=5, help="k-core: minimum interactions per user.")
    pre.add_argument("--min-item", type=int, default=5, help="k-core: minimum interactions per item.")
    pre.add_argument("--min-rating", type=float, default=4.0, help="Minimum star rating counted as positive.")
    pre.add_argument(
        "--source",
        default=None,
        help="Override with a local reviews JSONL path (defaults to downloading from the HF hub).",
    )
    pre.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/<category>). Not allowed with --all.",
    )

    grid = sub.add_parser("grid", help="Run the grid-search baseline.")
    _add_common_data_arg(grid)
    _add_seed_arg(grid)
    grid.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    grid.add_argument("--out", type=Path, default=RESULTS_DIR / "grid.json", help="Results JSON path.")

    random = sub.add_parser("random", help="Run the random-search baseline.")
    _add_common_data_arg(random)
    _add_seed_arg(random)
    random.add_argument(
        "--search-seed",
        type=int,
        default=0,
        help="Seed for the sampler drawing configs (controls the search trajectory, distinct from the fit --seed).",
    )
    random.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    random.add_argument("--out", type=Path, default=RESULTS_DIR / "random.json", help="Results JSON path.")

    bayes = sub.add_parser("bayes", help="Run the Optuna Bayesian-optimization baseline.")
    _add_common_data_arg(bayes)
    _add_seed_arg(bayes)
    bayes.add_argument(
        "--sampler-seed",
        type=int,
        default=0,
        help="Seed for the TPE sampler (controls the search trajectory, distinct from the fit --seed).",
    )
    bayes.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    bayes.add_argument("--out", type=Path, default=RESULTS_DIR / "bayes.json", help="Results JSON path.")

    rule = sub.add_parser("rule", help="Run the deterministic rule-engine ablation (the guide's patterns as code).")
    _add_common_data_arg(rule)
    _add_seed_arg(rule)
    rule.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    rule.add_argument("--out", type=Path, default=RESULTS_DIR / "rule.json", help="Results JSON path.")

    holmes_iter = sub.add_parser("holmes-iter", help="Run ONE HOLMES iteration and append to the trajectory.")
    _add_common_data_arg(holmes_iter)
    holmes_iter.add_argument(
        "--trajectory",
        type=Path,
        default=RESULTS_DIR / "trajectory.json",
        help="Append-only trajectory log path.",
    )
    _add_seed_arg(holmes_iter)
    holmes_iter.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    holmes_iter.add_argument("--factors", type=int, default=None, help="Latent dimensionality.")
    holmes_iter.add_argument("--regularization", type=float, default=None, help="L2 penalty.")
    holmes_iter.add_argument("--iterations", type=int, default=None, help="ALS sweeps.")
    holmes_iter.add_argument("--alpha", type=float, default=None, help="Confidence scaling on positives.")
    holmes_iter.add_argument(
        "--mechanism",
        default=None,
        help="Hypothesis: which diagnostic metrics move, which direction, roughly how much.",
    )
    holmes_iter.add_argument(
        "--outcome",
        default=None,
        help="Hypothesis: which goal metric (ndcg) moves as a consequence, and why.",
    )
    holmes_iter.add_argument(
        "--falsifiers",
        default=None,
        help="Hypothesis: what observation would say the causal model is wrong.",
    )
    _add_progress_arg(holmes_iter)

    heuristic = sub.add_parser(
        "heuristic",
        help="Print heuristic initial params, or (with --trajectory) fit them and append iter 1.",
    )
    _add_common_data_arg(heuristic)
    heuristic.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="If set, fit the heuristic config with the derived hypothesis and append iter 1 here. "
        "Without this flag, the heuristic params and rationale are printed for inspection.",
    )
    _add_seed_arg(heuristic)
    heuristic.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off (used when --trajectory is set).")
    _add_progress_arg(heuristic)

    sub.add_parser(
        "ranges",
        help="Print the supported HOLMES hyperparameter ranges as JSON (so the agent can stay in bounds).",
    )

    annotate = sub.add_parser(
        "annotate",
        help="Fill validation_status and interpretation on a recorded trajectory entry.",
    )
    annotate.add_argument(
        "--trajectory",
        type=Path,
        default=RESULTS_DIR / "trajectory.json",
        help="Trajectory log path.",
    )
    annotate.add_argument("--iteration", type=int, required=True, help="1-based iteration to annotate.")
    annotate.add_argument(
        "--status",
        required=True,
        choices=list(VALIDATION_STATUSES),
        help="Validation status to record.",
    )
    annotate.add_argument(
        "--interpretation",
        required=True,
        help="Post-run interpretation text seeding the next hypothesis.",
    )

    evaluate = sub.add_parser("eval", help="Evaluate a single explicit configuration.")
    _add_common_data_arg(evaluate)
    evaluate.add_argument("--params", type=Path, required=True, help="JSON file or string of ALS params.")
    _add_seed_arg(evaluate)
    evaluate.add_argument("--k", type=int, default=TOP_K, help="Ranking cut-off.")
    evaluate.add_argument("--split", default="test", choices=["val", "test"], help="Held-out split to score.")
    _add_progress_arg(evaluate)

    return parser


def _preprocess_one(category: str, out_dir: Path, args: argparse.Namespace) -> None:
    """Build one category's dataset and save it to ``out_dir``."""
    dataset = build_dataset(
        category=category,
        cache_dir=args.cache_dir,
        source=args.source,
        max_interactions=args.max_interactions,
        min_user=args.min_user,
        min_item=args.min_item,
        min_rating=args.min_rating,
    )
    dataset.save(out_dir)
    print(f"Saved preprocessed {category} dataset to {out_dir}")


def _cmd_preprocess(args: argparse.Namespace) -> None:
    if not args.all:
        out_dir = args.out if args.out is not None else PROCESSED_DIR / args.category
        _preprocess_one(args.category, out_dir, args)
        return

    if args.source is not None:
        raise SystemExit("--source names one category's file and cannot be combined with --all.")
    if args.out is not None:
        raise SystemExit("--out cannot be combined with --all; each category writes to data/processed/<category>.")

    # Batch mode: isolate each build so one category's failure (a transient HF error, an emptied
    # k-core) neither aborts the run nor discards the categories already done. Summary exits non-zero.
    failures: list[str] = []
    for category in AMAZON_CATEGORIES:
        try:
            _preprocess_one(category, PROCESSED_DIR / category, args)
        except Exception as exc:  # noqa: BLE001 - one category's failure must not abort the batch
            print(f"FAILED {category}: {exc}")
            failures.append(category)

    print(f"Preprocessed {len(AMAZON_CATEGORIES) - len(failures)}/{len(AMAZON_CATEGORIES)} categories.")
    if failures:
        msg = f"{len(failures)} of {len(AMAZON_CATEGORIES)} categories failed: {', '.join(failures)}"
        raise SystemExit(msg)


def _cmd_grid(args: argparse.Namespace) -> None:
    run_grid(Dataset.load(args.data), seed=args.seed, k=args.k, out_path=args.out)


def _cmd_random(args: argparse.Namespace) -> None:
    run_random(
        Dataset.load(args.data),
        seed=args.seed,
        search_seed=args.search_seed,
        k=args.k,
        out_path=args.out,
    )


def _cmd_bayes(args: argparse.Namespace) -> None:
    run_bayes(
        Dataset.load(args.data),
        seed=args.seed,
        sampler_seed=args.sampler_seed,
        k=args.k,
        out_path=args.out,
    )


_HP_FLAGS = ("factors", "regularization", "iterations", "alpha")
_HYPOTHESIS_FLAGS = ("mechanism", "outcome", "falsifiers")


def _build_iter_spec(args: argparse.Namespace) -> dict:
    """Build the iteration spec from the four HP flags and three hypothesis flags.

    All seven flags are required — the hypothesis-before-results discipline is enforced by
    making the hypothesis part of the command that runs the fit.

    Args:
        args: Parsed arguments from the ``holmes-iter`` subcommand.

    Returns:
        dict: A spec with ``params`` and ``hypothesis`` keys ready for :func:`run_iteration`.

    Raises:
        SystemExit: If any of the seven required flags is missing.
    """
    hp_values = {name: getattr(args, name) for name in _HP_FLAGS}
    hyp_values = {name: getattr(args, name) for name in _HYPOTHESIS_FLAGS}
    missing = [f"--{name}" for name in _HP_FLAGS if hp_values[name] is None]
    missing += [f"--{name}" for name in _HYPOTHESIS_FLAGS if hyp_values[name] is None]
    if missing:
        msg = f"holmes-iter requires all seven flags. Missing: {', '.join(missing)}."
        raise SystemExit(msg)
    return {"params": hp_values, "hypothesis": hyp_values}


def _cmd_rule(args: argparse.Namespace) -> None:
    run_rule_engine(Dataset.load(args.data), seed=args.seed, k=args.k, out_path=args.out)


def _cmd_holmes_iter(args: argparse.Namespace) -> None:
    spec = _build_iter_spec(args)
    run_iteration(
        Dataset.load(args.data),
        spec,
        args.trajectory,
        seed=args.seed,
        k=args.k,
        show_progress=args.progress,
    )


def _cmd_heuristic(args: argparse.Namespace) -> None:
    dataset = Dataset.load(args.data)
    params, rationale = initial_params(dataset)
    if args.trajectory is not None:
        spec = {"params": params.to_dict(), "hypothesis": initial_hypothesis(params, rationale)}
        run_iteration(
            dataset,
            spec,
            args.trajectory,
            seed=args.seed,
            k=args.k,
            show_progress=args.progress,
        )
        return
    print(json.dumps({"params": params.to_dict(), "rationale": rationale}, indent=2))


def _cmd_ranges(_args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "max_iterations": MAX_ITERATIONS,
                "ranges": {name: list(bounds) for name, bounds in HOLMES_SPACE.items()},
            },
            indent=2,
        ),
    )


def _cmd_annotate(args: argparse.Namespace) -> None:
    entry = annotate_iteration(
        args.trajectory,
        iteration=args.iteration,
        validation_status=args.status,
        interpretation=args.interpretation,
    )
    print(json.dumps({"iteration": entry["iteration"], "validation_status": entry["validation_status"]}, indent=2))


def _cmd_eval(args: argparse.Namespace) -> None:
    try:
        is_file = args.params.exists()
    except OSError:
        is_file = False  # not a valid path (e.g. an inline JSON string too long to be a filename)
    raw = args.params.read_text() if is_file else str(args.params)
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"--params must be a JSON file or JSON string; could not parse {raw!r}: {exc}"
        raise SystemExit(msg) from exc
    try:
        params = ALSParams.from_dict(spec)
    except ValueError as exc:
        msg = f"--params is not a valid ALS hyperparameter mapping: {exc}"
        raise SystemExit(msg) from exc
    result = evaluate_config(
        params,
        Dataset.load(args.data),
        seed=args.seed,
        k=args.k,
        split=args.split,
        show_progress=args.progress,
    )
    print(json.dumps({"params": result["params"], "seed": result["seed"], "metrics": result["metrics"]}, indent=2))


_COMMANDS = {
    "preprocess": _cmd_preprocess,
    "grid": _cmd_grid,
    "random": _cmd_random,
    "bayes": _cmd_bayes,
    "rule": _cmd_rule,
    "holmes-iter": _cmd_holmes_iter,
    "heuristic": _cmd_heuristic,
    "ranges": _cmd_ranges,
    "annotate": _cmd_annotate,
    "eval": _cmd_eval,
}


def main() -> None:
    """Parse arguments and dispatch to the selected subcommand."""
    args = _build_parser().parse_args()
    _COMMANDS[args.command](args)
