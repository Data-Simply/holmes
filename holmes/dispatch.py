"""Fan-out planner: split the baseline sweep into one runnable script per box.

The Makefile runs the whole ``grid``/``random``/``bayes`` sweep on a single machine, serially
(``.NOTPARALLEL`` -- each fit is a multi-GB ALS model). This planner takes the same sweep
dimensions and partitions its independent *jobs* -- one ``holmes <strategy>`` invocation per
``(category, fit-seed[, search-seed])`` -- across N boxes, emitting a shell script each box runs.

A job is the unit of fan-out, not an individual fit: each job is a full strategy run (its own
``MAX_ITERATIONS`` fits, done sequentially inside the CLI). Jobs are independent, so the only
parallelism is across boxes; within a box the script runs jobs one at a time, preserving the
single-model-in-memory guard the Makefile enforces with ``.NOTPARALLEL``.

Output paths and filenames mirror the Makefile exactly, so a box writing to a shared volume
interoperates with ``make`` skip/resume: a job whose result JSON already exists is skipped, both at
plan time (dropped from the partition) and at run time (guarded in the emitted script), making the
whole sweep idempotent and restartable.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

BASELINE_STRATEGIES = ("grid", "random", "bayes")
"""The unattended baselines this planner fans out. HOLMES is excluded -- it drives an LLM session
per job with a different cost profile and isolation needs, orchestrated by the Makefile target."""

# Repo-relative defaults (matching the Makefile's PROCESSED_DIR/RESULTS_DIR), NOT the absolute
# PROJECT_ROOT-anchored config constants: the job commands and remote paths must resolve both
# locally (run from the repo root) and on a box (cwd is /opt/holmes, a fresh clone). Absolute local
# paths would be shipped verbatim and not exist on the box.
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_PLAN_DIR = Path("plans")

# random and bayes sweep the fit-seed x search-seed cross product; grid is deterministic given the
# fit seed and takes no search seed. The flag name differs between the two search strategies.
_SEARCH_SEED_FLAG = {"random": "--search-seed", "bayes": "--sampler-seed"}


@dataclass(frozen=True)
class Job:
    """One independent unit of work: a single ``holmes <strategy>`` invocation.

    Attributes:
        strategy: The baseline strategy (``grid``/``random``/``bayes``).
        category: Preprocessed dataset name under ``processed_dir``.
        fit_seed: The ALS fit ``--seed`` (model init randomness).
        search_seed: The optimizer search-trajectory seed, or ``None`` for grid (which ignores it).
        out_path: Result JSON path; its existence marks the job complete.
        command: The full argv to execute, including the runner prefix.
    """

    strategy: str
    category: str
    fit_seed: int
    search_seed: int | None
    out_path: Path
    command: tuple[str, ...]


def _job_filename(strategy: str, fit_seed: int, search_seed: int | None) -> str:
    """Return the result filename for a job, matching the Makefile's naming.

    Args:
        strategy: The baseline strategy.
        fit_seed: The ALS fit seed.
        search_seed: The search-trajectory seed, or ``None`` for grid.

    Returns:
        str: ``grid-seed<N>.json`` for grid, else ``<strategy>-seed<N>-search<M>.json``.
    """
    if search_seed is None:
        return f"{strategy}-seed{fit_seed}.json"
    return f"{strategy}-seed{fit_seed}-search{search_seed}.json"


def _build_job(
    strategy: str,
    category: str,
    fit_seed: int,
    search_seed: int | None,
    *,
    processed_dir: Path,
    results_dir: Path,
    runner: tuple[str, ...],
) -> Job:
    """Construct one :class:`Job`, including its full command line.

    Args:
        strategy: The baseline strategy.
        category: Dataset name under ``processed_dir``.
        fit_seed: The ALS fit seed.
        search_seed: The search-trajectory seed, or ``None`` for grid.
        processed_dir: Parent directory of the per-category datasets.
        results_dir: Parent directory results are namespaced under (``<results_dir>/<category>``).
        runner: Command prefix invoking the CLI (e.g. ``("uv", "run", "holmes")``).

    Returns:
        Job: The fully specified work unit.
    """
    out_path = results_dir / category / _job_filename(strategy, fit_seed, search_seed)
    command = [*runner, strategy, "--data", str(processed_dir / category), "--seed", str(fit_seed)]
    if search_seed is not None:
        command += [_SEARCH_SEED_FLAG[strategy], str(search_seed)]
    command += ["--out", str(out_path)]
    return Job(strategy, category, fit_seed, search_seed, out_path, tuple(command))


def enumerate_jobs(
    strategies: list[str],
    categories: list[str],
    fit_seeds: list[int],
    search_seeds: list[int],
    *,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    runner: tuple[str, ...] = ("uv", "run", "holmes"),
) -> list[Job]:
    """Enumerate every sweep job for the requested strategies.

    grid yields one job per ``(category, fit_seed)``; random and bayes yield the full
    ``(category, fit_seed, search_seed)`` cross product -- the same dimensions the Makefile sweeps.

    Args:
        strategies: Baseline strategies to include (subset of :data:`BASELINE_STRATEGIES`).
        categories: Dataset names to run.
        fit_seeds: ALS fit seeds.
        search_seeds: Optimizer search-trajectory seeds (ignored by grid).
        processed_dir: Parent directory of the per-category datasets.
        results_dir: Parent directory results are namespaced under.
        runner: Command prefix invoking the CLI.

    Returns:
        list[Job]: Every job, ordered strategy-major then category, then seeds.
    """
    jobs: list[Job] = []
    for strategy in strategies:
        for category in categories:
            for fit_seed in fit_seeds:
                # grid takes no search seed, so it collapses to a single job per fit seed.
                seeds: list[int | None] = list(search_seeds) if strategy in _SEARCH_SEED_FLAG else [None]
                jobs.extend(
                    _build_job(
                        strategy,
                        category,
                        fit_seed,
                        search_seed,
                        processed_dir=processed_dir,
                        results_dir=results_dir,
                        runner=runner,
                    )
                    for search_seed in seeds
                )
    return jobs


def pending_jobs(jobs: list[Job]) -> list[Job]:
    """Drop jobs whose result already exists, so only outstanding work is dispatched.

    Mirrors the Makefile's skip-if-exists: a present result file is treated as complete. The emitted
    scripts re-check at run time too, so concurrent boxes and reruns never redo finished work.

    Args:
        jobs: Candidate jobs.

    Returns:
        list[Job]: The jobs whose ``out_path`` does not yet exist.
    """
    return [job for job in jobs if not job.out_path.exists()]


def partition(jobs: list[Job], n_boxes: int) -> list[list[Job]]:
    """Split jobs across ``n_boxes`` by round-robin, balancing count within one job.

    Round-robin (rather than contiguous slices) interleaves strategies and categories across boxes,
    so no single box inherits all of one expensive category. It balances *count*, not cost: jobs of
    a larger category (denser matrix) fit slower, so for very skewed catalogs sort or weight before
    partitioning. The independence of jobs makes any split correct; this one just keeps boxes even.

    Args:
        jobs: Jobs to distribute.
        n_boxes: Number of boxes (must be positive).

    Returns:
        list[list[Job]]: One job list per box; box counts differ by at most one.

    Raises:
        ValueError: If ``n_boxes`` is not positive.
    """
    if n_boxes < 1:
        msg = f"n_boxes must be >= 1, got {n_boxes}."
        raise ValueError(msg)
    boxes: list[list[Job]] = [[] for _ in range(n_boxes)]
    for i, job in enumerate(jobs):
        boxes[i % n_boxes].append(job)
    return boxes


def render_box_script(jobs: list[Job]) -> str:
    """Render the shell script a single box runs: its jobs, sequentially, skip-if-exists guarded.

    ``set -euo pipefail`` makes the box stop on the first failed fit rather than silently dropping a
    job. Jobs run one at a time (the multi-GB-model memory guard); the run-time existence guard
    keeps the script idempotent if re-run after a partial pass.

    Args:
        jobs: The jobs assigned to this box.

    Returns:
        str: A bash script body.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {len(jobs)} job(s)", ""]
    for job in jobs:
        out = shlex.quote(str(job.out_path))
        parent = shlex.quote(str(job.out_path.parent))
        tag = f"{job.strategy} {job.category} seed={job.fit_seed}"
        if job.search_seed is not None:
            tag += f" search={job.search_seed}"
        # Quote the tag: category comes from filesystem/CLI input, so a name with a shell metachar
        # ($, backtick, quote) must not expand or break the script under `set -euo pipefail`.
        skip_msg = shlex.quote(f"skip (exists): {tag}")
        start_msg = shlex.quote(f">>> {tag}")
        lines += [
            f"mkdir -p {parent}",
            f"if [ -f {out} ]; then echo {skip_msg}; else",
            f"  echo {start_msg}",
            f"  {shlex.join(job.command)}",
            "fi",
            "",
        ]
    return "\n".join(lines)


def discover_categories(processed_dir: Path) -> list[str]:
    """List preprocessed category names under ``processed_dir``, sorted for determinism.

    Args:
        processed_dir: Parent directory of the per-category datasets.

    Returns:
        list[str]: Sorted subdirectory names (each a ``--data`` target).
    """
    if not processed_dir.is_dir():
        return []
    return sorted(child.name for child in processed_dir.iterdir() if child.is_dir())


def add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the planning arguments shared by ``dispatch plan`` and ``dispatch up``.

    Args:
        parser: The subparser to populate.
    """
    parser.add_argument("--boxes", type=int, required=True, help="Number of boxes to fan out across.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(BASELINE_STRATEGIES),
        choices=BASELINE_STRATEGIES,
        help="Baselines to include (default: all three).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Datasets to run (default: every preprocessed category under --processed-dir).",
    )
    parser.add_argument("--fit-seeds", nargs="+", type=int, default=[0, 1, 2], help="ALS fit seeds.")
    parser.add_argument(
        "--search-seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Optimizer search-trajectory seeds (random/bayes; grid ignores them).",
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="Parent of the datasets.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Parent of the result JSON.")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR, help="Where per-box scripts are written.")
    parser.add_argument(
        "--runner",
        default="uv run holmes",
        help="Command prefix invoking the CLI on each box (default: 'uv run holmes').",
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include jobs whose result already exists (default: skip them).",
    )


def plan_boxes(args: argparse.Namespace) -> list[list[Job]]:
    """Resolve the sweep into a per-box partition of the jobs still needing a run.

    Args:
        args: Parsed planning arguments.

    Returns:
        list[list[Job]]: One job list per box (some may be empty if there is little work).

    Raises:
        SystemExit: If no categories are given and none are found under ``--processed-dir``.
    """
    categories = args.categories if args.categories is not None else discover_categories(args.processed_dir)
    if not categories:
        msg = f"No categories given and none found under {args.processed_dir}. Preprocess first."
        raise SystemExit(msg)
    jobs = enumerate_jobs(
        args.strategies,
        categories,
        args.fit_seeds,
        args.search_seeds,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        runner=tuple(shlex.split(args.runner)),
    )
    runnable = jobs if args.include_done else pending_jobs(jobs)
    return partition(runnable, args.boxes)


def write_box_scripts(boxes: list[list[Job]], plan_dir: Path) -> list[Path]:
    """Write one executable ``box-<i>.sh`` per box and return their paths.

    Args:
        boxes: The per-box partition from :func:`plan_boxes`.
        plan_dir: Directory the scripts are written to.

    Returns:
        list[Path]: The written script paths, one per box.
    """
    plan_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, box in enumerate(boxes):
        script_path = plan_dir / f"box-{i}.sh"
        script_path.write_text(render_box_script(box))
        script_path.chmod(0o755)
        paths.append(script_path)
    return paths


def run_plan(args: argparse.Namespace) -> None:
    """Plan the sweep, write one runnable script per box, and optionally run it locally.

    With ``--run`` the planned jobs are executed on this machine, one at a time (the multi-GB-model
    memory guard), instead of (only) writing scripts to ship elsewhere.

    Args:
        args: Parsed ``dispatch plan`` arguments.
    """
    boxes = plan_boxes(args)
    total = sum(len(box) for box in boxes)
    print(f"{total} pending job(s) across {args.boxes} box(es).")
    if total == 0:
        print("Nothing to dispatch; all results already exist.")
        return

    paths = write_box_scripts(boxes, args.plan_dir)
    for path, box in zip(paths, boxes, strict=True):
        print(f"  {path}: {len(box)} job(s)")
    print(f"Wrote {len(paths)} box script(s) to {args.plan_dir}/.")

    if getattr(args, "run", False):
        _run_locally(boxes)


def _run_locally(boxes: list[list[Job]]) -> None:
    """Execute every planned job on this machine, sequentially.

    Jobs run one at a time regardless of the box count: locally there is a single machine and a
    single multi-GB model fits at a time, so a >1 box plan just runs its jobs back to back.

    Args:
        boxes: The per-box partition; flattened in box-then-job order.

    Raises:
        SystemExit: If a job's command exits non-zero (surfacing the failing config).
    """
    jobs = [job for box in boxes for job in box]
    for i, job in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {job.strategy} {job.category} seed={job.fit_seed}")
        result = subprocess.run(job.command, check=False)  # noqa: S603 - command is built from our own config
        if result.returncode != 0:
            msg = f"Job failed ({' '.join(job.command)}); exit {result.returncode}."
            raise SystemExit(msg)
