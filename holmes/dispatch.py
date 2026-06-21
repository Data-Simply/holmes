"""Fan-out planner: split the baseline sweep into one runnable script per box.

The Makefile runs the whole ``grid``/``random``/``bayes`` sweep on a single machine, serially
(``.NOTPARALLEL`` -- each fit is a multi-GB ALS model). This planner takes the same sweep
dimensions and partitions its independent *cells* -- one ``holmes <strategy>`` invocation per
``(category, fit-seed[, search-seed])`` -- across N boxes, emitting a shell script each box runs.

A cell is the unit of fan-out, not an individual fit: each cell is a full strategy run (its own
``MAX_ITERATIONS`` fits, done sequentially inside the CLI). Cells are independent, so the only
parallelism is across boxes; within a box the script runs cells one at a time, preserving the
single-model-in-memory guard the Makefile enforces with ``.NOTPARALLEL``.

Output paths and filenames mirror the Makefile exactly, so a box writing to a shared volume
interoperates with ``make`` skip/resume: a cell whose result JSON already exists is skipped, both at
plan time (dropped from the partition) and at run time (guarded in the emitted script), making the
whole sweep idempotent and restartable.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from holmes.config import PROCESSED_DIR, RESULTS_DIR

if TYPE_CHECKING:
    import argparse

BASELINE_STRATEGIES = ("grid", "random", "bayes")
"""The unattended baselines this planner fans out. HOLMES is excluded -- it drives an LLM session
per cell with a different cost profile and isolation needs, orchestrated by the Makefile target."""

# random and bayes sweep the fit-seed x search-seed cross product; grid is deterministic given the
# fit seed and takes no search seed. The flag name differs between the two search strategies.
_SEARCH_SEED_FLAG = {"random": "--search-seed", "bayes": "--sampler-seed"}


@dataclass(frozen=True)
class Cell:
    """One independent unit of work: a single ``holmes <strategy>`` invocation.

    Attributes:
        strategy: The baseline strategy (``grid``/``random``/``bayes``).
        category: Preprocessed dataset name under ``processed_dir``.
        fit_seed: The ALS fit ``--seed`` (model init randomness).
        search_seed: The optimizer search-trajectory seed, or ``None`` for grid (which ignores it).
        out_path: Result JSON path; its existence marks the cell complete.
        command: The full argv to execute, including the runner prefix.
    """

    strategy: str
    category: str
    fit_seed: int
    search_seed: int | None
    out_path: Path
    command: tuple[str, ...]


def _cell_filename(strategy: str, fit_seed: int, search_seed: int | None) -> str:
    """Return the result filename for a cell, matching the Makefile's naming.

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


def _build_cell(
    strategy: str,
    category: str,
    fit_seed: int,
    search_seed: int | None,
    *,
    processed_dir: Path,
    results_dir: Path,
    runner: tuple[str, ...],
) -> Cell:
    """Construct one :class:`Cell`, including its full command line.

    Args:
        strategy: The baseline strategy.
        category: Dataset name under ``processed_dir``.
        fit_seed: The ALS fit seed.
        search_seed: The search-trajectory seed, or ``None`` for grid.
        processed_dir: Parent directory of the per-category datasets.
        results_dir: Parent directory results are namespaced under (``<results_dir>/<category>``).
        runner: Command prefix invoking the CLI (e.g. ``("uv", "run", "holmes")``).

    Returns:
        Cell: The fully specified work unit.
    """
    out_path = results_dir / category / _cell_filename(strategy, fit_seed, search_seed)
    command = [*runner, strategy, "--data", str(processed_dir / category), "--seed", str(fit_seed)]
    if search_seed is not None:
        command += [_SEARCH_SEED_FLAG[strategy], str(search_seed)]
    command += ["--out", str(out_path)]
    return Cell(strategy, category, fit_seed, search_seed, out_path, tuple(command))


def enumerate_cells(
    strategies: list[str],
    categories: list[str],
    fit_seeds: list[int],
    search_seeds: list[int],
    *,
    processed_dir: Path = PROCESSED_DIR,
    results_dir: Path = RESULTS_DIR,
    runner: tuple[str, ...] = ("uv", "run", "holmes"),
) -> list[Cell]:
    """Enumerate every sweep cell for the requested strategies.

    grid yields one cell per ``(category, fit_seed)``; random and bayes yield the full
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
        list[Cell]: Every cell, ordered strategy-major then category, then seeds.
    """
    cells: list[Cell] = []
    for strategy in strategies:
        for category in categories:
            for fit_seed in fit_seeds:
                # grid takes no search seed, so it collapses to a single cell per fit seed.
                seeds: list[int | None] = list(search_seeds) if strategy in _SEARCH_SEED_FLAG else [None]
                cells.extend(
                    _build_cell(
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
    return cells


def pending_cells(cells: list[Cell]) -> list[Cell]:
    """Drop cells whose result already exists, so only outstanding work is dispatched.

    Mirrors the Makefile's skip-if-exists: a present result file is treated as complete. The emitted
    scripts re-check at run time too, so concurrent boxes and reruns never redo finished work.

    Args:
        cells: Candidate cells.

    Returns:
        list[Cell]: The cells whose ``out_path`` does not yet exist.
    """
    return [cell for cell in cells if not cell.out_path.exists()]


def partition(cells: list[Cell], n_boxes: int) -> list[list[Cell]]:
    """Split cells across ``n_boxes`` by round-robin, balancing count within one cell.

    Round-robin (rather than contiguous slices) interleaves strategies and categories across boxes,
    so no single box inherits all of one expensive category. It balances *count*, not cost: cells of
    a larger category (denser matrix) fit slower, so for very skewed catalogs sort or weight before
    partitioning. The independence of cells makes any split correct; this one just keeps boxes even.

    Args:
        cells: Cells to distribute.
        n_boxes: Number of boxes (must be positive).

    Returns:
        list[list[Cell]]: One cell list per box; box counts differ by at most one.

    Raises:
        ValueError: If ``n_boxes`` is not positive.
    """
    if n_boxes < 1:
        msg = f"n_boxes must be >= 1, got {n_boxes}."
        raise ValueError(msg)
    boxes: list[list[Cell]] = [[] for _ in range(n_boxes)]
    for i, cell in enumerate(cells):
        boxes[i % n_boxes].append(cell)
    return boxes


def render_box_script(cells: list[Cell]) -> str:
    """Render the shell script a single box runs: its cells, sequentially, skip-if-exists guarded.

    ``set -euo pipefail`` makes the box stop on the first failed fit rather than silently dropping a
    cell. Cells run one at a time (the multi-GB-model memory guard); the run-time existence guard
    keeps the script idempotent if re-run after a partial pass.

    Args:
        cells: The cells assigned to this box.

    Returns:
        str: A bash script body.
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"# {len(cells)} cell(s)", ""]
    for cell in cells:
        out = shlex.quote(str(cell.out_path))
        parent = shlex.quote(str(cell.out_path.parent))
        tag = f"{cell.strategy} {cell.category} seed={cell.fit_seed}"
        if cell.search_seed is not None:
            tag += f" search={cell.search_seed}"
        lines += [
            f"mkdir -p {parent}",
            f'if [ -f {out} ]; then echo "skip (exists): {tag}"; else',
            f"  echo '>>> {tag}'",
            f"  {shlex.join(cell.command)}",
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


def add_dispatch_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the dispatcher's arguments to ``parser`` (the ``holmes dispatch`` subparser).

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
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="Parent of the datasets.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Parent of the result JSON.")
    parser.add_argument("--plan-dir", type=Path, default=Path("plans"), help="Where per-box scripts are written.")
    parser.add_argument(
        "--runner",
        default="uv run holmes",
        help="Command prefix invoking the CLI on each box (default: 'uv run holmes').",
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Include cells whose result already exists (default: skip them).",
    )


def run_dispatch(args: argparse.Namespace) -> None:
    """Plan the sweep and write one runnable script per box.

    Args:
        args: Parsed arguments from the ``dispatch`` subcommand.
    """
    categories = args.categories if args.categories is not None else discover_categories(args.processed_dir)
    if not categories:
        msg = f"No categories given and none found under {args.processed_dir}. Preprocess first."
        raise SystemExit(msg)

    cells = enumerate_cells(
        args.strategies,
        categories,
        args.fit_seeds,
        args.search_seeds,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        runner=tuple(shlex.split(args.runner)),
    )
    runnable = cells if args.include_done else pending_cells(cells)
    print(f"{len(cells)} cells total, {len(runnable)} pending across {len(categories)} categories.")
    if not runnable:
        print("Nothing to dispatch; all results already exist.")
        return

    boxes = partition(runnable, args.boxes)
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    for i, box in enumerate(boxes):
        script_path = args.plan_dir / f"box-{i}.sh"
        script_path.write_text(render_box_script(box))
        script_path.chmod(0o755)
        print(f"  {script_path}: {len(box)} cell(s)")
    print(f"Wrote {len(boxes)} box script(s) to {args.plan_dir}/. Run box-<i>.sh on box i.")
