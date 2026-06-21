"""Tests for the fan-out planner in :mod:`holmes.dispatch`."""

from __future__ import annotations

import shlex
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from holmes import cli
from holmes.dispatch import (
    Cell,
    discover_categories,
    enumerate_cells,
    partition,
    pending_cells,
    render_box_script,
)

if TYPE_CHECKING:
    from pathlib import Path

CATEGORIES = ["Books", "Electronics"]
FIT_SEEDS = [0, 1, 2]
SEARCH_SEEDS = [0, 1]


def _cells(strategies: list[str]) -> list[Cell]:
    return enumerate_cells(strategies, CATEGORIES, FIT_SEEDS, SEARCH_SEEDS)


def test_grid_cell_count_is_category_by_fit_seed() -> None:
    # grid ignores the search seed, so it collapses to one cell per (category, fit-seed).
    cells = _cells(["grid"])
    assert len(cells) == len(CATEGORIES) * len(FIT_SEEDS)
    assert all(cell.search_seed is None for cell in cells)


@pytest.mark.parametrize("strategy", ["random", "bayes"])
def test_search_strategy_cell_count_includes_search_seeds(strategy: str) -> None:
    cells = _cells([strategy])
    assert len(cells) == len(CATEGORIES) * len(FIT_SEEDS) * len(SEARCH_SEEDS)
    assert {cell.search_seed for cell in cells} == set(SEARCH_SEEDS)


def test_command_uses_the_strategy_specific_search_flag() -> None:
    # The two search strategies name the seed flag differently; grid has neither flag.
    grid = _cells(["grid"])[0]
    random = _cells(["random"])[0]
    bayes = _cells(["bayes"])[0]
    assert "--search-seed" not in grid.command
    assert "--sampler-seed" not in grid.command
    assert "--search-seed" in random.command
    assert "--sampler-seed" not in random.command
    assert "--sampler-seed" in bayes.command
    assert "--search-seed" not in bayes.command


def test_out_path_filenames_match_the_makefile_naming() -> None:
    grid = _build_one("grid", fit_seed=2, search_seed=None)
    random = _build_one("random", fit_seed=2, search_seed=1)
    bayes = _build_one("bayes", fit_seed=2, search_seed=1)
    assert grid.out_path.name == "grid-seed2.json"
    assert random.out_path.name == "random-seed2-search1.json"
    assert bayes.out_path.name == "bayes-seed2-search1.json"
    # Results are namespaced per category, so shards never collide on a shared volume.
    assert grid.out_path.parent.name == "Books"


def _build_one(strategy: str, *, fit_seed: int, search_seed: int | None) -> Cell:
    seeds = [search_seed] if search_seed is not None else SEARCH_SEEDS
    return next(
        cell
        for cell in enumerate_cells([strategy], ["Books"], [fit_seed], seeds)
        if cell.fit_seed == fit_seed and cell.search_seed == search_seed
    )


def test_pending_cells_drops_existing_results(tmp_path: Path) -> None:
    cells = enumerate_cells(["grid"], CATEGORIES, FIT_SEEDS, SEARCH_SEEDS, results_dir=tmp_path)
    done = cells[0]
    done.out_path.parent.mkdir(parents=True)
    done.out_path.write_text("{}")
    pending = pending_cells(cells)
    assert len(pending) == len(cells) - 1
    assert done not in pending


def test_partition_covers_every_cell_disjointly_and_evenly() -> None:
    cells = _cells(["grid", "random", "bayes"])
    n_boxes = 4
    boxes = partition(cells, n_boxes)
    assert len(boxes) == n_boxes
    # Every cell lands in exactly one box (a dead box must not silently drop work).
    flattened = [cell for box in boxes for cell in box]
    assert sorted(flattened, key=id) == sorted(cells, key=id)
    assert len(flattened) == len(cells)
    # Round-robin keeps box sizes within one of each other.
    sizes = [len(box) for box in boxes]
    assert max(sizes) - min(sizes) <= 1


def test_partition_rejects_non_positive_box_count() -> None:
    with pytest.raises(ValueError, match="n_boxes must be >= 1"):
        partition(_cells(["grid"]), 0)


def test_render_box_script_guards_and_runs_each_cell() -> None:
    cells = _cells(["random"])[:2]
    script = render_box_script(cells)
    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail")
    # Each cell's command and its skip-if-exists guard both appear.
    for cell in cells:
        assert shlex.join(cell.command) in script
        assert f"[ -f {cell.out_path}" in script
    assert script.count("mkdir -p") == len(cells)


def test_discover_categories_lists_sorted_subdirs(tmp_path: Path) -> None:
    for name in ["Electronics", "Books"]:
        (tmp_path / name).mkdir()
    (tmp_path / "note.txt").write_text("not a category")
    assert discover_categories(tmp_path) == ["Books", "Electronics"]


def test_discover_categories_missing_dir_is_empty(tmp_path: Path) -> None:
    assert discover_categories(tmp_path / "nope") == []


def test_dispatch_subcommand_wired_into_cli(tmp_path: Path) -> None:
    # Exercise the real CLI path so the nested `dispatch plan` subparser stays wired together.
    plan_dir = tmp_path / "plans"
    args = cli._build_parser().parse_args(
        [
            "dispatch",
            "plan",
            "--boxes",
            "2",
            "--strategies",
            "grid",
            "--categories",
            "Books",
            "Electronics",
            "--fit-seeds",
            "0",
            "1",
            "--results-dir",
            str(tmp_path / "results"),
            "--plan-dir",
            str(plan_dir),
        ],
    )
    cli._COMMANDS[args.command](args)
    assert (plan_dir / "box-0.sh").exists()
    assert (plan_dir / "box-1.sh").exists()


def test_dispatch_plan_run_executes_cells_locally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # --run shells out to each pending cell's command instead of only writing scripts.
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "holmes.dispatch.subprocess.run",
        lambda command, check: calls.append(command) or SimpleNamespace(returncode=0),
    )
    args = cli._build_parser().parse_args(
        [
            "dispatch",
            "plan",
            "--run",
            "--boxes",
            "1",
            "--strategies",
            "grid",
            "--categories",
            "Books",
            "--fit-seeds",
            "0",
            "1",
            "--runner",
            "holmes",
            "--results-dir",
            str(tmp_path / "results"),
            "--plan-dir",
            str(tmp_path / "plans"),
        ],
    )
    cli._COMMANDS[args.command](args)
    # One subprocess call per planned grid cell (2 fit seeds x 1 category).
    assert len(calls) == 2
    assert all(tuple(command[:2]) == ("holmes", "grid") for command in calls)
