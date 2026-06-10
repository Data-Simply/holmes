"""Grid-search baseline over the ALS hyperparameter grid."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from holmes.config import DEFAULT_SEED, GRID_SPACE, TOP_K, ALSParams
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, log_trial, write_search_output

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset


def _grid_configs() -> list[ALSParams]:
    """Enumerate every hyperparameter combination in :data:`holmes.config.GRID_SPACE`.

    Each Cartesian-product tuple is paired back with its parameter names so construction stays
    order-robust through ``ALSParams.from_dict`` (which is name-keyed).
    """
    names = list(GRID_SPACE.keys())
    return [
        ALSParams.from_dict(dict(zip(names, combo, strict=True))) for combo in itertools.product(*GRID_SPACE.values())
    ]


def run_grid(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    k: int = TOP_K,
    out_path: Path | None = None,
) -> SearchOutput:
    """Evaluate every grid configuration and return the trial log and best result.

    Args:
        dataset: Preprocessed interaction matrix.
        seed: Random seed fit per configuration.
        k: Ranking cut-off.
        out_path: Optional path to write the full results JSON.

    Returns:
        SearchOutput: ``trials`` (every evaluated config) and ``best`` (highest score).
    """
    configs = _grid_configs()
    trials: list[EvalResult] = []
    for i, params in enumerate(configs, start=1):
        result = evaluate_config(params, dataset, seed=seed, k=k, split="val")
        trials.append(result)
        log_trial("grid", i, len(configs), result)
    return write_search_output("grid", trials, out_path)
