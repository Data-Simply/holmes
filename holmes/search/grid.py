"""Grid-search baseline over the ALS hyperparameter grid."""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING

from holmes.config import DEFAULT_SEED, GRID_SPACE, TOP_K, ALSParams
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, select_best

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
        metrics = result["metrics"]
        timing = f"fit={metrics['fit_time_seconds']:.2f}s eval={metrics['eval_time_seconds']:.2f}s"
        print(f"[grid {i}/{len(configs)}] {params.to_dict()} -> val ndcg={result['score']:.4f}  {timing}")

    best = select_best(trials)
    output: SearchOutput = {"strategy": "grid", "n_trials": len(trials), "best": best, "trials": trials}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote {len(trials)} grid trials to {out_path}")
    print(f"Best grid config: {best['params']} (val ndcg={best['score']:.4f})")
    return output
