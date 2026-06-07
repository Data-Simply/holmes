"""Random-search baseline: i.i.d. samples over the ALS hyperparameter hull."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np

from holmes.config import DEFAULT_SEED, MAX_ITERATIONS, RANDOM_SPACE, TOP_K, ALSParams
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, select_best

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Draw one sample uniformly on a log scale over ``[low, high]``."""
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _sample_params(rng: np.random.Generator) -> ALSParams:
    """Draw one :class:`ALSParams` from ``rng`` over :data:`holmes.config.RANDOM_SPACE`.

    The log/linear scale per hyperparameter matches the Bayesian sampler: ``factors``,
    ``regularization``, and ``alpha`` span orders of magnitude so they are sampled log-uniformly;
    ``iterations`` is a small linear integer count sampled uniformly over its inclusive range.
    """
    factors_lo, factors_hi = RANDOM_SPACE["factors"]
    reg_lo, reg_hi = RANDOM_SPACE["regularization"]
    iter_lo, iter_hi = RANDOM_SPACE["iterations"]
    alpha_lo, alpha_hi = RANDOM_SPACE["alpha"]
    return ALSParams(
        factors=round(_log_uniform(rng, factors_lo, factors_hi)),
        regularization=_log_uniform(rng, reg_lo, reg_hi),
        iterations=int(rng.integers(int(iter_lo), int(iter_hi) + 1)),
        alpha=_log_uniform(rng, alpha_lo, alpha_hi),
    )


def run_random(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    search_seed: int = 0,
    k: int = TOP_K,
    out_path: Path | None = None,
) -> SearchOutput:
    """Evaluate :data:`holmes.config.MAX_ITERATIONS` random configs and return the trial log and best.

    The trial count is the shared fixed budget (like grid is exhaustive), not a per-call knob, so
    the three-way comparison is always at the same budget.

    Args:
        dataset: Preprocessed interaction matrix.
        seed: Random seed fit per configuration.
        search_seed: Seed for the sampler drawing the configurations, controlling the search
            trajectory (distinct from the per-fit ``seed``).
        k: Ranking cut-off.
        out_path: Optional path to write the full results JSON.

    Returns:
        SearchOutput: ``trials`` (every evaluated config) and ``best`` (highest score).
    """
    rng = np.random.default_rng(search_seed)
    trials: list[EvalResult] = []
    for i in range(1, MAX_ITERATIONS + 1):
        params = _sample_params(rng)
        result = evaluate_config(params, dataset, seed=seed, k=k, split="val")
        trials.append(result)
        metrics = result["metrics"]
        timing = f"fit={metrics['fit_time_seconds']:.2f}s eval={metrics['eval_time_seconds']:.2f}s"
        print(f"[random {i}/{MAX_ITERATIONS}] {params.to_dict()} -> val ndcg={result['score']:.4f}  {timing}")

    best = select_best(trials)
    output: SearchOutput = {"strategy": "random", "n_trials": len(trials), "best": best, "trials": trials}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote {len(trials)} random trials to {out_path}")
    print(f"Best random config: {best['params']} (val ndcg={best['score']:.4f})")
    return output
