"""Shared evaluation harness: fit one configuration across seeds and aggregate diagnostics."""

from __future__ import annotations

import json
import math
import time
from typing import TYPE_CHECKING, NotRequired, TypedDict

from holmes.als.model import ALSRecommender
from holmes.config import DEFAULT_SEED, TOP_K
from holmes.metrics.diagnostics import compute_diagnostics

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.config import ALSParams
    from holmes.data.dataset import Dataset

PRIMARY_METRIC = "ndcg"
"""The goal metric every search strategy optimizes (held-out NDCG@K)."""


class EvalResult(TypedDict):
    """The result of evaluating one configuration with one seed.

    A single fit per call: stability across initializations is obtained by repeating the run with
    different seeds and aggregating externally, not stored here.

    Attributes:
        params: The evaluated hyperparameters as a plain dict.
        seed: The random seed fit.
        k: Ranking cut-off used.
        split: Held-out split scored (``"val"`` or ``"test"``).
        score: The primary metric (NDCG@K) — the search objective.
        metrics: The full diagnostic battery for this fit.
        trial_number: Optuna trial index (present only for Bayesian-search trials).
    """

    params: dict[str, float]
    seed: int
    k: int
    split: str
    score: float
    metrics: dict[str, float]
    trial_number: NotRequired[int]


class SearchOutput(TypedDict):
    """The full log a search strategy returns: every trial plus the best one."""

    strategy: str
    n_trials: int
    best: EvalResult
    trials: list[EvalResult]


def select_best(trials: list[EvalResult]) -> EvalResult:
    """Return the highest-scoring trial.

    Args:
        trials: Evaluated configurations.

    Returns:
        EvalResult: The highest-scoring trial.

    Raises:
        ValueError: If ``trials`` is empty (rather than surfacing an opaque ``max()`` error), or
            any score is non-finite — NaN comparisons are all False, so ``max`` over NaN keys
            silently returns an arbitrary trial instead of failing.
    """
    if len(trials) == 0:
        msg = "No trials were evaluated; cannot select a best configuration."
        raise ValueError(msg)
    n_non_finite = sum(1 for trial in trials if not math.isfinite(trial["score"]))
    if n_non_finite:
        msg = f"{n_non_finite} of {len(trials)} trials have non-finite scores; refusing to select a best."
        raise ValueError(msg)
    # `max` keeps the earliest trial on ties; both callers (grid's Cartesian order, bayes's seeded
    # sampler) pass trials in a deterministic order, so the selection is deterministic.
    return max(trials, key=lambda r: r["score"])


def log_trial(strategy: str, index: int, total: int, result: EvalResult) -> None:
    """Print one trial's progress line in the format shared by every search driver.

    Args:
        strategy: Strategy name for the log prefix (``grid``/``random``/``bayes``).
        index: 1-based trial counter.
        total: The strategy's total trial count (its share of the fixed budget).
        result: The just-evaluated trial.
    """
    metrics = result["metrics"]
    timing = f"fit={metrics['fit_time_seconds']:.2f}s eval={metrics['eval_time_seconds']:.2f}s"
    print(f"[{strategy} {index}/{total}] {result['params']} -> val ndcg={result['score']:.4f}  {timing}")


def write_search_output(strategy: str, trials: list[EvalResult], out_path: Path | None) -> SearchOutput:
    """Assemble a strategy's :class:`SearchOutput`, optionally persist it, and report the best.

    The single exit point for the grid/random/bayes drivers, so the results-file schema and the
    console summary cannot drift between strategies.

    Args:
        strategy: Strategy name recorded in the output.
        trials: Every evaluated trial, in evaluation order.
        out_path: Optional path for the results JSON.

    Returns:
        SearchOutput: ``trials`` plus the highest-scoring ``best``.
    """
    best = select_best(trials)
    output: SearchOutput = {"strategy": strategy, "n_trials": len(trials), "best": best, "trials": trials}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote {len(trials)} {strategy} trials to {out_path}")
    print(f"Best {strategy} config: {best['params']} (val ndcg={best['score']:.4f})")
    return output


def evaluate_config(
    params: ALSParams,
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    k: int = TOP_K,
    split: str = "val",
    show_progress: bool = False,
) -> EvalResult:
    """Fit ``params`` once and compute the diagnostic battery.

    Args:
        params: The hyperparameters to evaluate.
        dataset: Preprocessed interaction matrix with held-out positives.
        seed: Random seed for the fit (diagnostic sampling uses the fixed
            :data:`~holmes.config.EVAL_SAMPLE_SEED`).
        k: Ranking cut-off for all top-k metrics.
        split: Held-out split to score (``"val"`` during search, ``"test"`` for reporting).
        show_progress: Forwarded to :meth:`ALSRecommender.fit`.

    Returns:
        EvalResult: The params, the seed, the diagnostic battery, and the ``score`` (NDCG@K).
        Wall-clock timings (``fit_time_seconds`` and ``eval_time_seconds``) are included in
        ``metrics`` so the same field set is available everywhere a trial is recorded.
    """
    fit_start = time.perf_counter()
    model = ALSRecommender(params, seed=seed).fit(dataset.train_ui, show_progress=show_progress)
    fit_time_seconds = time.perf_counter() - fit_start

    eval_start = time.perf_counter()
    diagnostics = compute_diagnostics(model, dataset, k, split=split)
    eval_time_seconds = time.perf_counter() - eval_start

    metrics = {**diagnostics, "fit_time_seconds": fit_time_seconds, "eval_time_seconds": eval_time_seconds}
    return {
        "params": params.to_dict(),
        "seed": seed,
        "k": k,
        "split": split,
        "score": metrics[PRIMARY_METRIC],
        "metrics": metrics,
    }
