"""Shared evaluation harness: fit one configuration across seeds and aggregate diagnostics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NotRequired, TypedDict

from holmes.als.model import ALSRecommender
from holmes.config import DEFAULT_SEED, TOP_K
from holmes.metrics.diagnostics import compute_diagnostics

if TYPE_CHECKING:
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
        ValueError: If ``trials`` is empty, rather than surfacing an opaque ``max()`` error.
    """
    if len(trials) == 0:
        msg = "No trials were evaluated; cannot select a best configuration."
        raise ValueError(msg)
    # `max` keeps the earliest trial on ties; both callers (grid's Cartesian order, bayes's seeded
    # sampler) pass trials in a deterministic order, so the selection is deterministic.
    return max(trials, key=lambda r: r["score"])


def evaluate_config(
    params: ALSParams,
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    k: int = TOP_K,
    split: str = "val",
) -> EvalResult:
    """Fit ``params`` once and compute the diagnostic battery.

    Args:
        params: The hyperparameters to evaluate.
        dataset: Preprocessed interaction matrix with held-out positives.
        seed: Random seed for the fit and diagnostic sampling.
        k: Ranking cut-off for all top-k metrics.
        split: Held-out split to score (``"val"`` during search, ``"test"`` for reporting).

    Returns:
        EvalResult: The params, the seed, the diagnostic battery, and the ``score`` (NDCG@K).
        Wall-clock timings (``fit_time_seconds`` and ``eval_time_seconds``) are included in
        ``metrics`` so the same field set is available everywhere a trial is recorded.
    """
    fit_start = time.perf_counter()
    model = ALSRecommender(params, seed=seed).fit(dataset.train_ui)
    fit_time_seconds = time.perf_counter() - fit_start

    eval_start = time.perf_counter()
    diagnostics = compute_diagnostics(model, dataset, k, split=split, seed=seed)
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
