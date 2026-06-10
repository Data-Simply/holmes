"""Bayesian-optimization baseline using Optuna's TPE sampler."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import optuna

from holmes.config import (
    BAYES_SPACE,
    DEFAULT_SEED,
    INTEGER_PARAMS,
    MAX_ITERATIONS,
    PARAM_SCALES,
    TOP_K,
    ALSParams,
)
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, log_trial, write_search_output

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset


def _suggest_params(trial: optuna.Trial) -> ALSParams:
    """Sample an :class:`ALSParams` from the Optuna trial over :data:`BAYES_SPACE`.

    The log/linear scale per hyperparameter comes from the shared
    :data:`holmes.config.PARAM_SCALES`, which the random sampler also reads — the two
    strategies sample the same measure over the hull by construction, not by convention.
    """
    values: dict[str, float] = {}
    for name, (low, high) in BAYES_SPACE.items():
        log = PARAM_SCALES[name] == "log"
        if name in INTEGER_PARAMS:
            values[name] = trial.suggest_int(name, int(low), int(high), log=log)
        else:
            values[name] = trial.suggest_float(name, low, high, log=log)
    return ALSParams.from_dict(values)


def _objective(
    trial: optuna.Trial,
    *,
    dataset: Dataset,
    seed: int,
    k: int,
    trials: list[EvalResult],
) -> float:
    """Evaluate one Optuna trial, record it, and return its validation score.

    Args:
        trial: The Optuna trial supplying the sampled hyperparameters.
        dataset: Preprocessed interaction matrix.
        seed: Random seed for the fit.
        k: Ranking cut-off.
        trials: Mutable accumulator the evaluated result is appended to.

    Returns:
        float: The trial's validation NDCG@k (the quantity Optuna maximizes).
    """
    params = _suggest_params(trial)
    result = evaluate_config(params, dataset, seed=seed, k=k, split="val")
    result["trial_number"] = trial.number
    trials.append(result)
    log_trial("bayes", trial.number + 1, MAX_ITERATIONS, result)
    return result["score"]


def run_bayes(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    sampler_seed: int = 0,
    k: int = TOP_K,
    out_path: Path | None = None,
) -> SearchOutput:
    """Run an Optuna study maximizing held-out NDCG@K and return the trial log.

    The count is the shared fixed budget :data:`holmes.config.MAX_ITERATIONS`.

    Args:
        dataset: Preprocessed interaction matrix.
        seed: Random seed for each trial's fit.
        sampler_seed: Seed for the TPE sampler, controlling the search trajectory (distinct from the
            per-fit ``seed``).
        k: Ranking cut-off.
        out_path: Optional path to write the full results JSON.

    Returns:
        SearchOutput: ``trials`` (every evaluated config) and ``best`` (highest score).
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    trials: list[EvalResult] = []

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=sampler_seed))
    study.optimize(
        partial(_objective, dataset=dataset, seed=seed, k=k, trials=trials),
        n_trials=MAX_ITERATIONS,
    )

    return write_search_output("bayes", trials, out_path)
