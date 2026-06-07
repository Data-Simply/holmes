"""Bayesian-optimization baseline using Optuna's TPE sampler."""

from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING

import optuna

from holmes.config import BAYES_SPACE, DEFAULT_SEED, MAX_ITERATIONS, TOP_K, ALSParams
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, select_best

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset


def _suggest_params(trial: optuna.Trial) -> ALSParams:
    """Sample an :class:`ALSParams` from the Optuna trial over :data:`BAYES_SPACE`.

    The log/linear scale per hyperparameter is intrinsic: ``factors``, ``regularization``, and
    ``alpha`` span orders of magnitude so they are log-scaled; ``iterations`` is a small linear count.
    """
    factors_lo, factors_hi = BAYES_SPACE["factors"]
    reg_lo, reg_hi = BAYES_SPACE["regularization"]
    iter_lo, iter_hi = BAYES_SPACE["iterations"]
    alpha_lo, alpha_hi = BAYES_SPACE["alpha"]
    return ALSParams(
        factors=trial.suggest_int("factors", int(factors_lo), int(factors_hi), log=True),
        regularization=trial.suggest_float("regularization", reg_lo, reg_hi, log=True),
        iterations=trial.suggest_int("iterations", int(iter_lo), int(iter_hi)),
        alpha=trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True),
    )


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
    metrics = result["metrics"]
    timing = f"fit={metrics['fit_time_seconds']:.2f}s eval={metrics['eval_time_seconds']:.2f}s"
    print(
        f"[bayes {trial.number + 1}/{MAX_ITERATIONS}] {params.to_dict()} -> val ndcg={result['score']:.4f}  {timing}",
    )
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

    best = select_best(trials)
    output: SearchOutput = {"strategy": "bayes", "n_trials": len(trials), "best": best, "trials": trials}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Wrote {len(trials)} bayes trials to {out_path}")
    print(f"Best bayes config: {best['params']} (val ndcg={best['score']:.4f})")
    return output
