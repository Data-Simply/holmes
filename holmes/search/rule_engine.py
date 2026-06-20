"""Rule-engine HPO strategy: the reasoning guide's diagnostic patterns as deterministic code.

A *rules-based ablation* of HOLMES. It runs the same diagnostic battery, the same search region,
and the same fixed budget as grid/random/bayes/HOLMES — but with **no LLM and no falsification
protocol**. It does not write a hypothesis, predict which mechanism will move, classify the result
into validated/coincidence/null, or disambiguate. It just reacts to the current diagnostics with
the guide's prescribed move (``skill/references/REASONING_GUIDE.md``). That removal is the point:
comparing HOLMES against this isolates whether HOLMES's win is the *discipline* or the injected
*guide*. The guide covers moves, not initialization, so iteration 1 starts from the
dataset-independent :class:`~holmes.config.ALSParams` defaults — the engine shares no starting rule
with HOLMES, which seeds iteration 1 from the heuristic.

Three deliberate design choices keep the comparison fair by construction:

  * **Conflict resolution = fixed priority order, first match wins.** The patterns are not mutually
    exclusive (a fit can match several at once); an LLM resolves this fluidly, the rule engine
    cannot, so it checks the most *specific* patterns first.
  * **Thresholds = hybrid.** Diagnostics with a natural scale use absolute / cross-metric cutoffs
    (so a *persistent* pathology still fires); gauge- or data-scale-free axes
    (``mean_factor_norm``, ``train_recon_error``, ``catalog_coverage``, ``ndcg``) are read relative
    to the metric's own trajectory history (median ± k*MAD), where no absolute reference exists.
  * **One fit per iteration.** The seed-stability protocol behind the guide's exploding-factor
    (pattern 6) and integrity (pattern 9) patterns is dropped: re-fitting at extra seeds would
    spend more fits per iteration than the other strategies and break the equal-fit-budget
    comparison. Pattern 6 keys on ``mean_factor_norm`` alone; pattern 9 is omitted (the harness is
    deterministic and reproducible by construction). The loop runs the full budget — when no move
    remains it re-fits the current config — so the result is a clean best-of-N vs the baselines.

Mirrors ``holmes/search/random_search.py`` structure: a ``run_rule_engine`` driver returning the
shared ``SearchOutput`` via ``write_search_output``.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from holmes.config import DEFAULT_SEED, INTEGER_PARAMS, MAX_ITERATIONS, RULE_SPACE, TOP_K, ALSParams
from holmes.search.harness import EvalResult, SearchOutput, evaluate_config, log_trial, write_search_output

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset

# Absolute / cross-metric cutoffs for the axes that have a natural scale. Grounded in the metric
# definitions (holmes/metrics/diagnostics.py), so a *persistent* pathology still trips the flag —
# a self-relative band would read a steadily-bad metric as "mid" and never fire.
_POPULARITY_HIGH = 0.8
"""``avg_rec_popularity`` is a [0, 1] popularity percentile; above this the recs are head-dominated
(the guide's '-> 1')."""

_TAIL_STARVED_RATIO = 0.5
"""Tail starvation: ``tail_recall`` below this fraction of overall ``recall`` — the tail is served
far worse than the head. Relative to ``recall`` rather than an absolute floor so it is robust to how
hard the dataset is."""

_OVERFIT_RATIO = 0.5
"""Overfitting: ``train_test_ndcg_gap`` exceeding this fraction of ``train_ndcg`` — held-out NDCG is
less than half the memorized training NDCG."""

_MIN_HISTORY_FOR_SPREAD = 3
"""Fewest trajectory points that yield a usable median/MAD; below this every value reads mid."""


def _rel_level(history: list[float], value: float, k: float = 1.0) -> int:
    """Classify ``value`` as -1 (low), 0 (mid), or +1 (high) vs the metric's running median.

    Spread is the median absolute deviation; the band is ``median +/- k*MAD``. Self-calibrating,
    so no per-metric magic constants. With fewer than three points there is no usable spread yet,
    so everything reads mid and the engine leans on its capacity-first fallback.

    Args:
        history: All values this metric has taken so far in the run, in order.
        value: The latest value to classify.
        k: Band half-width in MAD units.

    Returns:
        int: -1, 0, or +1.
    """
    if len(history) < _MIN_HISTORY_FOR_SPREAD:
        return 0
    med = statistics.median(history)
    mad = statistics.median([abs(x - med) for x in history]) or 1e-9
    if value > med + k * mad:
        return 1
    if value < med - k * mad:
        return -1
    return 0


def _rel(trajectory: list[EvalResult], key: str) -> int:
    """Relative level (-1/0/+1) of ``key``'s latest value against its trajectory history."""
    history = [t["metrics"][key] for t in trajectory]
    return _rel_level(history, history[-1])


def _clamp(name: str, value: float) -> float:
    """Clamp ``value`` for hyperparameter ``name`` into its ``RULE_SPACE`` bounds."""
    low, high = RULE_SPACE[name]
    return max(low, min(high, value))


def _move(params: ALSParams, **changes: float) -> ALSParams:
    """Return a new :class:`ALSParams` with ``changes`` applied, clamped to ``RULE_SPACE``.

    Integer hyperparameters are rounded before construction: a fractional move such as halving an
    odd ``factors`` would otherwise be rejected by :meth:`ALSParams.from_dict` (which refuses
    non-integral integer fields), so the rounding makes the move resolve deterministically.
    """
    values = params.to_dict()
    values.update(changes)
    clamped = {name: _clamp(name, value) for name, value in values.items()}
    for name in INTEGER_PARAMS:
        clamped[name] = round(clamped[name])
    return ALSParams.from_dict(clamped)


# A flat, priority-ordered chain of guarded returns is the clearest shape for this pattern matcher:
# the order is the conflict-resolution rule and each branch is one guide pattern, so the return
# count and branching exceed the default thresholds by design (splitting it into helpers would hide
# the priority order that is the algorithm).
def next_params(trajectory: list[EvalResult]) -> ALSParams:  # noqa: C901, PLR0911
    """Choose the next configuration by matching the reasoning guide's patterns.

    Patterns are checked in fixed priority order, most specific first, and the first match wins —
    that ordering is the conflict-resolution rule (a fit can match several patterns at once). The
    seed-stability patterns are not implemented here (see the module docstring): pattern 6 keys on
    ``mean_factor_norm`` alone and pattern 9 is omitted. Always returns a configuration — when no
    pattern fires and no capacity move remains, the current params are returned unchanged so the
    driver re-fits them and spends the full budget.

    Args:
        trajectory: Evaluated iterations so far, in order; the last entry is the most recent fit.

    Returns:
        ALSParams: The next configuration to fit.
    """
    last = trajectory[-1]
    params = ALSParams.from_dict(last["params"])
    metrics = last["metrics"]

    # Scale-free axes: read relative to the run's own history.
    ndcg = _rel(trajectory, "ndcg")
    recon = _rel(trajectory, "train_recon_error")
    norm = _rel(trajectory, "mean_factor_norm")
    coverage = _rel(trajectory, "catalog_coverage")

    # Natural-scale axes: absolute / cross-metric flags.
    recall = metrics["recall"]
    train_ndcg = metrics["train_ndcg"]
    popularity_high = metrics["avg_rec_popularity"] > _POPULARITY_HIGH
    tail_starved = recall > 0 and metrics["tail_recall"] < _TAIL_STARVED_RATIO * recall
    overfit = train_ndcg > 0 and metrics["train_test_ndcg_gap"] > _OVERFIT_RATIO * train_ndcg

    prev_entry = trajectory[-2] if len(trajectory) > 1 else None
    prev = ALSParams.from_dict(prev_entry["params"]) if prev_entry is not None else None
    iterations_just_raised = prev is not None and params.iterations > prev.iterations
    recon_fell = prev_entry is not None and metrics["train_recon_error"] < prev_entry["metrics"]["train_recon_error"]
    factors_just_doubled = prev is not None and params.factors > prev.factors

    fac_hi = RULE_SPACE["factors"][1]
    reg_lo, reg_hi = RULE_SPACE["regularization"]
    iter_hi = RULE_SPACE["iterations"][1]
    alpha_lo = RULE_SPACE["alpha"][0]

    # 5. Over-regularized collapse -> lower reg, then raise alpha if still flat. (Most specific:
    #    shrunk embeddings AND a poor fit AND poor ranking all at once.)
    if norm == -1 and recon == 1 and ndcg == -1:
        if params.regularization > reg_lo:
            return _move(params, regularization=params.regularization / 10)
        return _move(params, alpha=params.alpha * 2)

    # 4. Not converged -> push iterations to the bound (high recon that fell when iters last rose).
    if recon == 1 and iterations_just_raised and recon_fell:
        return _move(params, iterations=iter_hi)

    # 1. Underfitting -> double factors, else push iterations to the bound.
    if ndcg == -1 and recon == 1 and not overfit:
        if params.factors < fac_hi:
            return _move(params, factors=params.factors * 2)
        return _move(params, iterations=iter_hi)

    # 2. Overfitting / memorization -> raise reg a decade, else halve factors.
    if overfit and norm == 1:
        if params.regularization < reg_hi:
            return _move(params, regularization=params.regularization * 10)
        return _move(params, factors=params.factors / 2)

    # 6. Exploding factors / instability -> raise reg (seed-stability condition dropped).
    if norm == 1:
        return _move(params, regularization=params.regularization * 10)

    # 7. Tail starvation -> lower alpha and give niche structure room (deliberate two-HP move).
    if ndcg >= 0 and tail_starved and popularity_high:
        return _move(params, alpha=params.alpha / 4, factors=params.factors * 2)

    # 3. Popularity bias -> lower alpha, else raise factors.
    if ndcg >= 0 and popularity_high and coverage == -1:
        if params.alpha > alpha_lo:
            return _move(params, alpha=params.alpha / 4)
        return _move(params, factors=params.factors * 2)

    # 8. Near optimum -> refine with a small nudge rather than a bold capacity move.
    if ndcg == 0 and factors_just_doubled:
        return _move(params, regularization=params.regularization * 2)

    # Fallback: nothing fired -> one capacity move if room remains, else re-fit the current config.
    if params.factors < fac_hi:
        return _move(params, factors=params.factors * 2)
    return params


def run_rule_engine(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    k: int = TOP_K,
    out_path: Path | None = None,
) -> SearchOutput:
    """Run the rule engine for the shared fixed budget and return the trial log and best.

    Starts from the dataset-independent :class:`ALSParams` defaults — the guide prescribes moves,
    not an initial config, so the engine shares no starting rule with HOLMES (which seeds iteration
    1 from the heuristic). It then applies :func:`next_params` each round for the full
    :data:`holmes.config.MAX_ITERATIONS` budget. One seed is fit per configuration, matching the
    other strategies. The run is fully deterministic: identical inputs yield an identical trajectory.

    Args:
        dataset: Preprocessed interaction matrix.
        seed: Random seed for every fit.
        k: Ranking cut-off.
        out_path: Optional path to write the full results JSON.

    Returns:
        SearchOutput: ``trials`` (every evaluated config) and ``best`` (highest NDCG@K).
    """
    params = ALSParams()
    trials: list[EvalResult] = []
    for i in range(1, MAX_ITERATIONS + 1):
        result = evaluate_config(params, dataset, seed=seed, k=k, split="val")
        trials.append(result)
        log_trial("rule", i, MAX_ITERATIONS, result)
        params = next_params(trials)
    return write_search_output("rule", trials, out_path)
