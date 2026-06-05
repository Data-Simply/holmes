"""Heuristic initial hyperparameters derived from dataset characteristics.

Saves an iteration that would otherwise just confirm the defaults are reasonable. The rules
encode standard implicit-ALS practice: more factors for larger catalogs, stronger confidence
scaling (``alpha``) for sparser matrices, and moderate regularization to start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from holmes.config import ALSParams

if TYPE_CHECKING:
    from holmes.data.dataset import Dataset

_LARGE_INTERACTIONS = 1_000_000
_SMALL_INTERACTIONS = 100_000
_SPARSE_DENSITY = 1e-3


def initial_params(dataset: Dataset) -> tuple[ALSParams, dict[str, str]]:
    """Suggest a sensible starting configuration for ``dataset``.

    Args:
        dataset: The preprocessed interaction matrix.

    Returns:
        tuple[ALSParams, dict[str, str]]: The starting parameters and a per-field rationale
        explaining why each value was chosen, for the trajectory log.
    """
    n_interactions = dataset.n_interactions
    density = dataset.density

    if n_interactions >= _LARGE_INTERACTIONS:
        factors = 128
        factors_reason = f"{n_interactions:,} interactions is large; 128 factors to capture structure"
    elif n_interactions <= _SMALL_INTERACTIONS:
        factors = 64
        factors_reason = f"only {n_interactions:,} interactions; 64 factors (the searchable floor) to limit overfitting"
    else:
        factors = 64
        factors_reason = f"{n_interactions:,} interactions is moderate; 64 factors as a balanced start"

    if density < _SPARSE_DENSITY:
        alpha = 40.0
        alpha_reason = f"density {density:.1e} is very sparse; alpha=40 to upweight rare positives"
    else:
        alpha = 15.0
        alpha_reason = f"density {density:.1e} is moderate; alpha=15 as a gentler confidence weight"

    params = ALSParams(factors=factors, regularization=0.01, iterations=20, alpha=alpha)
    rationale = {
        "factors": factors_reason,
        "regularization": "0.01 is a standard mid-range L2 penalty for implicit ALS",
        "iterations": "20 sweeps is typically enough for ALS to converge on matrices this size",
        "alpha": alpha_reason,
    }
    return params, rationale


def initial_hypothesis(params: ALSParams, rationale: dict[str, str]) -> dict[str, str]:
    """Repackage the per-HP rationales into a falsifiable iter-1 hypothesis.

    The heuristic rules are themselves causal claims (e.g. "factors=128 because the dataset is
    large enough to support more capacity without overfitting"). This function lifts them into the
    skill's ``mechanism / outcome / falsifiers`` shape so iteration 1 can run via the CLI without
    the LLM having to author the hypothesis by hand.

    Args:
        params: The heuristic's chosen hyperparameters.
        rationale: The per-HP rationales returned alongside ``params`` by
            :func:`initial_params`.

    Returns:
        dict[str, str]: A hypothesis with the three required string fields.
    """
    values = params.to_dict()
    mechanism = "; ".join(
        f"{name}={values[name]} — {rationale[name]}" for name in ("factors", "regularization", "iterations", "alpha")
    )
    outcome = (
        "ndcg lands above a trivial baseline but below a well-tuned upper bound; "
        "this is the floor that subsequent iterations must beat."
    )
    falsifiers = (
        "If train_recon_error stays high, the factors/iterations rules underspecified the capacity "
        "needed to fit this matrix. If train_test_ndcg_gap is large, regularization=0.01 is too low "
        "for this density. If tail_recall is near zero, the alpha rule failed to surface tail items."
    )
    return {"mechanism": mechanism, "outcome": outcome, "falsifiers": falsifiers}
