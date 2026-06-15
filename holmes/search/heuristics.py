"""Heuristic initial hyperparameters derived from dataset characteristics.

Saves an iteration that would otherwise just confirm the defaults are reasonable. Each rule is a
causal claim grounded in a dataset signal or in the geometry of the search space, and every value
is clamped into the live :data:`holmes.config.HOLMES_SPACE`, so the heuristic starts in-bounds by
construction even when ``GRID_SPACE`` is retuned:

- ``factors`` — capacity tracks data volume: the searchable floor at 100k interactions, doubling
  per decade above it.
- ``regularization`` — the geometric midpoint of the searchable range: equal log-headroom for the
  loop's bold multiplicative moves in either direction.
- ``iterations`` — the midpoint of the sweep range, leaving headroom to raise it if the fit has
  not converged.
- ``alpha`` — derived from the dataset's mean stored rating to hit the Hu et al. (2008)
  confidence operating point ``c = 1 + alpha * r ~= 100`` on an average positive.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from holmes.config import HOLMES_SPACE, ALSParams

if TYPE_CHECKING:
    from holmes.data.dataset import Dataset

_FACTORS_ANCHOR_INTERACTIONS = 100_000
"""Interaction count at which the capacity rule sits exactly at the searchable floor."""

_TARGET_CONFIDENCE = 100.0
"""Target confidence ``c = 1 + alpha * r`` on an average positive — Hu et al. 2008's operating
point (their alpha=40 with r ~= 2.5); ``alpha`` is chosen to hit it given the mean stored rating."""


def _clamp(name: str, value: float) -> float:
    """Clamp ``value`` into the HOLMES search bounds for hyperparameter ``name``."""
    low, high = HOLMES_SPACE[name]
    return min(max(value, low), high)


def initial_params(dataset: Dataset) -> tuple[ALSParams, dict[str, str]]:
    """Suggest a sensible starting configuration for ``dataset``.

    Args:
        dataset: The preprocessed interaction matrix.

    Returns:
        tuple[ALSParams, dict[str, str]]: The starting parameters and a per-field rationale
        explaining why each value was chosen, for the trajectory log.
    """
    n_interactions = dataset.n_interactions
    mean_rating = float(dataset.train_ui.data.mean()) if n_interactions else 1.0

    factors_floor = HOLMES_SPACE["factors"][0]
    decades_above_anchor = math.log10(max(n_interactions, 1) / _FACTORS_ANCHOR_INTERACTIONS)
    factors = round(_clamp("factors", factors_floor * 2**decades_above_anchor))

    reg_low, reg_high = HOLMES_SPACE["regularization"]
    regularization = math.sqrt(reg_low * reg_high)

    iter_low, iter_high = HOLMES_SPACE["iterations"]
    iterations = round((iter_low + iter_high) / 2)

    alpha = _clamp("alpha", (_TARGET_CONFIDENCE - 1.0) / mean_rating)

    params = ALSParams(factors=factors, regularization=regularization, iterations=iterations, alpha=alpha)
    rationale = {
        "factors": (
            f"{n_interactions:,} interactions; capacity tracks data volume — the searchable floor "
            f"({factors_floor}) at {_FACTORS_ANCHOR_INTERACTIONS:,} interactions, doubled per decade "
            f"above it and clamped to the search bounds"
        ),
        "regularization": (
            f"{regularization} is the geometric midpoint of the searchable range [{reg_low}, {reg_high}] — "
            f"equal log-headroom to move boldly in either direction"
        ),
        "iterations": (
            f"midpoint of the sweep range [{iter_low}, {iter_high}]; enough for ALS to converge on most "
            f"matrices, with headroom to raise it if train_recon_error stays high"
        ),
        "alpha": (
            f"mean stored rating {mean_rating:.2f}; alpha={alpha:g} targets the Hu et al. confidence "
            f"operating point c = 1 + alpha*r ~= {_TARGET_CONFIDENCE:g} on an average positive, "
            f"clamped to the search bounds"
        ),
    }
    return params, rationale


def initial_hypothesis(params: ALSParams, rationale: dict[str, str]) -> dict[str, str]:
    """Repackage the per-HP rationales into a falsifiable iter-1 hypothesis.

    The heuristic rules are themselves causal claims (e.g. "this many interactions support this
    much capacity without overfitting"). This function lifts them into the skill's
    ``mechanism / outcome / falsifiers`` shape so iteration 1 can run via the CLI without the
    LLM having to author the hypothesis by hand.

    Args:
        params: The heuristic's chosen hyperparameters.
        rationale: The per-HP rationales returned alongside ``params`` by
            :func:`initial_params`.

    Returns:
        dict[str, str]: A hypothesis with the three required string fields.
    """
    mechanism = "; ".join(f"{name}={value} — {rationale[name]}" for name, value in params.to_dict().items())
    outcome = (
        "ndcg lands above a trivial baseline but below a well-tuned upper bound; "
        "this is the floor that subsequent iterations must beat."
    )
    falsifiers = (
        f"If train_recon_error stays high, the capacity rule (factors={params.factors}) or the "
        f"sweep midpoint (iterations={params.iterations}) underspecified the fit. If "
        f"train_test_ndcg_gap is large, regularization={params.regularization} sits too low in the "
        f"searchable range for this matrix. If tail_recall is near zero, the confidence target "
        f"behind alpha={params.alpha:g} over-weights head items."
    )
    return {"mechanism": mechanism, "outcome": outcome, "falsifiers": falsifiers}
