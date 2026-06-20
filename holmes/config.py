"""Shared configuration: hyperparameter space, evaluation settings, and paths."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_type_hints

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_CACHE_DIR = DATA_DIR / "raw_cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Evaluation settings ---------------------------------------------------
TOP_K = 10
"""Cut-off for all ranking metrics (Recall@K, NDCG@K, ...)."""

DEFAULT_SEED = 0
"""Random seed for a single fit. Stability across initializations is obtained by repeating a run
with different seeds and aggregating externally, not by fitting multiple seeds per call."""

EVAL_SAMPLE_USERS = 5000
"""Cap on users scored during evaluation, for tractable diagnostics on large matrices."""

EVAL_SAMPLE_SEED = 7919
"""Fixed seed for all evaluation-time sampling (eval users, reconstruction entries).

Deliberately decoupled from the fit ``--seed``: re-running a config with another seed must change
only the model initialization, never the evaluated population — otherwise eval-sampling noise is
indistinguishable from model instability in the seed-stability protocol."""

HEAD_ITEM_FRACTION = 0.2
"""Fraction of items (by popularity) treated as the popular 'head'; the rest are the tail."""


def _coerce_int(value: float | str, name: str) -> int:
    """Coerce ``value`` to int, rejecting non-integral inputs instead of silently truncating.

    Args:
        value: A value expected to represent an integer (``128``, ``128.0``, or ``"128"``).
        name: Field name, used in the error message.

    Returns:
        int: The integer value.

    Raises:
        ValueError: If ``value`` is not integer-valued (e.g. ``63.9``) or not numeric.
    """
    coerced = int(value)
    if coerced != float(value):
        msg = f"{name} must be an integer, got {value!r}."
        raise ValueError(msg)
    return coerced


@dataclass(frozen=True)
class ALSParams:
    """Hyperparameters for the ALS recommender.

    Attributes:
        factors: Latent dimensionality of user and item embeddings.
        regularization: L2 penalty applied during the alternating least-squares solve.
        iterations: Number of alternating optimization sweeps.
        alpha: Confidence scaling applied to the interaction matrix (Hu et al. 2008).
    """

    factors: int = 64
    regularization: float = 0.01
    iterations: int = 20
    alpha: float = 40.0

    def to_dict(self) -> dict[str, float]:
        """Return the parameters as a plain dictionary.

        Returns:
            dict[str, float]: Mapping of hyperparameter name to value.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> ALSParams:
        """Build an :class:`ALSParams` from a mapping of exactly the hyperparameter fields.

        Unknown and missing keys are rejected rather than silently dropped/defaulted — a typo'd
        key or a wrong-shaped mapping (e.g. a results file passed to ``holmes eval --params``)
        must error, not evaluate the all-default configuration. The field set is derived from
        the dataclass so a new hyperparameter cannot be silently filtered out here.

        Args:
            data: Mapping containing exactly the hyperparameter fields.

        Returns:
            ALSParams: The parsed parameters.

        Raises:
            ValueError: If ``data`` has unknown or missing keys, or a non-integral value for an
                integer field.
        """
        field_names = [field.name for field in fields(cls)]
        unknown = sorted(set(data) - set(field_names))
        if unknown:
            msg = f"Unknown hyperparameter keys {unknown}; expected exactly {field_names}."
            raise ValueError(msg)
        missing = sorted(set(field_names) - set(data))
        if missing:
            msg = f"Missing hyperparameter keys {missing}; expected exactly {field_names}."
            raise ValueError(msg)
        values = {
            name: (_coerce_int(data[name], name) if name in INTEGER_PARAMS else float(data[name]))
            for name in field_names
        }
        return cls(**values)


PARAM_SCALES: dict[str, str] = {
    "factors": "log",
    "regularization": "log",
    "iterations": "linear",
    "alpha": "log",
}
"""Sampling scale per hyperparameter, shared by the random and Bayesian samplers.

Both samplers read this table so the two strategies draw from the same measure over the hull by
construction — if the scales drifted apart, the benchmark would compare sampling distributions,
not optimizer behavior. ``factors``/``regularization``/``alpha`` span orders of magnitude (log);
``iterations`` is a small linear count."""

INTEGER_PARAMS = frozenset(name for name, type_ in get_type_hints(ALSParams).items() if type_ is int)
"""Hyperparameters taking integer values, the single source of integer-ness for both ``from_dict``
coercion and the samplers' draw/round. Derived from resolved type hints (``get_type_hints``
collapses the stringized annotations ``from __future__`` produces) so a non-``int`` field can't be
silently treated as integer."""


# --- Grid-search space -----------------------------------------------------
# Anchored on the heuristic point estimate. With the rating-weighted matrix (r in {4,5}; see
# ``holmes/data/preprocess.py``), Hu et al. confidence ``c_ui = 1 + alpha * r_ui`` sets the
# joint scale: ``lambda*I`` has to bind against the diagonal of ``Y^T C_u Y`` which tracks
# ``alpha * r``, so reg sits in 0.01-1 (a standard log-spaced implicit-ALS L2 range, bracketing
# the 0.01 ``ALSParams``/heuristic default) and alpha is centred where c_ui lands in ~20-200.
# ``alpha=1.0`` is omitted (it turns off confidence weighting). BAYES_SPACE and HOLMES_SPACE
# are derived from this grid's hull, so changing GRID_SPACE retunes the other two too.
GRID_SPACE: dict[str, list[float]] = {
    "factors": [64, 128, 256, 512],
    "regularization": [0.01, 0.1, 1.0],
    "iterations": [15, 30],
    "alpha": [5.0, 15.0, 40.0],
}

# All three search strategies share this cap so the comparison is at a fixed fit budget.
# Derived from the grid so a future GRID_SPACE change automatically propagates to bayes and
# HOLMES — the three never drift out of alignment.
MAX_ITERATIONS = math.prod(len(values) for values in GRID_SPACE.values())


def _grid_hull(grid: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Return the (min, max) per-axis continuous hull of a discrete grid.

    Used to derive the Bayes and HOLMES continuous spaces from :data:`GRID_SPACE` so the three
    strategies optimize over the same region by construction — changing the grid automatically
    propagates to the other two.
    """
    return {name: (min(values), max(values)) for name, values in grid.items()}


# --- Random-, Bayesian- (Optuna), and HOLMES agentic-search spaces ---------
# All three are the continuous hull of GRID_SPACE so every strategy optimizes over the same
# region — the comparison isolates optimizer behavior from search-space coverage. They are
# kept as distinct names so each strategy owns its space and an in-test ``monkeypatch`` on
# one does not silently bind the others.
RANDOM_SPACE: dict[str, tuple[float, float]] = _grid_hull(GRID_SPACE)
BAYES_SPACE: dict[str, tuple[float, float]] = _grid_hull(GRID_SPACE)
HOLMES_SPACE: dict[str, tuple[float, float]] = _grid_hull(GRID_SPACE)
# The deterministic rule-engine ablation (holmes/search/rule_engine.py) clamps its moves to this
# space. A distinct object, equal by value, for the same reason as the others above.
RULE_SPACE: dict[str, tuple[float, float]] = _grid_hull(GRID_SPACE)
