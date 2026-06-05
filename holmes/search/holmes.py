"""The HOLMES agentic loop: run ONE iteration and append it to the trajectory log.

The loop is driven by an LLM (Claude Code) reading the trajectory between iterations. This
module owns the mechanical half: it fits a configuration, computes the diagnostic battery,
and appends a structured entry. The LLM owns the reasoning half: before each run it writes a
falsifiable hypothesis (mechanism + outcome + falsifiers); after each run it writes the
interpretation and one of the four validation statuses. See ``skill/references/REASONING_GUIDE.md``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict

from holmes.config import DEFAULT_SEED, HOLMES_SPACE, MAX_ITERATIONS, TOP_K, ALSParams
from holmes.search.harness import evaluate_config

if TYPE_CHECKING:
    from pathlib import Path

    from holmes.data.dataset import Dataset

VALIDATION_STATUSES = ("validated", "partial_mechanism", "coincidence", "null")
"""The four outcome states an iteration can resolve to, per the HOLMES method."""

_EMPTY_HYPOTHESIS = {"mechanism": "", "outcome": "", "falsifiers": ""}


def _check_params_in_bounds(params: ALSParams) -> None:
    """Reject LLM-authored params that fall outside :data:`holmes.config.HOLMES_SPACE`.

    Grid and Bayes stay in their respective spaces by construction; the HOLMES loop relies on
    this check because the LLM authors the params directly. ``HOLMES_SPACE`` currently matches
    ``BAYES_SPACE`` so all three strategies optimize over the same region.

    Args:
        params: The hyperparameters parsed from the iteration input JSON.

    Raises:
        ValueError: If any hyperparameter is outside ``HOLMES_SPACE``.
    """
    values = params.to_dict()
    for name, (low, high) in HOLMES_SPACE.items():
        value = values[name]
        if value < low or value > high:
            msg = (
                f"{name}={value} is outside the HOLMES search space [{low}, {high}]. "
                "All three strategies share this range for fair comparison; "
                "adjust the input params and retry."
            )
            raise ValueError(msg)


class TrajectoryEntry(TypedDict):
    """One iteration of the append-only trajectory log.

    Attributes:
        iteration: 1-based iteration number.
        params: The ALS hyperparameters fit this iteration.
        seed: The random seed fit.
        hypothesis: The pre-run falsifiable hypothesis (mechanism / outcome / falsifiers).
        metrics: The diagnostic battery for this fit.
        score: The primary metric (NDCG@K).
        validation_status: One of :data:`VALIDATION_STATUSES`, written by the LLM after the run.
        interpretation: The LLM's post-run interpretation.
    """

    iteration: int
    params: dict[str, float]
    seed: int
    hypothesis: dict[str, str]
    metrics: dict[str, float]
    score: float
    validation_status: str | None
    interpretation: str | None


def load_trajectory(path: Path) -> list[TrajectoryEntry]:
    """Load the append-only trajectory log, returning an empty list if absent.

    Args:
        path: Path to the trajectory JSON file.

    Returns:
        list[TrajectoryEntry]: The recorded iterations in order.
    """
    if not path.exists():
        return []
    return json.loads(path.read_text())


def annotate_iteration(
    trajectory_path: Path,
    *,
    iteration: int,
    validation_status: str,
    interpretation: str,
) -> TrajectoryEntry:
    """Fill ``validation_status`` and ``interpretation`` on an existing trajectory entry.

    The CLI calls this so the LLM never has to edit ``trajectory.json`` by hand — flag-driven
    annotation is deterministic where a free-form file edit is not.

    Args:
        trajectory_path: Path to the trajectory JSON file.
        iteration: 1-based iteration number to annotate.
        validation_status: One of :data:`VALIDATION_STATUSES`.
        interpretation: The LLM's post-run interpretation text.

    Returns:
        TrajectoryEntry: The updated trajectory entry.

    Raises:
        FileNotFoundError: If ``trajectory_path`` does not exist.
        ValueError: If ``validation_status`` is not in :data:`VALIDATION_STATUSES`, or no entry
            matches ``iteration``.
    """
    if validation_status not in VALIDATION_STATUSES:
        msg = f"validation_status must be one of {VALIDATION_STATUSES}, got {validation_status!r}."
        raise ValueError(msg)
    if not trajectory_path.exists():
        msg = f"Trajectory file {trajectory_path} does not exist."
        raise FileNotFoundError(msg)
    trajectory = load_trajectory(trajectory_path)
    matching = [entry for entry in trajectory if entry["iteration"] == iteration]
    if len(matching) == 0:
        msg = f"No entry for iteration {iteration} in {trajectory_path}."
        raise ValueError(msg)
    entry = matching[0]
    entry["validation_status"] = validation_status
    entry["interpretation"] = interpretation
    trajectory_path.write_text(json.dumps(trajectory, indent=2))
    return entry


def run_iteration(
    dataset: Dataset,
    spec: dict,
    trajectory_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    k: int = TOP_K,
    max_iterations: int = MAX_ITERATIONS,
    show_progress: bool = False,
) -> TrajectoryEntry:
    """Run one HOLMES iteration from an input spec and append it to the trajectory.

    The spec must contain ``params`` (an ALS hyperparameter mapping) and may contain ``hypothesis``
    (with ``mechanism``, ``outcome``, and ``falsifiers`` written *before* the run). The
    ``validation_status`` and ``interpretation`` fields are left null for the LLM to fill in
    after reading the metrics. One seed is fit per iteration; assess stability by repeating the
    iteration with different seeds. File I/O (reading the spec from disk) is the caller's job;
    keeping this function dict-based lets the CLI build the spec from either a JSON file or
    individual flags.

    Args:
        dataset: Preprocessed interaction matrix.
        spec: Iteration input as an in-memory mapping ``{"params": {...}, "hypothesis": {...}}``.
        trajectory_path: Path to the append-only trajectory log.
        seed: Random seed fit this iteration.
        k: Ranking cut-off.
        max_iterations: Hard cap on total trajectory length, shared across grid/bayes/HOLMES so
            the search-budget comparison is fixed. Defaults to :data:`holmes.config.MAX_ITERATIONS`.
        show_progress: Forwarded to :func:`evaluate_config`.

    Returns:
        TrajectoryEntry: The appended trajectory entry, including computed metrics.

    Raises:
        KeyError: If ``spec`` lacks a ``params`` field.
        RuntimeError: If the trajectory has already reached ``max_iterations``.
    """
    if "params" not in spec:
        msg = "Iteration spec must contain a 'params' field."
        raise KeyError(msg)
    params = ALSParams.from_dict(spec["params"])
    _check_params_in_bounds(params)
    hypothesis = {**_EMPTY_HYPOTHESIS, **spec.get("hypothesis", {})}

    trajectory = load_trajectory(trajectory_path)
    if len(trajectory) >= max_iterations:
        msg = (
            f"Budget exhausted: trajectory has {len(trajectory)} iterations, "
            f"--max-iterations is {max_iterations}. Stop and report the best entry; do not "
            "fit further configs."
        )
        raise RuntimeError(msg)

    result = evaluate_config(params, dataset, seed=seed, k=k, split="val", show_progress=show_progress)
    entry: TrajectoryEntry = {
        "iteration": len(trajectory) + 1,
        "params": params.to_dict(),
        "seed": seed,
        "hypothesis": hypothesis,
        "metrics": result["metrics"],
        "score": result["score"],
        "validation_status": None,
        "interpretation": None,
    }
    trajectory.append(entry)
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path.write_text(json.dumps(trajectory, indent=2))

    print(json.dumps(entry, indent=2))
    return entry
