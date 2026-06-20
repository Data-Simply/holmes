"""Guardrail: the skill docs and the diagnostic battery cannot drift apart.

The LLM driving the HOLMES loop reasons from ``skill/SKILL.md`` and ``skill/references/*.md``.
A metric name in those docs that the battery never emits produces unfalsifiable hypotheses —
the falsifier references a key that is never measured, so the iteration can never be classified
on its stated terms. The allowed vocabulary is derived from its sources (a computed battery,
the trajectory-entry and eval-result schemas, the hyperparameter dataclass), never hand-copied.
"""

import dataclasses
import re

import pytest

from holmes.config import PROJECT_ROOT, ALSParams
from holmes.search.harness import EvalResult, evaluate_config
from holmes.search.holmes import VALIDATION_STATUSES, TrajectoryEntry

_SKILL_DIR = PROJECT_ROOT / "skill"
_DOC_PATHS = (
    _SKILL_DIR / "SKILL.md",
    _SKILL_DIR / "references" / "REASONING_GUIDE.md",
    _SKILL_DIR / "references" / "TRAJECTORY_SCHEMA.md",
)

# Identifier-like tokens containing an underscore — the shape every diagnostic metric name has.
_UNDERSCORE_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_BACKTICKED_TOKEN = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")


@pytest.fixture
def battery_keys(books_dataset) -> set[str]:
    """The metric keys one real evaluation records, including the harness's timing fields."""
    result = evaluate_config(ALSParams(factors=8, iterations=3), books_dataset, seed=0, k=5)
    return set(result["metrics"])


@pytest.fixture
def documented_vocabulary(battery_keys, books_dataset) -> set[str]:
    """Every underscore-containing identifier the docs may legitimately reference."""
    return (
        battery_keys
        | set(TrajectoryEntry.__annotations__)
        | set(EvalResult.__annotations__)
        | set(VALIDATION_STATUSES)
        | {field.name for field in dataclasses.fields(ALSParams)}
        # What `holmes ranges` prints (see cli._cmd_ranges): the budget plus the dataset signal
        # the agent reasons from to choose iteration 1.
        | {"max_iterations"}
        | set(books_dataset.describe())
    )


def test_schema_example_metrics_block_matches_the_battery(battery_keys):
    """The canonical example's ``metrics`` object must list exactly the recorded battery —
    a metric added to (or renamed in) the battery must show up here, and vice versa."""
    schema = (_SKILL_DIR / "references" / "TRAJECTORY_SCHEMA.md").read_text()
    metrics_block = schema.split('"metrics": {')[1].split("}")[0]
    documented = set(re.findall(r'"(\w+)":', metrics_block))
    assert documented == battery_keys


@pytest.mark.parametrize("doc_path", _DOC_PATHS, ids=lambda p: p.name)
def test_docs_reference_only_real_identifiers(doc_path, documented_vocabulary):
    """Every metric-shaped identifier in the skill docs must exist in the code's vocabulary.

    This is the guardrail against the exact bug it was written for: a schema example
    hypothesizing about a metric (``mean_item_factor_norm``) the battery never emits, which an
    imitating agent turns into unfalsifiable hypotheses.
    """
    text = doc_path.read_text()
    # Prose names identifiers in backticks; fenced example blocks contain bare identifiers.
    fenced = "".join(re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.DOTALL))
    tokens = set(_BACKTICKED_TOKEN.findall(text)) | set(_UNDERSCORE_TOKEN.findall(fenced))
    unknown = sorted(tokens - documented_vocabulary)
    assert unknown == [], f"{doc_path.name} references identifiers the code does not define: {unknown}"
