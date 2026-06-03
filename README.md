# HOLMES

**HOLMES — Hypothesis-driven Optimization via LLM-guided Model Exploration and Search**

Agentic Hyperparameterization with an Agent in the Loop.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync
```

## Development

```bash
# Run the entry point
uv run agentic-hpo-in-the-loop

# Run the test suite
uv run pytest

# Run tests with coverage
uv run pytest --cov=agentic_hpo_in_the_loop

# Lint and format
uv run ruff format .
uv run ruff check .
```

## Project layout

```
src/agentic_hpo_in_the_loop/   # package source
tests/                         # test suite
existing_literature/           # reference material
```
