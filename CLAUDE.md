# HOLMES Development Guide

This project follows the global development guide (Google style, red/green TDD, the test-writing
guidelines, `uv` for everything). The notes below are **additions and project-specifics** — they do
not repeat the global guide.

## What this project is

HOLMES ("Hypothesis-driven Optimization via LLM-guided Model Exploration and Search") benchmarks
hyperparameter optimization of an implicit-feedback **ALS book recommender** (Amazon Reviews 2023,
Books) across four strategies — grid search, random search, Bayesian optimization (Optuna), and the
agentic HOLMES loop. Every strategy optimizes the same objective (held-out NDCG@10) on the same
splits via the shared `ALSRecommender`, diagnostic battery, and evaluation harness.

Package layout: `holmes/{config,cli}.py`, `holmes/data/`, `holmes/als/`, `holmes/metrics/`,
`holmes/search/`. The agentic loop is driven by an LLM via `skill/SKILL.md`,
`skill/references/REASONING_GUIDE.md`, and `skill/references/TRAJECTORY_SCHEMA.md`. See
`README.md` for the CLI workflow.

## Data scale

Write for the real sizes, not the test fixtures (~hundreds of users). Books ≈ **30M+ raw rows** →
**millions of users**, **100k+ items**, `train_ui` `nnz` in the tens of millions; one ALS model is
multi-GB and grid search fits dozens of them (`MAX_ITERATIONS` in `holmes/config.py` is the
authoritative count, shared across grid/bayes/HOLMES). So: never treat O(n_users)/O(nnz) work as
free, keep the
Polars ingestion lazy until indices are assigned, cache matrix-wide arrays on `Dataset`, and sample
(not materialize) in diagnostics. A green test on the fixture does not prove it scales.

## Build & Test Commands

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run a specific test: `uv run pytest -k "test_name"` (don't guess full node paths)
- Lint and format: `uv run ruff format . && uv run ruff check .`
- Run the CLI: `uv run holmes <preprocess|grid|random|bayes|holmes-iter|heuristic|eval> ...`
- Always use `uv run` so the correct environment and dependencies are used. Never call `pip`.

## Architecture & Design Principles

### Simplicity Criterion

All else being equal, simpler is better. Weigh every change as **improvement magnitude vs. complexity
cost**: a marginal gain that drags in a new abstraction, an extra parameter, or another layer of
indirection is rarely worth it. **Removing code while preserving or improving behavior is one of the
best outcomes available** — a net-negative diff is a feature, not a sign of cut corners.

Apply this lens to every change:

- **Prefer deletion over addition** when both achieve the goal. If a requirement can be met by removing
  a code path rather than guarding it, do that.
- **Reject sunk-cost momentum.** If a refactor is getting uglier as you go — more special cases, more
  wrappers, more comments needed to justify each step — stop and reconsider. The right answer is often a
  smaller change, or a different shape of change entirely.
- **No premature abstraction.** A helper for one caller has to be read every time too. Three similar
  lines beat a wrapper introduced "just in case" a second use site appears.
- **Default to the boring shape.** Flat functions beat dispatch tables for two cases; direct checks beat
  extension points; explicit code beats clever code.

When in doubt about whether a piece of complexity is earning its keep, leave it out. Adding it back when
a real second use case demands it is cheap; removing an abstraction once code depends on its shape is not.

### Comparability is fair by design

"Fair by design" means the invariants that make the four strategies comparable should be locked by
construction and by tests, so a future edit can't silently break the comparison. The benchmark is only
meaningful if grid, random, bayes, and HOLMES differ in **optimizer behavior alone** — same objective,
same held-out split, same ranking cut-off, same search region, and same fit budget. Enforce this
structurally first: every strategy scores through the shared `evaluate_config`; all continuous spaces
derive from `_grid_hull(GRID_SPACE)`; the budget is the single `MAX_ITERATIONS` with no per-call
override. Where a property can only hold by convention — e.g. `RANDOM_SPACE`/`BAYES_SPACE`/`HOLMES_SPACE`
are deliberately distinct objects so a test `monkeypatch` on one doesn't bind the others — back it with a
guardrail test (`TestComparabilityInvariants`) so drift fails loudly instead of quietly biasing results.

## Code Style Additions

- **No nested function definitions.** A `def` (or `async def`) must not appear inside another function's
  body. Lift inline helpers to module scope. If the helper genuinely needs outer-scope state, take that
  state as an explicit parameter (e.g. via `functools.partial`) rather than relying on a closure. Lambdas
  are exempt; the rule is specifically about `def` / `async def` inside another `def`. (Ruff has no
  built-in rule for this; enforce by review.)
- **Determinism is a correctness property — guarantee it as much as possible.** Identical inputs must give
  identical outputs (the loop compares iterations). Seed every RNG, give outcome-feeding sorts a full
  tiebreaker, and don't depend on hash/`group_by`/unstable-sort ordering. BLAS is pinned to one thread for
  this reason.

## Test Writing Additions

These refine the global testing guidelines:

- **Pin counts to their source rather than asserting existence.** `assert len(x) > 0` / `>= 1` is a weak
  check when the expected count is knowable — a regression dropping all but one item would still pass.
  Prefer `assert len(recs) == len(users)` or `assert dataset.n_items == N_GROUPS * BOOKS_PER_GROUP`.
- **Avoid `check_dtype=False`** in `assert_frame_equal` / `assert_allclose` — match the expected dtype
  instead; loosening it lazily hides real dtype regressions.
- **Network is an external dependency.** Test the network-free helpers directly (e.g. the preprocessing
  `_k_core_filter` / `_leave_last_out_split`); do not hit Hugging Face in the test suite.
- **Lock comparability invariants with guardrail tests.** A property that keeps the strategies fair but
  holds only by convention (equal search spaces, the shared `MAX_ITERATIONS` budget, the common
  hyperparameter set) must have a test that fails on drift — see `TestComparabilityInvariants` and the
  *Comparability is fair by design* principle above. Derive the expected value from its source
  (`_grid_hull(GRID_SPACE)`, `dataclasses.fields(ALSParams)`), never a hand-copied literal that can drift.
