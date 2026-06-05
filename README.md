# HOLMES

**HOLMES — Hypothesis-driven Optimization via LLM-guided Model Exploration and Search**

HOLMES benchmarks hyperparameter optimization of an implicit-feedback **ALS recommender** on the
Amazon Reviews 2023 dataset — any category (Books is the default; preprocess one or `--all` of them
side by side) — comparing three strategies on the same fit budget:

- **grid** — exhaustive grid search
- **bayes** — Bayesian optimization (Optuna TPE)
- **holmes** — an agentic loop where an LLM reasons over a diagnostic battery, forms a
  falsifiable hypothesis each round, and proposes the next hyperparameters as a test of it

The shared `ALSRecommender`, diagnostic battery, and evaluation harness mean every strategy
optimizes the identical objective (held-out NDCG@10) on the identical splits.

## Why implicit ALS (and not explicit) for rating data?

There are two ALS algorithms in the literature, and we deliberately use the implicit variant
despite the data being rating-shaped:

- **Zhou et al. 2008 (Explicit ALS)** — for rating prediction. Loss is
  `Σ_observed (r_ui − x_u·y_i)² + λ(...)`. Only observed entries enter the loss; unobserveds are
  "missing." This is the Netflix-Prize-style algorithm.
- **Hu et al. 2008 (Implicit ALS)** — for engagement signals. Loss is
  `Σ_all c_ui (p_ui − x_u·y_i)² + λ(...)`. *All* user-item pairs contribute, with observed =
  high-confidence positive and unobserved = low-confidence zero. This is what
  `implicit.als.AlternatingLeastSquares` implements.

For Amazon Books, implicit ALS is the right choice, despite the data being rating-shaped:

- Most users haven't reviewed most books — the unobserveds are *missing not at random* (didn't
  encounter), not "disliked." Explicit ALS would silently treat them as missing and only train on
  observed entries, throwing away the strongest signal in the dataset.
- The task is top-K ranking (NDCG@10 on a held-out next item), not rating prediction. Ranking is
  exactly what implicit ALS optimizes.
- Hu et al.'s `c_ui = 1 + α·r_ui` was *designed* to consume a continuous engagement signal — they
  used TV watch-time; we use the review's star rating (stored as the matrix value rather than a
  binary 1, so a 5-star carries ~25% more confidence than a 4-star). Using the rating is the
  framework's intended use, not a hack.

If the goal were to *predict the rating* a user would give a book, Zhou-style explicit ALS (or
SVD++) would be right. But for "what should we recommend?", implicit ALS with rating-weighted
confidence is the principled choice.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
uv sync
```

## Workflow

```bash
# 1. Build the interaction matrix from any Amazon Reviews 2023 category. The first run downloads the
#    gzipped reviews (~20GB uncompressed for Books) from the McAuley mirror and caches the columns it
#    needs as data/raw_cache/<category>.parquet; later runs reuse that cache and skip the download.
#    Polars then streams the dedup, k-core filter, and leave-last-out split into
#    data/processed/<category>/ — so each category is a separate, side-by-side dataset.
uv run holmes preprocess                                   # Books (default), full dataset
uv run holmes preprocess --category Electronics            # any category by name
uv run holmes preprocess --category Video_Games --max-interactions 2000000  # cap rows for a quick dev matrix
uv run holmes preprocess --all                             # every category into data/processed/<category>

# `--data` is required on every command below (no default), so a run can never silently pick up the
# wrong category's matrix. Point it at the category subdirectory you preprocessed.
#
# Each run fits ONE seed. For stability across initializations, repeat a run with different
# --seed values and aggregate the results yourself.

# 2a. Grid-search baseline.
uv run holmes grid --data data/processed/Books --seed 0

# 2b. Bayesian-optimization baseline (--seed is the per-fit seed; --sampler-seed is the TPE
#     search trajectory).
uv run holmes bayes --data data/processed/Books --trials 30 --seed 0 --sampler-seed 0

# 2c. The agentic HOLMES loop is driven by Claude Code via skill/SKILL.md. Each round runs
#     ONE iteration, appending diagnostics to an append-only trajectory log:
uv run holmes heuristic --data data/processed/Books          # suggested starting params
uv run holmes holmes-iter --data data/processed/Books \
  --input iter.json --trajectory results/trajectory.json --seed 0

# 3. Score a chosen config on the held-out test split for an unbiased number.
uv run holmes eval --data data/processed/Books --params '{"factors": 96, "regularization": 0.05, "iterations": 30, "alpha": 20.0}' --split test
```

## The agentic loop

The HOLMES strategy is not fully scripted — it is run by an LLM (Claude Code) reading the
trajectory between iterations. The wiring lives in `skill/`:

- `skill/SKILL.md` — the autonomous loop: hypothesize → run one iteration → interpret → repeat.
- `skill/REASONING_GUIDE.md` — the metric-pattern → hypothesis → HP-move playbook.
- `skill/TRAJECTORY_SCHEMA.md` — the append-only log schema (params + mechanism + outcome +
  falsifiers + metrics + validation status + interpretation).

## Development

```bash
uv run pytest                  # run the test suite
uv run pytest --cov=holmes     # with coverage
uv run ruff format . && uv run ruff check .
```

## Project layout

```
holmes/
  config.py            # hyperparameter spaces, evaluation settings, ALSParams
  data/
    preprocess.py      # Amazon Reviews 2023 (Books) -> sparse matrix + leave-last-out splits
    dataset.py         # on-disk Dataset container
  als/model.py         # ALSRecommender (implicit wrapper) shared by every strategy
  metrics/diagnostics.py  # the diagnostic battery
  search/
    harness.py         # fit one config (one seed), compute the diagnostic battery
    heuristics.py      # initial params from dataset characteristics
    grid.py            # grid search
    bayes.py           # Optuna Bayesian optimization
    holmes.py          # run ONE HOLMES iteration, append to trajectory
  cli.py               # `holmes preprocess|grid|bayes|holmes-iter|heuristic|eval`
skill/                 # SKILL.md, reasoning guide, trajectory schema
tests/                 # test suite
existing_literature/   # reference material
```
