# HOLMES

HOLMES — **H**ypothesis-driven **O**ptimization via **L**LM-guided **M**odel **E**xploration and **S**earch

HOLMES benchmarks hyperparameter optimization of an implicit-feedback **ALS recommender** on the
Amazon Reviews 2023 dataset — any category (Books is the default; preprocess one or `--all` of them
side by side) — comparing four strategies on the same fit budget:

- **grid** — exhaustive grid search
- **random** — random search (i.i.d. samples over the same space)
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

For Amazon reviews (Books or any other category), implicit ALS is the right choice, despite the
data being rating-shaped:

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

## Datasets

The framework runs on any Amazon Reviews 2023 category (`preprocess --all` builds all 23), but the
**reported experiments use a fixed, pre-registered subset of 7** — broad enough for the optimizer
comparison to generalize, small enough to run at a sane compute budget.

**Selection criteria**, fixed *before* running any optimizer (so the choice can't be biased by
results): stratify across **two scale tiers** (~tens-of-thousands to ~1M interactions, and ~1M to
~10M) and **two domains** — *media* (high repeat engagement) vs. *physical goods*. The result spans
~2.5 orders of magnitude in size and the full density range.

| category | interactions | users | items | density | int/user | domain | scale tier |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Arts_Crafts_and_Sewing | 25,511 | 3,338 | 3,441 | 0.222% | 7.6 | goods | small |
| Video_Games | 406,724 | 63,413 | 19,022 | 0.034% | 6.4 | media | small–mid |
| Baby_Products | 615,003 | 101,429 | 27,702 | 0.022% | 6.1 | goods | medium |
| CDs_and_Vinyl | 1,019,946 | 101,986 | 74,645 | 0.013% | 10.0 | media | medium |
| Automotive | 3,397,370 | 457,844 | 202,814 | 0.004% | 7.4 | goods | large |
| Books | 5,903,332 | 603,422 | 395,385 | 0.002% | 9.8 | media | large |
| Electronics | 8,260,845 | 1,145,516 | 284,008 | 0.003% | 7.2 | goods | very large |

(Stats are post-preprocessing: after dedup, a 5-core filter on users and items, and the
leave-last-out split. `int/user` = interactions per user, the repeat-engagement signal that
separates media from goods.)

Two categories are **deliberately excluded**: **Gift_Cards** (123 items — top-K ranking is trivial;
retained only as a test fixture) and **Kindle_Store** (a redundant very-large media set already
represented by Electronics at that scale tier).

All reported results are generated on a **single instance type**, so every strategy scores each
config on an identical objective — the comparability the benchmark enforces by construction (the
shared `evaluate_config` and `TestComparabilityInvariants`).

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

# Point --data at the category subdirectory you preprocessed; it's required on every command below.
#
# Each run fits ONE seed. For stability across initializations, repeat a run with different
# --seed values and aggregate the results yourself.

# 2a. Grid-search baseline.
uv run holmes grid --data data/processed/Books --seed 0

# 2b. Random-search baseline (--seed is the per-fit seed; --search-seed draws the configs).
#     Like grid, it always runs the shared fit budget; only the trajectory seed is configurable.
uv run holmes random --data data/processed/Books --seed 0 --search-seed 0

# 2c. Bayesian-optimization baseline (--seed is the per-fit seed; --sampler-seed is the TPE
#     search trajectory). Like grid and random, it always runs the shared fit budget.
uv run holmes bayes --data data/processed/Books --seed 0 --sampler-seed 0

# 2d. The agentic HOLMES loop is driven by Claude Code via skill/SKILL.md. Each round runs
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

```text
holmes/
  config.py            # hyperparameter spaces, evaluation settings, ALSParams
  data/
    preprocess.py      # any Amazon Reviews 2023 category -> sparse matrix + leave-last-out splits
    dataset.py         # on-disk Dataset container
  als/model.py         # ALSRecommender (implicit wrapper) shared by every strategy
  metrics/diagnostics.py  # the diagnostic battery
  search/
    harness.py         # fit one config (one seed), compute the diagnostic battery
    heuristics.py      # initial params from dataset characteristics
    grid.py            # grid search
    random_search.py   # random search
    bayes.py           # Optuna Bayesian optimization
    holmes.py          # run ONE HOLMES iteration, append to trajectory
  cli.py               # `holmes preprocess|grid|random|bayes|holmes-iter|heuristic|eval`
skill/                 # SKILL.md, reasoning guide, trajectory schema
tests/                 # test suite
existing_literature/   # reference material
```
