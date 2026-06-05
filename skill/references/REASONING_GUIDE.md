# HOLMES reasoning guide — ALS diagnostic patterns

Map the diagnostic battery to a hypothesis about what the model is doing wrong, then to a
concrete HP move with a magnitude. Each pattern is `Signals → Hypothesis → Move`. The metrics
are computed on the **validation** split from a single seed per run. *Stability* is not a stored
metric: assess it by re-running the same `params` with a different `--seed` and comparing `ndcg`.

## The diagnostic battery (what each axis tells you)

| Metric | Axis | "Bad" looks like |
|--------|------|------------------|
| `ndcg`, `recall`, `map` | **outcome** — ranking quality | low |
| `train_ndcg` | memorization of training history | — (read with the gap) |
| `train_test_ndcg_gap` | **overfitting** | large positive |
| `catalog_coverage` | **diversity** — share of catalog recommended | low |
| `avg_rec_popularity` | **popularity bias** — mean popularity percentile of recs | high (→1) |
| `novelty` | long-tail exposure (self-information) | low |
| `tail_recall` | **tail serving** — recall on unpopular held-out items | low / 0 |
| `mean_factor_norm` | **embedding magnitude** (regularization) | very high (under-regularized) or near 0 (collapsed) |
| `train_recon_error` | **convergence / fit** on observed positives | high or seed-unstable |

> `mean_factor_norm` is the *geometric mean* of the user and item factor norms. Don't reason about
> user vs. item norms separately — matrix factorization has a scaling gauge (`x_u . y_i` is unchanged
> if you scale users up and items down), so only their product is meaningful. `mean_factor_norm`
> decreases monotonically as `regularization` rises.

`factors` and `iterations` mostly raise capacity/fit; `regularization` and `alpha` mostly
shape *how* that capacity is spent (generalization vs. memorization, head vs. tail).

## Patterns

### 1. Underfitting (not enough capacity)
**Signals:** low `ndcg`, high `train_recon_error`, small `train_test_ndcg_gap`.
**Hypothesis:** the factorization cannot represent the structure — too few factors or too few
iterations; or regularization is suppressing the fit.
**Move:** double `factors` (e.g. 64→128→256, up to the upper bound from `holmes ranges`). If
`train_recon_error` stays high, the solve isn't converging — raise `iterations` (15→30). One
lever at a time.

### 2. Overfitting / memorization
**Signals:** high `train_ndcg`, **large** `train_test_ndcg_gap`, low/declining `ndcg`, high
`mean_factor_norm`.
**Hypothesis:** the model memorizes training history instead of generalizing — capacity too
high relative to regularization.
**Move:** raise `regularization` 10× (0.01→0.1). If the gap persists, halve `factors`.

### 3. Popularity bias
**Signals:** decent `ndcg` but low `catalog_coverage`, high `avg_rec_popularity` (→1), low
`novelty`, low `tail_recall`.
**Hypothesis:** confidence weighting concentrates the recommender on head items — `alpha` is
over-weighting heavily-interacted (popular) entries.
**Move:** lower `alpha` 4× (40→10). If coverage is still low, raise `factors` to give niche
structure room. Watch that `ndcg` doesn't collapse — the tradeoff is the point.

### 4. Not converged
**Signals:** high `train_recon_error` that **falls** when `iterations` is raised; possibly
seed-sensitive (`ndcg` varies when you re-run with a different `--seed`).
**Hypothesis:** the alternating solve hasn't reached a fixed point within the iteration budget.
**Move:** raise `iterations` (15→30, the upper bound). Once `train_recon_error` plateaus, stop adding sweeps; if it's still falling at 30, the budget itself is the bottleneck — flag it rather than try to escape the bound.

### 5. Over-regularized collapse
**Signals:** low `ndcg`, very low `mean_factor_norm`, high `train_recon_error`, low `train_ndcg`,
low `catalog_coverage`.
**Hypothesis:** the L2 penalty (or too-low `alpha`) has shrunk embeddings toward zero, so all
users/items look alike.
**Move:** lower `regularization` 10× (1.0→0.1→0.01). If still flat, raise `alpha` (1→10).

### 6. Exploding factors / instability
**Signals:** high `mean_factor_norm` AND seed-sensitive `ndcg` (it varies a lot when you re-run the
same `params` with a different `--seed`).
**Hypothesis:** too little regularization lets embeddings grow unconstrained; the optimum is
sharp and seed-dependent.
**Move:** raise `regularization` 10×. Re-check by re-running seeds — the spread should shrink.

### 7. Tail starvation
**Signals:** acceptable `ndcg` but `tail_recall` near 0 and `avg_rec_popularity` high.
**Hypothesis:** the model serves the head well and ignores the tail; the goal metric hides this
because most held-out items are themselves popular.
**Move:** lower `alpha` (reduces head dominance) and/or raise `factors` (room for niche taste).
This is a deliberate two-HP move aimed at different metrics — acceptable here.

### 8. Near optimum (diminishing returns)
**Signals:** `ndcg` flat across a `factors` doubling, `train_test_ndcg_gap` moderate and stable.
**Hypothesis:** you are near the response-surface peak; bold capacity moves no longer pay.
**Move:** switch to refinement — small `regularization` / `alpha` adjustments (±2×), not 10×.

### 9. Seed instability as a setup-integrity flag
**Signals:** `ndcg` varies a lot when you re-run the same `params` with *different* `--seed`s,
persisting even at higher `regularization` and `iterations`.
**Hypothesis:** the model is on a knife-edge — OR (worse) the fit isn't even reproducible.
**Move:** first rule out a broken setup — re-run with the *same* `--seed`; if `ndcg` differs at all,
the harness isn't reproducible → **stop and surface to the user**. If same-seed reproduces but
different-seeds vary widely, it's genuine model sharpness (treat via pattern 6).

## Coincidence / partial-mechanism disambiguation

- **Coincidence** (ndcg moved, predicted mechanism didn't): re-run the identical config to test
  seed noise; or change one HP that should move `ndcg` but not the hypothesized mechanism and
  see whether `ndcg` still moves independently.
- **Partial mechanism** (mechanism moved, ndcg didn't): the lever works but the link to the goal
  is conditional. Test a threshold ("does it only help above factors=128?") or an interaction
  ("does lowering alpha only help once regularization is also lower?").
