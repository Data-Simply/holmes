# Trajectory log schema

The trajectory is an append-only JSON **array**, one object per iteration, written to
`results/trajectory.json` by `holmes holmes-iter`. The mechanical fields (`params`, `seed`,
`k`, `split`, `metrics`, `score`) are filled by the CLI. The reasoning fields (`hypothesis`
before the run; `validation_status` and `interpretation` after) are written by the LLM driving
the loop.

Each iteration fits **one seed**. To judge whether a result is stable across initializations, run
the same `params` again with a different `--seed` and compare — the spread across those runs is the
stability signal (it is not a field on a single entry).

## Entry schema

```jsonc
{
  "iteration": 3,                       // 1-based, assigned by the CLI
  "seed": 0,                            // the seed fit this iteration
  "k": 10,                              // ranking cut-off scored (recorded so a drift is visible)
  "split": "val",                       // held-out split scored
  "params": {                           // the config that was fit
    "factors": 128,
    "regularization": 0.1,
    "iterations": 30,
    "alpha": 10.0
  },
  "hypothesis": {                       // WRITTEN BEFORE THE RUN
    "mechanism": "Raising regularization 10x (0.01->0.1) shrinks mean_factor_norm by ~half and cuts train_test_ndcg_gap from ~0.4 to <0.2.",
    "outcome": "Validation ndcg rises ~15% because less memorization means better generalization.",
    "falsifiers": "If the gap does not narrow, the gap wasn't driven by under-regularization. If the gap narrows but ndcg falls, regularization is now too strong (pattern 5)."
  },
  "metrics": {                          // the diagnostic battery for this fit, filled by the CLI
    "ndcg": 0.041,
    "recall": 0.072,
    "map": 0.038,
    "train_ndcg": 0.21,
    "train_test_ndcg_gap": 0.169,
    "catalog_coverage": 0.34,
    "avg_rec_popularity": 0.71,
    "novelty": 6.2,
    "tail_recall": 0.009,
    "mean_factor_norm": 0.95,
    "train_recon_error": 0.18,
    "fit_time_seconds": 4.31,           // wall time of the ALS fit; noisy run-to-run, not a decision input
    "eval_time_seconds": 0.92           // wall time of the diagnostic pass; same caveat
  },
  "score": 0.041,                       // == metrics.ndcg (the goal metric)

  "validation_status": "validated",     // WRITTEN AFTER: validated | partial_mechanism | coincidence | null
  "interpretation": "Gap fell from 0.41 to 0.17 as predicted and ndcg rose 14% -> validated. The lever is regularization; next, probe whether factors can now go higher without re-opening the gap."
}
```

## Field rules

- **`hypothesis`** must have all three sub-fields and they must be falsifiable. The `mechanism`
  names *which diagnostic metrics move, which direction, roughly how much* — not just the goal.
- **`validation_status`** is exactly one of the four states. Decide it by comparing the
  mechanism prediction and outcome prediction to what actually moved:
    - mechanism moved + outcome followed → `validated`
    - mechanism moved + outcome didn't → `partial_mechanism`
    - outcome moved + mechanism didn't → `coincidence`
    - nothing moved → `null`
- **`interpretation`** records what you learned and seeds the next hypothesis. For non-validated
  results, state the disambiguation plan (see the reasoning guide).
- Never edit `params` or `metrics` after the run — those are the evidence. Only the two
  reasoning fields are filled in post-hoc.
