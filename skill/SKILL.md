---
name: holmes-hpo
description: >-
  Drive the HOLMES agentic hyperparameter-tuning loop for the ALS book recommender.
  Use when asked to tune ALS hyperparameters with HOLMES, run the agentic search, or
  beat the grid/Bayesian baselines on a small fit budget. Runs autonomously: forms a
  falsifiable hypothesis, runs one iteration, interprets it, repeats.
---

# HOLMES — Hypothesis-driven Optimization via LLM-guided Model Exploration and Search

You are tuning an implicit-feedback **ALS book recommender** by reasoning over a diagnostic
battery, not by blindly optimizing a single score. Each iteration you write a *falsifiable
hypothesis*, run one fit, and decide whether the hypothesis was validated. The trajectory of
(hypothesis → params → metrics → interpretation) is the deliverable.

Every step is a `holmes` CLI call. The tool is installed on your PATH, so invoke it bare
(`holmes ranges`, `holmes holmes-iter …`) — do not prefix it with `uv run` or anything else.

You are given two paths: the input dataset (the `--data` directory) and the trajectory log (the
`--trajectory` file). Pass *exactly those* to every command — do not hardcode or invent paths. The
literal paths in the examples below (`data/processed/Books`, `trajectory.json`) are placeholders for
the ones you were given. Do not edit the trajectory file by hand — the CLI owns it, and
edits-via-flag are deterministic where free-form edits are not.

## When to use this

- A preprocessed dataset exists at the `--data` directory you were given. Every command requires
  `--data` pointing at it — there is no default.
- A shared fit budget caps the loop — a hard cap shared with the grid and Bayes baselines so
  the three strategies are compared at the same number of ALS fits. Call `holmes ranges` at the
  start of the loop to read the HP bounds, the `max_iterations` budget, and the dataset signal
  you choose iteration 1 from; the CLI enforces the cap (`holmes-iter` refuses to run once the
  trajectory reaches it).
- You want to understand *why* a config is better, not just which one scored highest.

## The hyperparameters

You tune four ALS hyperparameters: `factors` (latent dimensionality), `regularization` (L2
penalty), `iterations` (ALS sweeps), and `alpha` (confidence scaling on positives).

Run `holmes ranges --data <data>` once at the start. It prints, as JSON, the supported bounds
(derived from `GRID_SPACE`, so the agentic loop, grid, and Bayes all optimize over the same
region), the `max_iterations` budget, and a `dataset` block of size/density signal you reason
from to choose iteration 1. Stay within the bounds; out-of-bounds params are rejected by
`holmes-iter`.

## Running a fit

Each `holmes-iter` and `eval` call fits one ALS model (~1–4 min, longer at
`factors=512`). Run every fit as a **background** command and wait for the completion
notification — do not poll. Never use `sleep` / `pgrep` / `pkill` / `kill` / `while`-wait
loops to wait on it: foreground `sleep` is blocked in this harness, those loops get
auto-backgrounded or killed, and the process-control commands trigger permission prompts you
don't need. The completion notification is the signal; check the trajectory afterwards.

Run **one fit at a time**. `holmes-iter` read-modify-appends the trajectory file, so two
concurrent fits race on it and one iteration's entry will be lost.

If a run looks suspiciously long and you want a liveness signal, pass `--progress` to
`holmes holmes-iter` / `holmes eval`: the underlying ALS fit streams a
per-sweep tqdm bar to stderr (e.g. `13/20 [01:42<00:38]`), which you can read out of the
backgrounded command's output file to see sweeps tick and an ETA. Leave it off by default to
keep the trajectory output clean — turn it on when you have reason to suspect a hang.

## The workflow — run this loop autonomously

Do **not** pause to ask permission between iterations. Each printed trajectory entry includes
its iteration number — track it against the `max_iterations` budget returned by `holmes ranges`
to pace yourself. Stop when any of these conditions is met:

- The trajectory reaches `max_iterations` (the CLI enforces this hard — `holmes-iter` will
  refuse to fit once there).
- `ndcg` has plateaued — three consecutive `validated` iterations with `ndcg` moves below
  ~1% relative are diminishing returns; switch to the **Finishing** step.
- Three consecutive iterations are `null` or `coincidence` with no disambiguation plan that has
  moved the needle — you are out of ideas; stop, report the best entry, name what is still
  weak, and ask for direction. Do not burn the rest of the budget chasing noise.
- A setup-integrity condition fires (see below).

### Iteration 1 — choose your own starting point

There is no separate seeding step: **iteration 1 is a normal `holmes-iter` call.** You pick the
starting configuration yourself by reasoning from the `dataset` block `holmes ranges` prints
(`n_users`, `n_items`, `n_interactions`, `density`, `mean_rating`) and the HP bounds — the
starting point is part of your optimizer behavior, not a privileged seed the baselines lack.
Standard implicit-ALS starting points, grounded in that signal:

- `factors` — capacity tracks data volume: low on the searchable range for a small or sparse
  matrix, higher when `n_interactions` is large.
- `regularization` — near the geometric midpoint of its log range, leaving equal multiplicative
  headroom to move up or down with the loop's bold moves.
- `iterations` — near the midpoint of the sweep range, with headroom to raise it if
  `train_recon_error` stays high.
- `alpha` — chosen so the Hu et al. (2008) confidence `c = 1 + alpha * mean_rating` lands near
  ~100 on an average positive.

Run it like any other iteration, with all seven flags. Because there is no prior iteration to
compare against, frame the opening hypothesis in **absolute** terms: the outcome is the *floor*
later iterations must beat, and the falsifiers name absolute diagnostic conditions rather than
deltas.

```bash
holmes holmes-iter --data <data> --trajectory <trajectory> --seed 0 \
  --factors 128 --regularization 0.1 --iterations 22 --alpha 22.0 \
  --mechanism "factors=128 gives capacity for this interaction count; alpha=22 puts Hu et al. \
    confidence c = 1 + alpha*mean_rating near 100; regularization=0.1 sits mid-range." \
  --outcome  "ndcg lands above a popularity baseline but below a tuned upper bound — the floor \
    subsequent iterations must beat." \
  --falsifiers "If train_recon_error stays high, capacity (factors) or iterations underspecified \
    the fit. If train_test_ndcg_gap is large, regularization=0.1 is too low for this matrix. If \
    tail_recall is near zero, alpha over-weights head items."
```

The diagnostic battery is computed on the **validation** split. The entry is recorded with
`validation_status` and `interpretation` set to null for the annotate step below.

### Iterations 2+ — author the hypothesis as flags

For every subsequent iteration, read the latest trajectory entry, decide the next params and
hypothesis, and submit the iteration as a single Bash command:

```bash
holmes holmes-iter --data <data> --trajectory <trajectory> --seed 0 \
  --factors 128 --regularization 0.1 --iterations 20 --alpha 40.0 \
  --mechanism "Raising regularization 10x cuts mean_factor_norm by ~half and closes \
    train_test_ndcg_gap from ~0.4 to <0.2." \
  --outcome  "Validation ndcg rises ~10-15% because less memorization improves generalization." \
  --falsifiers "If the gap does not narrow, the gap wasn't driven by under-regularization. \
    If the gap narrows but ndcg falls, regularization is now too strong (pattern 5)."
```

All seven flags are required. The hypothesis-before-results discipline is enforced this way:
the hypothesis is recorded *as part of the command that runs the fit*, not after seeing the
metrics.

A hypothesis is a causal chain: *"this HP change improves ndcg **because** it shifts
[intermediate diagnostic]."* Always name the mechanism, not just the outcome.

#### Hypothesis rubric

Before running, your three fields must each clear a concrete bar — not all hypotheses are
falsifiable, and a vague one wastes the iteration:

- **`--mechanism`** names a *specific* diagnostic metric from the battery (not "the model"),
  the direction it moves, and a rough magnitude (e.g. "`train_test_ndcg_gap` falls from ~0.4
  to <0.2"). "Generalization improves" is not a mechanism.
- **`--outcome`** ties `ndcg` to the mechanism causally (*"ndcg rises ~10% **because** the gap
  closed"*) — it is not a separate prediction. If the outcome doesn't reference the mechanism,
  the hypothesis can't be `coincidence`-disambiguated later.
- **`--falsifiers`** (i) names the diagnostic that would refute the mechanism, (ii) gives a
  directional condition ("if the gap does not fall below 0.3..."), and (iii) rules out at
  least one alternative explanation ("...or if it falls but ndcg also falls, regularization is
  too strong — pattern 5"). A falsifier of the form "if nothing moves" is too weak; it leaves
  every non-validated outcome ambiguous.

#### Self-check before running

Re-read your three fields together as one paragraph. If the falsifier *couldn't realistically
come true* given how that diagnostic actually responds (per `references/REASONING_GUIDE.md`),
the falsifier is decorative — rewrite it. If the mechanism could be true even when the named
diagnostic doesn't move (i.e., you're hedging), make it sharper. Only submit the
`holmes-iter` command once this check passes.

### Interpret AFTER running — annotate via the CLI

`holmes holmes-iter` echoes the just-appended trajectory entry to stdout as JSON — there is no
second command to read it. Compare the
observed metric moves to your mechanism and outcome predictions. Classify the result into
exactly one of four states, then record both via:

```bash
holmes annotate --trajectory <trajectory> --iteration N \
  --status validated \
  --interpretation "Gap fell 0.41 -> 0.17 as predicted and ndcg rose 14% -> validated. \
    The lever is regularization; next, probe whether factors can now go higher without \
    re-opening the gap."
```

The four states:

- **validated** — mechanism shifted AND outcome followed. Continue this direction or move to a
  different weakness.
- **partial_mechanism** — mechanism shifted, outcome did not. You found the lever but not how
  it connects to ndcg. This is the next puzzle.
- **coincidence** — outcome moved WITHOUT the mechanism shifting. Could be a secondary effect,
  seed noise, or a bug. Next iteration disambiguates.
- **null** — nothing moved. The move was too timid or aimed at the wrong lever.

Then feed the result into the next hypothesis and run the next `holmes holmes-iter` call. A
non-validated result is not a stopping point — it is the next hypothesis. For a `coincidence`,
design a test that separates the boring explanation (seed noise → re-run; secondary effect →
change an HP that should move outcome but not mechanism). For a `partial_mechanism`, revise
the causal model (threshold? interaction with another HP? different pathway?) and test it.

## Principles

- **Falsifiable predictions.** Mechanism + outcome + falsifier on every hypothesis.
- **One or two HPs per move.** Exception: when disambiguating a coincidence, deliberately
  change two HPs aimed at different metrics to separate their effects.
- **Bold moves.** Prefer 10× changes to learn the response surface; 1.5× changes mostly
  produce noise. Refine only once near an optimum.
- **Check stability by re-running seeds.** Each run fits one seed. When a result is surprising or
  about to drive a decision, re-run the same params with another `--seed`: if ndcg swings a lot,
  the config is fragile (treat like a `coincidence` — don't extend on it).
- **Honest tradeoffs.** The final recommendation names what is still weak, not just a winner.

## Setup-integrity stop conditions — the ONE case that halts the loop

Stop and surface to the user if the diagnostics reveal the experiment itself is broken, not
the model:

- re-running identical params with the same `--seed` gives a different ndcg (the fit isn't
  reproducible → iterations aren't comparable).
- `train_recon_error` does not fall as `iterations` rises (training not converging in budget).
- Validation metrics implausibly match training memorization (possible leakage).

When detected, say clearly: *"Stopping early — iteration N isn't comparable to 1..N-1 because
[X]. Recommend [fix], then restart."* Do not burn budget on uninterpretable results.

## Reference

Read `references/REASONING_GUIDE.md` for the metric-pattern → hypothesis → HP-move playbook,
and `references/TRAJECTORY_SCHEMA.md` for the exact log schema.

## Finishing

When the budget is spent, ndcg has plateaued, or you have exhausted plausible hypotheses, run
the winner on the held-out **test** split once for an unbiased number:

```bash
holmes eval --data <data> --params '{"factors": 96, "regularization": 0.1, "iterations": 20, "alpha": 40.0}' --split test
```

Then write a short summary: best config, the ndcg it reached, the trajectory of hypotheses
that got there, and what remains weak (e.g. tail_recall still low).
