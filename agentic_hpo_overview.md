# Agentic hyperparameter tuning

### Agentic hyperparameter tuning: the method

The core idea is to replace grid, random, or Bayesian search with a small budget of iterations driven by an LLM's reasoning. Instead of optimizing a single scalar score, you compute a battery of diagnostic metrics each round; the LLM forms a hypothesis about what the model is doing wrong, and proposes the next HPs as a falsifiable test of that hypothesis. The trajectory of (hypothesis → params → metrics → interpretation) becomes the search log, and crucially, becomes a record of causal claims that were tested — not just configurations that scored well.

It works well when three conditions hold: training is expensive enough that you can't afford hundreds of fits, the model has enough HPs that grid search blows up combinatorially, and you have richer signals than a single fitness score. The richness is what makes the reasoning non-trivial. If all you have is validation accuracy, an LLM can't do better than Bayesian optimization. If you have a battery of metrics that point at different failure modes, the LLM's job is to read which failure mode is currently dominant and pick the HP move that addresses it.

### The pieces

**A diagnostic battery, not a single score.** The most important design decision. Each metric should illuminate a different axis of model behavior. For your model you need at least 4-8 metrics where you can articulate what "bad" looks like on each and what HP would plausibly fix it. If two metrics always move together, drop one. If a metric never moves across reasonable HP changes, it isn't informative.

**A reasoning guide.** A markdown document mapping metric patterns to hypotheses and concrete HP moves, in the form "Pattern: [signals] → Hypothesis: [what's happening internally] → Move: [specific HP change, with magnitude]". Six to ten patterns covering the realistic failure modes is enough. This is where your ML expertise gets encoded — the LLM isn't deriving these from first principles, it's applying them.

**Heuristic initial params.** A function from dataset or problem characteristics to a sensible starting point. Saves an iteration that would otherwise just confirm the defaults are reasonable.

**An append-only trajectory log.** JSON file with one entry per iteration: params, hypothesis written before the run, metrics, interpretation written after. The hypothesis-before-results discipline is what prevents post-hoc rationalization.

**A SKILL.md or system prompt** wiring it together: when to use it, the workflow, principles, and instructions to run the loop autonomously rather than asking permission between rounds.

### The hypothesis is a causal chain, not a prediction

This is the part that matters most for getting real learning out of the loop. A useful hypothesis isn't "this HP change will improve the goal metric." It's *"this HP change will improve the goal metric **because** it shifts [some intermediate property]."* The intermediate property is the mechanism, and it's doing causal work. If you only check the outcome, you're confirming the conclusion of a syllogism while ignoring the premise — and you'll happily accept any path that produced the outcome, including paths that are bugs, coincidences, or unrelated secondary effects.

So every hypothesis is written before the run with three explicit parts: (1) **mechanism prediction** — which diagnostic metrics will move, in which direction, by roughly how much; (2) **outcome prediction** — which goal metric will move as a consequence; (3) **falsifiers** — what observations would say "my causal model is wrong," typically the mechanism not moving or moving the wrong way.

After the run, every iteration ends in one of four states:

- **Validated** — mechanism shifted and outcome followed as predicted. Continue the direction or move on to address a different weakness.
- **Partial mechanism** — mechanism shifted, outcome didn't follow. You understood the lever but not how it connects to the goal. Treat this as the new puzzle for the next iteration.
- **Coincidence** — outcome moved without the mechanism shifting. Could be a secondary effect, could be a bug. The next iteration's job is to disambiguate.
- **Null** — nothing moved. HP move was too small or aimed at the wrong lever; pick differently or move more boldly.

Only `validated` justifies confidently extending in the current direction. The middle two are precisely the cases a coarse "did the goal metric go up?" check misses, and they're the ones that matter most for learning.

### Coincidence and partial-mechanism feed back into the loop

This is the key behavioral rule: a non-validated result is not a stopping condition, it's a new hypothesis to test. The loop doesn't pause to ask the user what to do; it forms a hypothesis about *how to get a more certain answer* and continues.

For a `coincidence` result, the next iteration's hypothesis is targeted at disambiguating between the boring explanations and the real ones. Concretely: re-run the same config to check whether the outcome shift was just seed noise; or change a single HP in a way that should affect the *outcome* but not the *hypothesized mechanism*, to see if outcome moves independently; or perturb the hypothesized mechanism directly through a different lever to check whether outcome follows. Each of these is a normal iteration with a normal hypothesis — it just happens to be aimed at resolving uncertainty rather than improving the model.

For a `partial mechanism` result, the next hypothesis revises the causal model: maybe the mechanism affects the outcome only above some threshold, or in combination with another factor, or via a different pathway than originally thought. Propose a test that distinguishes between the revised explanations.

The one case that *does* stop the loop is when the diagnostic itself reveals a fundamental problem with the experimental setup — seeds not being fixed across runs so iterations aren't apples-to-apples, evaluation data leaking between train and test, a metric being silently computed on stale or wrong inputs, the training not actually converging in the iteration budget. These aren't model-fitting questions, they're setup integrity questions, and continuing to iterate without telling the user wastes their budget on uninterpretable results. When detected, surface clearly: "Stopping early — iteration 4 results aren't comparable to iterations 1-3 because [X]. Recommend [fix], then restart."

### Principles that make it work

*Falsifiable predictions.* The mechanism + outcome + falsifier structure forces every hypothesis to make claims that could turn out wrong.

*One or two HPs per move.* If you change three things and metrics shift, you don't know which mattered. The exception: when disambiguating a coincidence, you might deliberately change two HPs that should affect different metrics, precisely to separate their effects.

*Bold moves.* 10× changes teach you about the response surface; 1.5× changes mostly produce noise. The exception is once you're near an optimum and refining.

*Multiple seeds.* You need stability metrics, which means 2-5 fits per iteration. This is the budget cost — typical AI-search runs are 5-15 iterations × 3 seeds = 15-45 fits, vs. hundreds for grid search.

*Honest tradeoffs.* The final recommendation names what's still weak, not just declares a winner.

### What to port to your problem

Keep: the trajectory schema (params + mechanism prediction + outcome prediction + falsifiers + metrics + validation status + interpretation), the run-one-iteration script pattern, the reasoning-guide-as-reference-document, the autonomous loop structure with the four-outcomes framework, the setup-integrity stop conditions, the initial-params heuristic function.

Replace: the diagnostic battery, the reasoning guide, the heuristic initial params, the training wrapper.

The questions worth answering before you start:

1. What are 4-8 metrics that illuminate *different* axes of your model's behavior? Be specific about what each tells you that the others don't.
2. For each, what does "bad" look like and what HP would plausibly fix it?
3. Which metrics are mechanisms (intermediate, diagnostic) and which are outcomes (what you actually care about)? Some metrics will play both roles depending on the hypothesis.
4. What are the 6-10 most common failure patterns, as "if metrics look like X, the model is doing Y"?
5. What does a sensible default config look like as a function of dataset or problem characteristics?
6. What are the setup-integrity failure modes for your training pipeline — the things that, if they happened, would make iterations uncomparable?

If you can answer those, the skill is mostly a port of the structure with your domain knowledge plugged in.
