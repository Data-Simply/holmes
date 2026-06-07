# HOLMES experiment runner.
#
# Wraps the `holmes` CLI so the four strategies can be launched with one command each.
# grid / random / bayes run unattended; holmes runs an interactive Claude session per
# fit-seed x trial. Pure orchestration -- no changes to holmes/ code.
#
# Override any variable on the command line, e.g.:
#   make grid DATA=data/processed/Electronics FIT_SEEDS="0 1 2 3 4"

# --- Variables (override on the command line) ------------------------------
# DATA         preprocessed dataset directory (passed as --data)
# FIT_SEEDS    ALS fit seeds (--seed): the model's own init randomness. Every
#              strategy runs once per fit seed.
# SEARCH_SEEDS optimizer search-trajectory seeds (random --search-seed /
#              bayes --sampler-seed): which configs get tried. random/bayes
#              sweep the full FIT_SEEDS x SEARCH_SEEDS cross product; grid is
#              deterministic given the fit seed and ignores this.
# TRIALS       HOLMES runs per fit seed. The LLM search has no integer seed, so
#              repeated trials are how its search variance is characterized (the
#              HOLMES analog of SEARCH_SEEDS).
# RESULTS_DIR  where result JSON is written
# UV           runner; ensures the project env is used
DATA         ?= data/processed/Books
FIT_SEEDS    ?= 0 1 2
SEARCH_SEEDS ?= 0
TRIALS       ?= 3
RESULTS_DIR  ?= results
UV           ?= uv run

.DEFAULT_GOAL := help
.NOTPARALLEL:  # baselines fit multi-GB ALS models; never run them concurrently
.PHONY: help grid random bayes holmes baselines clean

# --- Help (default) --------------------------------------------------------
help:
	@echo "HOLMES experiment runner"
	@echo
	@echo "Targets:"
	@echo "  grid        Grid search,   per fit seed         -> $(RESULTS_DIR)/grid-seed<N>.json"
	@echo "  random      Random search, per fit x search seed -> $(RESULTS_DIR)/random-seed<N>-search<M>.json"
	@echo "  bayes       Optuna TPE,    per fit x search seed -> $(RESULTS_DIR)/bayes-seed<N>-search<M>.json"
	@echo "  holmes      Interactive session per fit x trial -> $(RESULTS_DIR)/trajectory-seed<N>-trial<T>.json"
	@echo "  baselines   Run grid, random, bayes serially (all seeds)."
	@echo "  clean       Remove $(RESULTS_DIR)/*.json."
	@echo
	@echo "Variables (current values):"
	@echo "  DATA         = $(DATA)"
	@echo "  FIT_SEEDS    = $(FIT_SEEDS)   (ALS --seed; all strategies)"
	@echo "  SEARCH_SEEDS = $(SEARCH_SEEDS)   (random/bayes search trajectory; grid ignores it)"
	@echo "  TRIALS       = $(TRIALS)   (HOLMES runs per fit seed)"
	@echo
	@echo "Note: each fit is a full ALS model (multi-GB at real scale); the baseline"
	@echo "targets fit once per seed, so they are long-running and block until done."

# --- Baselines -------------------------------------------------------------
# grid runs once per fit seed (it is deterministic given the seed). random and bayes sweep the
# full FIT_SEEDS x SEARCH_SEEDS cross product so fit-noise and search variance can be separated.
grid:
	@mkdir -p $(RESULTS_DIR)
	@for fs in $(FIT_SEEDS); do \
		echo ">>> grid   fit-seed=$$fs"; \
		$(UV) holmes grid --data $(DATA) --seed $$fs \
			--out $(RESULTS_DIR)/grid-seed$$fs.json || exit $$?; \
	done

random:
	@mkdir -p $(RESULTS_DIR)
	@for fs in $(FIT_SEEDS); do for ss in $(SEARCH_SEEDS); do \
		echo ">>> random fit-seed=$$fs search-seed=$$ss"; \
		$(UV) holmes random --data $(DATA) --seed $$fs --search-seed $$ss \
			--out $(RESULTS_DIR)/random-seed$$fs-search$$ss.json || exit $$?; \
	done; done

bayes:
	@mkdir -p $(RESULTS_DIR)
	@for fs in $(FIT_SEEDS); do for ss in $(SEARCH_SEEDS); do \
		echo ">>> bayes  fit-seed=$$fs sampler-seed=$$ss"; \
		$(UV) holmes bayes --data $(DATA) --seed $$fs --sampler-seed $$ss \
			--out $(RESULTS_DIR)/bayes-seed$$fs-search$$ss.json || exit $$?; \
	done; done

# --- HOLMES agentic loop ---------------------------------------------------
# The loop needs an LLM between rounds, so each run is an interactive Claude session with a
# pre-filled prompt; Claude drives the loop autonomously per skill/SKILL.md, then returns to the
# REPL (Ctrl-D out to launch the next run). For a fair comparison with the baselines, sweep one
# run per fit seed with TRIALS repeats each -- the repeats stand in for the search seed the LLM
# does not have. Runs are sequential. The prompt avoids backticks/quotes so the shell does not
# interpret it; $(DATA) is make-expanded while $$fs/$$t/$$traj interpolate in the loop's shell.
holmes:
	@mkdir -p $(RESULTS_DIR)
	@for fs in $(FIT_SEEDS); do for t in $$(seq 1 $(TRIALS)); do \
		traj=$(RESULTS_DIR)/trajectory-seed$$fs-trial$$t.json; \
		echo ">>> holmes fit-seed=$$fs trial=$$t -> $$traj"; \
		claude "Run the HOLMES agentic hyperparameter-tuning loop using the holmes-hpo skill (skill/SKILL.md). Dataset --data is $(DATA); the trajectory log is $$traj. Use --seed $$fs for every fit, and prefix every holmes command with '$(UV)'. Start by running $(UV) holmes ranges to read the bounds and the max_iterations budget, seed iteration 1 by running $(UV) holmes heuristic --data $(DATA) --trajectory $$traj --seed $$fs, then run the loop autonomously per SKILL.md until the trajectory reaches max_iterations, and finish with a held-out test eval and a short summary." || exit $$?; \
	done; done

# --- Baselines aggregate ---------------------------------------------------
# The three unattended baselines, run serially. HOLMES is interactive, so it is its own target.
baselines: grid random bayes

# --- Clean -----------------------------------------------------------------
clean:
	rm -f $(RESULTS_DIR)/*.json
