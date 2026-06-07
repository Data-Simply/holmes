# HOLMES experiment runner.
#
# Wraps the `holmes` CLI so the four strategies can be launched with one command each.
# grid / random / bayes run unattended; holmes runs an interactive Claude session per
# fit-seed x trial. Every target loops over CATEGORIES and skips any run whose result file
# already exists, so an interrupted run resumes without redoing completed trials. Pure
# orchestration -- no changes to holmes/ code.
#
# Override any variable on the command line, e.g.:
#   make grid CATEGORIES=Books FIT_SEEDS="0 1 2 3 4"

# --- Variables (override on the command line) ------------------------------
# PROCESSED_DIR parent of the per-category preprocessed datasets
# CATEGORIES    categories to run; defaults to every preprocessed dataset found
#               under PROCESSED_DIR. Each is fed as --data PROCESSED_DIR/<cat>.
# FIT_SEEDS     ALS fit seeds (--seed): the model's own init randomness. Every
#               strategy runs once per fit seed.
# SEARCH_SEEDS  optimizer search-trajectory seeds (random --search-seed /
#               bayes --sampler-seed): which configs get tried. random/bayes
#               sweep the full FIT_SEEDS x SEARCH_SEEDS cross product; grid is
#               deterministic given the fit seed and ignores this.
# TRIALS        HOLMES runs per fit seed. The LLM search has no integer seed, so
#               repeated trials are how its search variance is characterized (the
#               HOLMES analog of SEARCH_SEEDS).
# RESULTS_DIR   where result JSON is written (namespaced per category)
# UV            runner; ensures the project env is used
PROCESSED_DIR ?= data/processed
CATEGORIES    ?= $(notdir $(patsubst %/,%,$(wildcard $(PROCESSED_DIR)/*/)))
FIT_SEEDS     ?= 0 1 2
SEARCH_SEEDS  ?= 0
TRIALS        ?= 3
RESULTS_DIR   ?= results
UV            ?= uv run

# Error guard for the empty-CATEGORIES case (nothing preprocessed yet).
NEED_CATS = test -n "$(CATEGORIES)" || { echo "No preprocessed categories under $(PROCESSED_DIR). Run 'uv run holmes preprocess' first."; exit 1; }

.DEFAULT_GOAL := help
.NOTPARALLEL:  # baselines fit multi-GB ALS models; never run them concurrently
.PHONY: help grid random bayes holmes baselines clean

# --- Help (default) --------------------------------------------------------
help:
	@echo "HOLMES experiment runner"
	@echo
	@echo "Targets (each loops over CATEGORIES; existing result files are skipped):"
	@echo "  grid        Grid search,   per fit seed         -> $(RESULTS_DIR)/<cat>/grid-seed<N>.json"
	@echo "  random      Random search, per fit x search seed -> $(RESULTS_DIR)/<cat>/random-seed<N>-search<M>.json"
	@echo "  bayes       Optuna TPE,    per fit x search seed -> $(RESULTS_DIR)/<cat>/bayes-seed<N>-search<M>.json"
	@echo "  holmes      Interactive session per fit x trial -> $(RESULTS_DIR)/<cat>/trajectory-seed<N>-trial<T>.json"
	@echo "  baselines   Run grid, random, bayes serially (all categories/seeds)."
	@echo "  clean       Remove result JSON under $(RESULTS_DIR)."
	@echo
	@echo "Variables (current values):"
	@echo "  PROCESSED_DIR = $(PROCESSED_DIR)"
	@echo "  CATEGORIES    = $(CATEGORIES)"
	@echo "  FIT_SEEDS     = $(FIT_SEEDS)   (ALS --seed; all strategies)"
	@echo "  SEARCH_SEEDS  = $(SEARCH_SEEDS)   (random/bayes search trajectory; grid ignores it)"
	@echo "  TRIALS        = $(TRIALS)   (HOLMES runs per fit seed)"
	@echo
	@echo "Note: each fit is a full ALS model (multi-GB at real scale); the baseline"
	@echo "targets fit once per seed, so they are long-running and block until done."

# --- Baselines -------------------------------------------------------------
# grid runs once per fit seed (it is deterministic given the seed). random and bayes sweep the
# full FIT_SEEDS x SEARCH_SEEDS cross product so fit-noise and search variance can be separated.
# A run whose --out file already exists is skipped, so reruns resume after a failure.
grid:
	@$(NEED_CATS)
	@for cat in $(CATEGORIES); do for fs in $(FIT_SEEDS); do \
		out=$(RESULTS_DIR)/$$cat/grid-seed$$fs.json; \
		if [ -f "$$out" ]; then echo "=== grid   $$cat seed=$$fs  (skip: exists)"; continue; fi; \
		mkdir -p $(RESULTS_DIR)/$$cat; \
		echo ">>> grid   $$cat fit-seed=$$fs"; \
		$(UV) holmes grid --data $(PROCESSED_DIR)/$$cat --seed $$fs --out $$out || exit $$?; \
	done; done

random:
	@$(NEED_CATS)
	@for cat in $(CATEGORIES); do for fs in $(FIT_SEEDS); do for ss in $(SEARCH_SEEDS); do \
		out=$(RESULTS_DIR)/$$cat/random-seed$$fs-search$$ss.json; \
		if [ -f "$$out" ]; then echo "=== random $$cat seed=$$fs search=$$ss  (skip: exists)"; continue; fi; \
		mkdir -p $(RESULTS_DIR)/$$cat; \
		echo ">>> random $$cat fit-seed=$$fs search-seed=$$ss"; \
		$(UV) holmes random --data $(PROCESSED_DIR)/$$cat --seed $$fs --search-seed $$ss --out $$out || exit $$?; \
	done; done; done

bayes:
	@$(NEED_CATS)
	@for cat in $(CATEGORIES); do for fs in $(FIT_SEEDS); do for ss in $(SEARCH_SEEDS); do \
		out=$(RESULTS_DIR)/$$cat/bayes-seed$$fs-search$$ss.json; \
		if [ -f "$$out" ]; then echo "=== bayes  $$cat seed=$$fs search=$$ss  (skip: exists)"; continue; fi; \
		mkdir -p $(RESULTS_DIR)/$$cat; \
		echo ">>> bayes  $$cat fit-seed=$$fs sampler-seed=$$ss"; \
		$(UV) holmes bayes --data $(PROCESSED_DIR)/$$cat --seed $$fs --sampler-seed $$ss --out $$out || exit $$?; \
	done; done; done

# --- HOLMES agentic loop ---------------------------------------------------
# The loop needs an LLM between rounds, so each run is an interactive Claude session with a
# pre-filled prompt; Claude drives the loop autonomously per skill/SKILL.md, then returns to the
# REPL (Ctrl-D out to launch the next run). For a fair comparison with the baselines, sweep one
# run per category x fit seed with TRIALS repeats each -- the repeats stand in for the search seed
# the LLM does not have. Runs are sequential, and any trajectory that already exists is skipped.
# The prompt avoids backticks/quotes so the shell does not interpret it; $(PROCESSED_DIR)/$(UV)
# are make-expanded while $$cat/$$fs/$$t/$$traj/$$datadir interpolate in the loop's shell.
holmes:
	@$(NEED_CATS)
	@for cat in $(CATEGORIES); do for fs in $(FIT_SEEDS); do for t in $$(seq 1 $(TRIALS)); do \
		traj=$(RESULTS_DIR)/$$cat/trajectory-seed$$fs-trial$$t.json; \
		if [ -f "$$traj" ]; then echo "=== holmes $$cat seed=$$fs trial=$$t  (skip: exists)"; continue; fi; \
		mkdir -p $(RESULTS_DIR)/$$cat; \
		datadir=$(PROCESSED_DIR)/$$cat; \
		echo ">>> holmes $$cat fit-seed=$$fs trial=$$t -> $$traj"; \
		claude "Run the HOLMES agentic hyperparameter-tuning loop using the holmes-hpo skill (skill/SKILL.md). Dataset --data is $$datadir; the trajectory log is $$traj. Use --seed $$fs for every fit, and prefix every holmes command with '$(UV)'. Start by running $(UV) holmes ranges to read the bounds and the max_iterations budget, seed iteration 1 by running $(UV) holmes heuristic --data $$datadir --trajectory $$traj --seed $$fs, then run the loop autonomously per SKILL.md until the trajectory reaches max_iterations, and finish with a held-out test eval and a short summary." || exit $$?; \
	done; done; done

# --- Baselines aggregate ---------------------------------------------------
# The three unattended baselines, run serially. HOLMES is interactive, so it is its own target.
baselines: grid random bayes

# --- Clean -----------------------------------------------------------------
clean:
	rm -f $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*/*.json
