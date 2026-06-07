# HOLMES experiment runner.
#
# Wraps the `holmes` CLI so the four strategies can be launched with one command each.
# grid / random / bayes run unattended over $(SEEDS); holmes opens an interactive Claude
# Code session primed to drive the agentic loop (see the Stop hook in .claude/settings.json,
# which paces and ends the loop). Pure orchestration -- no changes to holmes/ code.
#
# Override any variable on the command line, e.g.:
#   make grid DATA=data/processed/Electronics SEEDS="0 1 2 3 4"

# --- Variables (override on the command line) ------------------------------
# DATA        preprocessed dataset directory (passed as --data)
# CATEGORY    category built by `make preprocess`
# SEEDS       fit seeds; each strategy runs once per seed
# RESULTS_DIR where per-seed result JSON is written
# TRAJECTORY  HOLMES append-only log
# UV          runner; ensures the project env is used
DATA        ?= data/processed/Books
CATEGORY    ?= Books
SEEDS       ?= 0 1 2
RESULTS_DIR ?= results
TRAJECTORY  ?= $(RESULTS_DIR)/trajectory.json
UV          ?= uv run

.DEFAULT_GOAL := help
.PHONY: help preprocess grid random bayes holmes all clean

# --- Help (default) --------------------------------------------------------
help:
	@echo "HOLMES experiment runner"
	@echo
	@echo "Targets:"
	@echo "  preprocess  Build the interaction matrix for CATEGORY ($(CATEGORY))."
	@echo "  grid        Grid search,   one fit per seed -> $(RESULTS_DIR)/grid-seed<N>.json"
	@echo "  random      Random search, one fit per seed -> $(RESULTS_DIR)/random-seed<N>.json"
	@echo "  bayes       Optuna TPE,    one fit per seed -> $(RESULTS_DIR)/bayes-seed<N>.json"
	@echo "  holmes      Open a Claude Code session that drives the agentic loop."
	@echo "  all         Run grid, random, bayes (all seeds), then holmes."
	@echo "  clean       Remove $(RESULTS_DIR)/*.json."
	@echo
	@echo "Variables (current values):"
	@echo "  DATA      = $(DATA)"
	@echo "  CATEGORY  = $(CATEGORY)"
	@echo "  SEEDS     = $(SEEDS)"
	@echo
	@echo "Note: each fit is a full ALS model (multi-GB at real scale); the baseline"
	@echo "targets fit once per seed, so they are long-running and block until done."

# --- Preprocess ------------------------------------------------------------
preprocess:
	$(UV) holmes preprocess --category $(CATEGORY)

# --- Baselines (one fit per seed) ------------------------------------------
grid:
	@mkdir -p $(RESULTS_DIR)
	@for s in $(SEEDS); do \
		echo ">>> grid   seed=$$s"; \
		$(UV) holmes grid --data $(DATA) --seed $$s \
			--out $(RESULTS_DIR)/grid-seed$$s.json || exit $$?; \
	done

random:
	@mkdir -p $(RESULTS_DIR)
	@for s in $(SEEDS); do \
		echo ">>> random seed=$$s"; \
		$(UV) holmes random --data $(DATA) --seed $$s --search-seed $$s \
			--out $(RESULTS_DIR)/random-seed$$s.json || exit $$?; \
	done

bayes:
	@mkdir -p $(RESULTS_DIR)
	@for s in $(SEEDS); do \
		echo ">>> bayes  seed=$$s"; \
		$(UV) holmes bayes --data $(DATA) --seed $$s --sampler-seed $$s \
			--out $(RESULTS_DIR)/bayes-seed$$s.json || exit $$?; \
	done

# --- HOLMES agentic loop ---------------------------------------------------
# The loop needs an LLM between rounds, so this opens an interactive Claude Code session with
# a pre-filled prompt. HOLMES_LOOP/HOLMES_TRAJECTORY are read by the Stop hook, which keeps the
# loop running until the trajectory reaches max_iterations, then lets the session return to the
# REPL (exit with Ctrl-D). The prompt avoids backticks so the shell does not interpret it.
define HOLMES_PROMPT
Run the HOLMES agentic hyperparameter-tuning loop using the holmes-hpo skill (skill/SKILL.md). \
Dataset --data is $(DATA); the trajectory log is $(TRAJECTORY). Start by running holmes ranges \
to read the bounds and the max_iterations budget, seed iteration 1 with \
"holmes heuristic --data $(DATA) --trajectory $(TRAJECTORY) --seed 0", then run the loop \
autonomously per SKILL.md until the trajectory reaches max_iterations, and finish with a held-out \
test eval and a short summary.
endef
export HOLMES_PROMPT

holmes:
	@mkdir -p $(RESULTS_DIR)
	HOLMES_LOOP=1 HOLMES_TRAJECTORY=$(TRAJECTORY) HOLMES_DATA=$(DATA) \
		claude "$$HOLMES_PROMPT"

# --- Everything ------------------------------------------------------------
# Baselines first (unattended), then the interactive HOLMES session.
all: grid random bayes holmes

# --- Clean -----------------------------------------------------------------
clean:
	rm -f $(RESULTS_DIR)/*.json
