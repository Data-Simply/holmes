# HOLMES agentic-loop runner.
#
# The grid / random / bayes baselines are driven by `holmes dispatch` (one runnable script per box;
# use `--boxes 1` for a single machine) -- see deploy/README.md. This Makefile is only for the
# HOLMES strategy, which dispatch deliberately does not handle: it runs each category x fit-seed x
# trial x model x effort as an interactive Claude session in an isolated sandbox. It loops over
# CATEGORIES and skips trajectories already complete, so an interrupted sweep resumes.
# Pure orchestration -- no changes to holmes/ code.
#
# Override any variable on the command line, e.g.:
#   make holmes CATEGORIES=Books FIT_SEEDS="0 1 2" TRIALS=5

# --- Variables (override on the command line) ------------------------------
# PROCESSED_DIR parent of the per-category preprocessed datasets
# CATEGORIES    categories to run; defaults to every preprocessed dataset found
#               under PROCESSED_DIR. Each is fed as --data PROCESSED_DIR/<cat>.
# FIT_SEEDS     ALS fit seeds (--seed): the model's own init randomness.
# TRIALS        HOLMES runs per fit seed. The LLM search has no integer seed, so
#               repeated trials are how its search variance is characterized.
# EFFORT_LEVELS Claude reasoning-effort levels to sweep (claude --effort:
#               low medium high xhigh max). Defaults to a single level; expand to sweep.
# MODELS        Claude model aliases to sweep (claude --model, e.g. opus sonnet).
#               Defaults to a single model; expand to sweep.
# RESULTS_DIR   where result JSON is written (namespaced per category)
# UV            runner; ensures the project env is used
PROCESSED_DIR ?= data/processed
CATEGORIES    ?= $(notdir $(patsubst %/,%,$(wildcard $(PROCESSED_DIR)/*/)))
FIT_SEEDS     ?= 0 1 2
TRIALS        ?= 3
EFFORT_LEVELS ?= high
MODELS        ?= opus
RESULTS_DIR   ?= results
UV            ?= uv run

# Absolute paths for the HOLMES sandbox symlinks (Claude runs with cwd = the sandbox dir).
ABS_SKILL   := $(abspath skill)
ABS_PROC    := $(abspath $(PROCESSED_DIR))
ABS_RESULTS := $(abspath $(RESULTS_DIR))

# Error guard for the empty-CATEGORIES case (nothing preprocessed yet).
NEED_CATS = test -n "$(CATEGORIES)" || { echo "No preprocessed categories under $(PROCESSED_DIR). Run 'uv run holmes preprocess' first."; exit 1; }

.DEFAULT_GOAL := help
.PHONY: help holmes

# --- Help (default) --------------------------------------------------------
help:
	@echo "HOLMES agentic-loop runner"
	@echo
	@echo "  holmes   Sandboxed Claude session per category x fit-seed x trial x model x effort"
	@echo "           -> $(RESULTS_DIR)/<cat>/trajectory-seed<N>-trial<T>-<model>-<effort>.json"
	@echo
	@echo "The grid/random/bayes baselines are run via 'holmes dispatch' (see deploy/README.md),"
	@echo "not this Makefile."
	@echo
	@echo "Variables (current values):"
	@echo "  PROCESSED_DIR = $(PROCESSED_DIR)"
	@echo "  CATEGORIES    = $(CATEGORIES)"
	@echo "  FIT_SEEDS     = $(FIT_SEEDS)   (ALS --seed)"
	@echo "  TRIALS        = $(TRIALS)   (HOLMES runs per fit seed)"
	@echo "  EFFORT_LEVELS = $(EFFORT_LEVELS)   (HOLMES claude --effort sweep)"
	@echo "  MODELS        = $(MODELS)   (HOLMES claude --model sweep)"

# --- HOLMES agentic loop ---------------------------------------------------
# HOLMES is the one strategy driven by an LLM, and the comparison only holds if that agent sees
# ONLY its skill and its own dataset -- never the other optimizers' code or results. So each trial
# runs in a throwaway sandbox dir (the agent's cwd) holding just: the holmes-hpo skill, a symlink to
# that one category's dataset, and a symlink to its own trajectory log (which lives under
# $(RESULTS_DIR) so it persists and resumes across runs). The `holmes` CLI is installed once as a
# uv tool so the skill can call it bare; isolation is only as strong as the sandbox (the symlinks
# still point back into the repo). The sweep also spans MODELS x EFFORT_LEVELS (the claude --model
# / --effort flags, set on the session, not seen by the agent). Completion is "trajectory length ==
# MAX_ITERATIONS", not mere existence: complete trajectories are skipped, incomplete ones resume,
# corrupt ones restart.
holmes:
	@$(NEED_CATS)
	@command -v holmes >/dev/null 2>&1 || { echo "Installing holmes as a uv tool (editable)..."; uv tool install --editable "$(CURDIR)"; }
	@command -v holmes >/dev/null 2>&1 || { echo "holmes still not on PATH after install; run 'uv tool update-shell' (or add the uv tool bin to PATH) and retry."; exit 1; }
	@maxit=$$($(UV) python -c "from holmes.config import MAX_ITERATIONS; print(MAX_ITERATIONS)"); \
	total=$$(( $(words $(CATEGORIES)) * $(words $(FIT_SEEDS)) * $(words $(MODELS)) * $(words $(EFFORT_LEVELS)) * $(TRIALS) )); i=0; \
	for cat in $(CATEGORIES); do for fs in $(FIT_SEEDS); do for m in $(MODELS); do for e in $(EFFORT_LEVELS); do for t in $$(seq 1 $(TRIALS)); do \
		i=$$((i+1)); \
		traj=$(ABS_RESULTS)/$$cat/trajectory-seed$$fs-trial$$t-$$m-$$e.json; \
		tag="$$cat seed=$$fs trial=$$t model=$$m effort=$$e"; \
		n=0; \
		if [ -f "$$traj" ]; then \
			n=$$($(UV) python -c "import json; print(len(json.load(open('$$traj'))))" 2>/dev/null) \
			|| { echo "[$$i/$$total] !!! holmes $$tag  (corrupt trajectory, restarting)"; rm -f "$$traj"; n=0; }; \
		fi; \
		if [ "$$n" -ge "$$maxit" ]; then echo "[$$i/$$total] === holmes $$tag  (skip: complete, $$n/$$maxit)"; continue; fi; \
		echo "[$$i/$$total] >>> holmes $$tag  ($$n/$$maxit done) -> $$traj"; \
		mkdir -p "$(ABS_RESULTS)/$$cat"; \
		sb=$$(mktemp -d); mkdir -p "$$sb/.claude/skills" "$$sb/data/processed"; \
		: 'isolation: symlink in ONLY the skill, one dataset, and the trajectory -- the sandbox is the agent cwd, so it sees nothing else of the repo'; \
		ln -s "$(ABS_SKILL)" "$$sb/.claude/skills/holmes-hpo"; \
		ln -s "$(ABS_PROC)/$$cat" "$$sb/data/processed/$$cat"; \
		ln -s "$$traj" "$$sb/trajectory.json"; \
		( cd "$$sb" && claude --model "$$m" --effort "$$e" "Use the holmes-hpo skill to run the agentic tuning loop on --data data/processed/$$cat, logging to --trajectory trajectory.json with --seed $$fs. If the trajectory already has entries, continue from there rather than re-seeding the heuristic." ); \
		rc=$$?; rm -rf "$$sb"; [ "$$rc" = 0 ] || exit "$$rc"; \
	done; done; done; done; done
