"""Search strategies: grid, Bayesian (Optuna), and the agentic HOLMES loop."""

from holmes.search.harness import evaluate_config
from holmes.search.heuristics import initial_params

__all__ = ["evaluate_config", "initial_params"]
