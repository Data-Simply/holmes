"""HOLMES: Hypothesis-driven Optimization via LLM-guided Model Exploration and Search.

Benchmarks hyperparameter optimization of an implicit-feedback ALS recommender on the
Amazon Reviews 2023 (Books) dataset, comparing grid search, Bayesian optimization, and
the agentic HOLMES loop.
"""

# Pin OpenBLAS to one thread for the whole process before any numpy/scipy import. The ALS
# fit also wraps itself in ``threadpoolctl.threadpool_limits(1, "blas")`` for the runtime
# guarantee, but ``implicit.check_blas_config`` inspects the OpenBLAS env var rather than
# the live thread count, so setting this here both silences that warning and gives us a
# single, declarative statement of the project's "BLAS = 1 thread" determinism property.
# ``setdefault`` keeps an explicit ``OPENBLAS_NUM_THREADS=N`` override from the shell.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from holmes.config import ALSParams

__all__ = ["ALSParams"]
