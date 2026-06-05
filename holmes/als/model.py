"""A thin, reproducible wrapper around ``implicit``'s ALS for implicit feedback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import threadpoolctl
from implicit.als import AlternatingLeastSquares

if TYPE_CHECKING:
    import scipy.sparse as sp

    from holmes.config import ALSParams


class ALSRecommender:
    """Alternating least squares recommender for implicit feedback.

    Wraps :class:`implicit.als.AlternatingLeastSquares` to pin the random seed, expose the
    learned factors for diagnostics, and provide batched top-k recommendation. The ``alpha``
    confidence-scaling hyperparameter is passed to the underlying model (Hu et al. 2008),
    which forms confidence ``c_ui = 1 + alpha * r_ui`` internally.

    Attributes:
        params: The hyperparameters this recommender was constructed with.
        seed: Random seed controlling factor initialization.
    """

    def __init__(self, params: ALSParams, seed: int = 0) -> None:
        """Initialize the recommender.

        Args:
            params: ALS hyperparameters (factors, regularization, iterations, alpha).
            seed: Random seed for reproducible factor initialization.
        """
        self.params = params
        self.seed = seed
        self._model = AlternatingLeastSquares(
            factors=params.factors,
            regularization=params.regularization,
            alpha=params.alpha,
            iterations=params.iterations,
            calculate_training_loss=False,
            random_state=seed,
        )
        self._fitted = False

    def fit(self, train_ui: sp.csr_matrix) -> ALSRecommender:
        """Fit the model on a user-item training matrix.

        Args:
            train_ui: Rating-weighted CSR matrix of shape ``(n_users, n_items)``; stored values are
                the raw star ratings consumed as ``r_ui`` in ``c_ui = 1 + α·r_ui``.

        Returns:
            ALSRecommender: ``self``, to allow call chaining.
        """
        # Pin BLAS to a single thread for the duration of the fit: implicit runs its own
        # parallelism over the ALS solve, and nesting it with a multithreaded BLAS both causes
        # severe contention and makes the fit nondeterministic. Run-to-run reproducibility is a
        # correctness property the HOLMES loop relies on, so this must hold for every fit — a
        # context manager guarantees that regardless of import order or garbage collection.
        with threadpoolctl.threadpool_limits(1, "blas"):
            self._model.fit(train_ui, show_progress=False)
        self._fitted = True
        return self

    @property
    def user_factors(self) -> np.ndarray:
        """User embedding matrix of shape ``(n_users, factors)``."""
        return np.asarray(self._model.user_factors)

    @property
    def item_factors(self) -> np.ndarray:
        """Item embedding matrix of shape ``(n_items, factors)``."""
        return np.asarray(self._model.item_factors)

    def recommend(
        self,
        user_ids: np.ndarray,
        train_ui: sp.csr_matrix,
        k: int,
        *,
        filter_seen: bool = True,
    ) -> np.ndarray:
        """Recommend the top-k items for a batch of users.

        This assumes the catalog is much larger than ``k`` (true for the Amazon Books matrix:
        thousands of items vs. ``k=10``), so every user has at least ``k`` recommendable items and
        ``implicit`` never needs to pad the row. Were that assumption violated — a user having seen
        all but fewer than ``k`` items of the entire catalog — implicit would backfill the row with
        arbitrary item ids; that case does not arise here and is intentionally not handled.

        Args:
            user_ids: Array of user indices to score.
            train_ui: The full training matrix; rows for ``user_ids`` supply seen items.
            k: Number of items to return per user.
            filter_seen: If true, exclude items the user already interacted with in training.

        Returns:
            np.ndarray: Item-index array of shape ``(len(user_ids), k)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if not self._fitted:
            msg = "ALSRecommender.recommend called before fit()."
            raise RuntimeError(msg)
        ids, _ = self._model.recommend(user_ids, train_ui[user_ids], N=k, filter_already_liked_items=filter_seen)
        return np.asarray(ids)
