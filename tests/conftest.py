"""Shared fixtures: a small but learnable book-recommendation interaction matrix.

Users belong to one of four taste groups; each interacts only with books from its group, so a
correctly fit ALS model should recommend in-group books. Group membership is recoverable from
the indices: book ``b`` is in group ``b // BOOKS_PER_GROUP`` and customer ``u`` is in group
``u % N_GROUPS``.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from holmes.data.dataset import Dataset

N_GROUPS = 4
BOOKS_PER_GROUP = 40
N_CUSTOMERS = 200
INTERACTIONS_PER_CUSTOMER = 16


def book_group(book_idx: int) -> int:
    """Return the taste group a book index belongs to."""
    return book_idx // BOOKS_PER_GROUP


def customer_group(customer_idx: int) -> int:
    """Return the taste group a customer index belongs to."""
    return customer_idx % N_GROUPS


@pytest.fixture
def books_dataset() -> Dataset:
    """A leave-last-out book dataset with clear group structure for ALS to recover."""
    rng = np.random.default_rng(7)
    n_items = N_GROUPS * BOOKS_PER_GROUP
    rows: list[int] = []
    cols: list[int] = []
    val_users, val_items, test_users, test_items = [], [], [], []

    for customer in range(N_CUSTOMERS):
        group = customer_group(customer)
        group_books = np.arange(group * BOOKS_PER_GROUP, (group + 1) * BOOKS_PER_GROUP)
        chosen = rng.choice(group_books, size=INTERACTIONS_PER_CUSTOMER, replace=False)
        # Last two interactions are held out (val, test); the rest are training.
        test_users.append(customer)
        test_items.append(int(chosen[-1]))
        val_users.append(customer)
        val_items.append(int(chosen[-2]))
        for book in chosen[:-2]:
            rows.append(customer)
            cols.append(int(book))

    train_ui = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(N_CUSTOMERS, n_items),
    )
    return Dataset(
        train_ui=train_ui,
        val_users=np.array(val_users),
        val_items=np.array(val_items),
        test_users=np.array(test_users),
        test_items=np.array(test_items),
    )
