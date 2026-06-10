"""Tests for the network-free Polars preprocessing helpers.

These exercise the transform pipeline on small in-memory frames; the Parquet scan (network) is
not tested here, consistent with treating I/O as an external dependency.
"""

import polars as pl

from holmes.data.preprocess import (
    _REVIEW_COLUMNS,
    _assign_indices_and_split,
    _build_review_cache,
    _deduplicate_interactions,
    _k_core_filter,
    build_dataset,
)


def _reviews(rows: list[dict]) -> pl.LazyFrame:
    return pl.DataFrame(
        rows,
        schema={"user_id": pl.Utf8, "parent_asin": pl.Utf8, "rating": pl.Float64, "timestamp": pl.Int64},
    ).lazy()


class TestDeduplicateInteractions:
    def test_collapses_repeats_keeping_latest_timestamp(self):
        reviews = _reviews(
            [
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 10},
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 30},  # later repeat
                {"user_id": "cust_a", "parent_asin": "B002", "rating": 4.0, "timestamp": 20},
            ],
        )
        out = _deduplicate_interactions(reviews, min_rating=4.0).collect()
        assert out.height == 2
        latest = out.filter(pl.col("parent_asin") == "B001")["timestamp"].item()
        assert latest == 30

    def test_downgraded_rereview_keeps_the_latest_reviews_rating(self):
        """The kept row must be one real review — never the newest timestamp paired with an
        older review's higher rating, which would inflate confidence (c_ui = 1 + alpha * r)."""
        reviews = _reviews(
            [
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 10},
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 4.0, "timestamp": 20},  # later downgrade
            ],
        )
        out = _deduplicate_interactions(reviews, min_rating=4.0).collect()
        assert out.to_dicts() == [{"user_id": "cust_a", "parent_asin": "B001", "timestamp": 20, "rating": 4.0}]

    def test_tied_timestamps_keep_the_higher_rating_regardless_of_row_order(self):
        """Two reviews tied on timestamp must dedupe identically for either input order — the
        rating tiebreaker makes the kept row value-determined, not row-order-determined."""
        forward = _reviews(
            [
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 4.0, "timestamp": 10},
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 10},
            ],
        )
        backward = _reviews(
            [
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 10},
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 4.0, "timestamp": 10},
            ],
        )
        expected = [{"user_id": "cust_a", "parent_asin": "B001", "timestamp": 10, "rating": 5.0}]
        assert _deduplicate_interactions(forward, min_rating=4.0).collect().to_dicts() == expected
        assert _deduplicate_interactions(backward, min_rating=4.0).collect().to_dicts() == expected

    def test_drops_low_rated_null_and_empty_rows(self):
        reviews = _reviews(
            [
                {"user_id": "cust_a", "parent_asin": "B001", "rating": 2.0, "timestamp": 1},  # below threshold
                {"user_id": "cust_b", "parent_asin": "B002", "rating": None, "timestamp": 1},  # null rating
                {"user_id": "", "parent_asin": "B003", "rating": 5.0, "timestamp": 1},  # empty user
                {"user_id": "cust_c", "parent_asin": "B004", "rating": 5.0, "timestamp": 1},  # kept
            ],
        )
        out = _deduplicate_interactions(reviews, min_rating=4.0).collect()
        # Each drop reason must independently fire — height alone would still pass if e.g. only
        # the empty-user row survived through a regression that re-broke the rating filter.
        assert out["user_id"].to_list() == ["cust_c"]
        assert out["parent_asin"].to_list() == ["B004"]
        assert out["rating"].null_count() == 0
        assert (out["rating"] >= 4.0).all()


class TestKCoreFilter:
    def test_drops_users_and_items_below_threshold(self):
        frame = pl.DataFrame(
            {
                "user_id": ["cust_a", "cust_a", "cust_a", "cust_b"],
                "parent_asin": ["B001", "B002", "B003", "B999"],
                "timestamp": [1, 2, 3, 4],
            },
        )
        filtered = _k_core_filter(frame, min_user=2, min_item=1)
        assert set(filtered["user_id"]) == {"cust_a"}
        assert set(filtered["parent_asin"]) == {"B001", "B002", "B003"}
        assert filtered.height == 3

    def test_iterates_until_stable(self):
        # B777 has 1 interaction -> dropped; cust_b then has 1 -> dropped; B002 then has 1 -> dropped.
        frame = pl.DataFrame(
            {
                "user_id": ["cust_a", "cust_a", "cust_b", "cust_b"],
                "parent_asin": ["B001", "B002", "B001", "B777"],
                "timestamp": [1, 2, 3, 4],
            },
        )
        assert _k_core_filter(frame, min_user=2, min_item=2).height == 0


class TestAssignIndicesAndSplit:
    def _frame(self) -> pl.DataFrame:
        # cust_a: 4 interactions; cust_b: 2; cust_c: 1 (no training history once test is held out).
        return pl.DataFrame(
            {
                "user_id": ["cust_a", "cust_a", "cust_a", "cust_a", "cust_b", "cust_b", "cust_c"],
                "parent_asin": ["B10", "B11", "B12", "B13", "B20", "B21", "B30"],
                "timestamp": [1, 2, 3, 4, 1, 2, 1],
                "rating": [5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0],
            },
        )

    def test_latest_interaction_is_the_test_positive(self):
        dataset = _assign_indices_and_split(self._frame())
        # cust_a's latest is B13, cust_b's latest is B21; map their indices back via popularity layout.
        # Reconstruct id->idx by re-deriving dense rank order (alphabetical for these ids).
        items = ["B10", "B11", "B12", "B13", "B20", "B21", "B30"]
        idx = {name: i for i, name in enumerate(sorted(items))}
        test_map = dict(zip(dataset.test_users.tolist(), dataset.test_items.tolist(), strict=True))
        users = ["cust_a", "cust_b", "cust_c"]
        uidx = {name: i for i, name in enumerate(sorted(users))}
        assert test_map[uidx["cust_a"]] == idx["B13"]
        assert test_map[uidx["cust_b"]] == idx["B21"]

    def test_user_without_training_history_is_excluded(self):
        dataset = _assign_indices_and_split(self._frame())
        users = ["cust_a", "cust_b", "cust_c"]
        uidx = {name: i for i, name in enumerate(sorted(users))}
        # cust_c has a single interaction: it becomes the test positive but there is no train row,
        # so it must not appear in the test set.
        assert uidx["cust_c"] not in set(dataset.test_users.tolist())

    def test_validation_only_for_users_with_enough_history(self):
        dataset = _assign_indices_and_split(self._frame())
        users = ["cust_a", "cust_b", "cust_c"]
        uidx = {name: i for i, name in enumerate(sorted(users))}
        val_users = set(dataset.val_users.tolist())
        assert uidx["cust_a"] in val_users  # 4 interactions -> has a validation positive
        assert uidx["cust_b"] not in val_users  # only 2 interactions -> no validation positive

    def test_training_matrix_excludes_held_out_items(self):
        dataset = _assign_indices_and_split(self._frame())
        # cust_a (4 interactions) keeps exactly 2 training items (B10, B11); B12 val, B13 test.
        assert int(dataset.train_ui.getrow(0).nnz) == 2
        assert int(dataset.item_popularity.sum()) == dataset.n_interactions

    def test_training_matrix_stores_raw_ratings_not_binary_ones(self):
        """The matrix value is the raw rating (used as ``r_ui`` in ``c_ui = 1 + alpha * r_ui``), not 1.0.

        Cust_a's first two interactions (B10 @ rating 5.0, B11 @ rating 4.0) end up in train; the
        stored CSR values must equal those ratings, otherwise we are back to binary confidence and
        the strong/lukewarm positive distinction is silently lost.
        """
        dataset = _assign_indices_and_split(self._frame())
        items = ["B10", "B11", "B12", "B13", "B20", "B21", "B30"]
        idx = {name: i for i, name in enumerate(sorted(items))}
        users = ["cust_a", "cust_b", "cust_c"]
        uidx = {name: i for i, name in enumerate(sorted(users))}
        row = dataset.train_ui.getrow(uidx["cust_a"])
        stored = dict(zip(row.indices.tolist(), row.data.tolist(), strict=True))
        assert stored == {idx["B10"]: 5.0, idx["B11"]: 4.0}

    def test_split_is_order_independent_with_tied_timestamps(self):
        """Tied boundary timestamps must split identically regardless of input row order.

        Polars `sort` preserves input order on ties, and `group_by` output order is unstable — so
        without a value-based tiebreaker the held-out item for a tied user depends on incoming row
        order. Here the two interactions tied at the latest timestamp are presented in OPPOSITE
        orders; the split must be identical either way.
        """

        def frame(tied_order: list[str]) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "user_id": ["cust_a", "cust_a", *(["cust_a"] * 2)],
                    "parent_asin": ["B10", "B11", *tied_order],
                    "timestamp": [1, 2, 3, 3],  # B12 and B13 tie at the latest timestamp
                    "rating": [5.0, 5.0, 5.0, 5.0],
                },
            )

        def split_maps(dataset):
            test = dict(zip(dataset.test_users.tolist(), dataset.test_items.tolist(), strict=True))
            val = dict(zip(dataset.val_users.tolist(), dataset.val_items.tolist(), strict=True))
            return test, val

        forward = _assign_indices_and_split(frame(["B12", "B13"]))
        reversed_ties = _assign_indices_and_split(frame(["B13", "B12"]))
        assert split_maps(forward) == split_maps(reversed_ties)
        assert (forward.train_ui != reversed_ties.train_ui).nnz == 0


class TestReviewCache:
    def test_build_cache_keeps_only_required_columns(self, tmp_path):
        """The raw reviews carry many fields; the cache must project to just the four we consume."""
        src = tmp_path / "reviews.jsonl"
        src.write_text(
            '{"user_id":"cust_a","parent_asin":"B001","rating":5.0,"timestamp":10,"title":"t","helpful_vote":3}\n'
            '{"user_id":"cust_b","parent_asin":"B002","rating":4.0,"timestamp":20,"title":"u","helpful_vote":0}\n',
        )
        cache_path = tmp_path / "Books.parquet"
        _build_review_cache(str(src), cache_path)

        cached = pl.read_parquet(cache_path)
        assert cached.columns == list(_REVIEW_COLUMNS)
        # Pin row values too — a regression that wrote the right column set but the wrong row
        # values (e.g. swapped rating/timestamp) would pass a column-set + height check alone.
        assert cached.to_dicts() == [
            {"user_id": "cust_a", "parent_asin": "B001", "rating": 5.0, "timestamp": 10},
            {"user_id": "cust_b", "parent_asin": "B002", "rating": 4.0, "timestamp": 20},
        ]

    def test_build_dataset_reuses_existing_cache_without_rereading_source(self, tmp_path, monkeypatch):
        """A present cache is scanned directly; the cache builder is never invoked."""
        reviews = pl.DataFrame(
            {
                "user_id": ["cust_a", "cust_a", "cust_a", "cust_b", "cust_b", "cust_b"],
                "parent_asin": ["B10", "B11", "B12", "B20", "B21", "B22"],
                "rating": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                "timestamp": [1, 2, 3, 1, 2, 3],
            },
        )
        reviews.write_parquet(tmp_path / "Books.parquet")

        def boom(*args, **kwargs):
            msg = "build_dataset rebuilt the cache despite a present one"
            raise AssertionError(msg)

        monkeypatch.setattr("holmes.data.preprocess._build_review_cache", boom)
        dataset = build_dataset(category="Books", cache_dir=tmp_path, min_user=1, min_item=1)
        assert dataset.n_users == 2
        assert dataset.n_items == 6
