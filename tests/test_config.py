"""Tests for ALSParams parsing and boundary validation."""

import math

import pytest

from holmes.config import BAYES_SPACE, GRID_SPACE, HOLMES_SPACE, MAX_ITERATIONS, ALSParams


def test_max_iterations_matches_grid_size():
    """All three strategies share this cap; it must equal the grid's natural fit count."""
    assert math.prod(len(values) for values in GRID_SPACE.values()) == MAX_ITERATIONS


def test_grid_space_excludes_no_confidence_alpha():
    """``alpha=1.0`` reduces implicit-ALS confidence ``1 + α·r`` to a near-uniform constant, which
    effectively turns off the Hu et al. confidence weighting that distinguishes this model from
    plain SVD — so it stays out of the grid as a degenerate regime."""
    assert 1.0 not in GRID_SPACE["alpha"]


@pytest.mark.parametrize("derived_space", [BAYES_SPACE, HOLMES_SPACE])
def test_continuous_spaces_are_grid_hull(derived_space):
    """Bayes and HOLMES bounds are the per-axis (min, max) of GRID_SPACE — single source of truth."""
    expected = {name: (min(values), max(values)) for name, values in GRID_SPACE.items()}
    assert derived_space == expected


class TestFromDict:
    def test_parses_full_config(self):
        params = ALSParams.from_dict({"factors": 128, "regularization": 0.05, "iterations": 25, "alpha": 12.0})
        assert params == ALSParams(factors=128, regularization=0.05, iterations=25, alpha=12.0)

    def test_unknown_keys_are_ignored(self):
        params = ALSParams.from_dict({"factors": 64, "unrelated": "x"})
        assert params.factors == 64

    def test_missing_fields_fall_back_to_defaults(self):
        params = ALSParams.from_dict({"factors": 64})
        assert params.regularization == ALSParams().regularization
        assert params.alpha == ALSParams().alpha

    @pytest.mark.parametrize("value", [128, 128.0, "128"])
    def test_integer_valued_inputs_are_accepted(self, value):
        assert ALSParams.from_dict({"factors": value}).factors == 128

    @pytest.mark.parametrize("field", ["factors", "iterations"])
    def test_non_integer_values_are_rejected(self, field):
        # Silently truncating 63.9 -> 63 would misattribute results to the wrong configuration.
        with pytest.raises(ValueError, match=field):
            ALSParams.from_dict({field: 63.9})
