"""Tests for ALSParams parsing and boundary validation."""

import pytest

from holmes.config import ALSParams


class TestFromDict:
    def test_parses_full_config(self):
        params = ALSParams.from_dict({"factors": 128, "regularization": 0.05, "iterations": 25, "alpha": 12.0})
        assert params == ALSParams(factors=128, regularization=0.05, iterations=25, alpha=12.0)

    def test_unknown_keys_are_ignored(self):
        params = ALSParams.from_dict({"factors": 64, "unrelated": "x"})
        assert params.factors == 64

    def test_missing_fields_fall_back_to_defaults(self):
        defaults = ALSParams()
        params = ALSParams.from_dict({"factors": 64})
        assert params.regularization == defaults.regularization
        assert params.iterations == defaults.iterations
        assert params.alpha == defaults.alpha

    @pytest.mark.parametrize("value", [128, 128.0, "128"])
    def test_integer_valued_inputs_are_accepted(self, value):
        assert ALSParams.from_dict({"factors": value}).factors == 128

    @pytest.mark.parametrize("field", ["factors", "iterations"])
    def test_non_integer_values_are_rejected(self, field):
        # Silently truncating 63.9 -> 63 would misattribute results to the wrong configuration.
        with pytest.raises(ValueError, match=field):
            ALSParams.from_dict({field: 63.9})
