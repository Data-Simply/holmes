"""Tests for ALSParams parsing and boundary validation."""

import pytest

from holmes.config import ALSParams


def _full_params(**overrides) -> dict:
    """A complete hyperparameter mapping (derived from the dataclass defaults) with overrides."""
    return {**ALSParams().to_dict(), **overrides}


class TestFromDict:
    def test_parses_full_config(self):
        params = ALSParams.from_dict({"factors": 128, "regularization": 0.05, "iterations": 25, "alpha": 12.0})
        assert params == ALSParams(factors=128, regularization=0.05, iterations=25, alpha=12.0)

    def test_unknown_keys_are_rejected(self):
        # Silently dropping unknown keys would let a typo'd key (or a whole wrong-shaped mapping,
        # e.g. a results file) evaluate the all-default config and report it as a real score.
        with pytest.raises(ValueError, match="unrelated"):
            ALSParams.from_dict(_full_params(unrelated="x"))

    def test_missing_fields_are_rejected(self):
        spec = _full_params()
        del spec["regularization"]
        del spec["alpha"]
        with pytest.raises(ValueError) as excinfo:
            ALSParams.from_dict(spec)
        assert "regularization" in str(excinfo.value)
        assert "alpha" in str(excinfo.value)

    def test_wrong_shaped_mapping_is_rejected(self):
        # The exact failure mode of `holmes eval --params results/grid.json`: a nested mapping
        # whose top-level keys are not hyperparameters must error, not evaluate the defaults.
        with pytest.raises(ValueError, match="params"):
            ALSParams.from_dict({"params": _full_params(), "seed": 0, "score": 0.1})

    @pytest.mark.parametrize("value", [128, 128.0, "128"])
    def test_integer_valued_inputs_are_accepted(self, value):
        assert ALSParams.from_dict(_full_params(factors=value)).factors == 128

    @pytest.mark.parametrize("field", ["factors", "iterations"])
    def test_non_integer_values_are_rejected(self, field):
        # Silently truncating 63.9 -> 63 would misattribute results to the wrong configuration.
        with pytest.raises(ValueError, match=field):
            ALSParams.from_dict(_full_params(**{field: 63.9}))
