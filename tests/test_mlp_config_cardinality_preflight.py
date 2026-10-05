"""Regression tests for serialized MLPLearner hidden-size bounds."""

import pytest

from alberta_framework.core.learners import MLPLearner


def test_mlp_from_config_rejects_oversized_hidden_sizes_before_decode() -> None:
    payload = MLPLearner(hidden_sizes=()).to_config()
    payload["hidden_sizes"] = [1] * 4_097

    with pytest.raises(ValueError, match="serialized hidden_sizes must contain at most 4096"):
        MLPLearner.from_config(payload)


def test_mlp_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length must be at most 4096"):
        MLPLearner(hidden_sizes=(1,) * 4_097)
