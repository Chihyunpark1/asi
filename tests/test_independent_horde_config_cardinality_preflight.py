"""Regression tests for independent Horde serialized config bounds."""

import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.independent_demon_horde import IndependentDemonHorde


def _spec():
    return create_horde_spec(
        [
            GVFSpec(
                name="d0",
                demon_type=DemonType.PREDICTION,
                gamma=0.0,
                lamda=0.0,
                cumulant_index=0,
            )
        ]
    )


def test_independent_horde_rejects_oversized_hidden_sizes_before_conversion() -> None:
    payload = IndependentDemonHorde(_spec(), hidden_sizes=()).to_config()
    payload["hidden_sizes"] = [1] * 4_097

    with pytest.raises(ValueError, match="serialized hidden_sizes must contain at most 4096"):
        IndependentDemonHorde.from_config(payload)


def test_independent_horde_constructor_rejects_oversized_hidden_sizes() -> None:
    with pytest.raises(ValueError, match="hidden_sizes length must be at most 4096"):
        IndependentDemonHorde(_spec(), hidden_sizes=(1,) * 4_097)
