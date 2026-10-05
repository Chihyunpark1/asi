"""Regression tests for Horde serialized hidden-size preflight."""

import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.horde import HordeLearner, MixedHorde


def test_horde_decoders_reject_oversized_hidden_sizes_before_conversion() -> None:
    spec = create_horde_spec(
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
    for learner in (HordeLearner(spec, hidden_sizes=()), MixedHorde(spec, hidden_sizes=())):
        payload = learner.to_config()
        payload["hidden_sizes"] = [1] * 4_097
        with pytest.raises(ValueError, match="serialized hidden_sizes must contain at most 4096"):
            type(learner).from_config(payload)


def test_horde_constructors_reject_oversized_hidden_sizes() -> None:
    spec = create_horde_spec(
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
    for constructor in (HordeLearner, MixedHorde):
        with pytest.raises(ValueError, match="hidden_sizes length must be at most 4096"):
            constructor(spec, hidden_sizes=(1,) * 4_097)
