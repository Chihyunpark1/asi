"""Regression tests for off-policy Horde serialized config bounds."""

import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.off_policy_horde import OffPolicyHordeLearner


def test_off_policy_horde_rejects_oversized_hidden_sizes_before_conversion() -> None:
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
    payload = OffPolicyHordeLearner(spec, hidden_sizes=()).to_config()
    payload["hidden_sizes"] = [1] * 4_097

    with pytest.raises(ValueError, match="serialized hidden_sizes must contain at most 4096"):
        OffPolicyHordeLearner.from_config(payload)


def test_off_policy_horde_constructor_rejects_oversized_hidden_sizes() -> None:
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
    with pytest.raises(ValueError, match="hidden_sizes length must be at most 4096"):
        OffPolicyHordeLearner(spec, hidden_sizes=(1,) * 4_097)
