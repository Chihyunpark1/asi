"""Regression tests for fixed-trace decay-rate config bounds."""

import pytest

from alberta_framework.core.state_builder import FixedTraceStateBuilderConfig


def test_fixed_trace_from_config_rejects_oversized_decay_lists_before_conversion() -> None:
    payload = FixedTraceStateBuilderConfig(observation_dim=2).to_config()
    payload["observation_decay_rates"] = [0.5] * 4_097

    with pytest.raises(
        ValueError,
        match="serialized observation_decay_rates must contain at most 4096",
    ):
        FixedTraceStateBuilderConfig.from_config(payload)


def test_fixed_trace_config_rejects_oversized_decay_tuple() -> None:
    with pytest.raises(ValueError, match="observation_decay_rates length must be at most 4096"):
        FixedTraceStateBuilderConfig(
            observation_dim=2,
            observation_decay_rates=(0.5,) * 4_097,
        )
