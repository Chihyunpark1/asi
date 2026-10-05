"""Regression tests for the independent-horde array scan ceiling."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework import DemonType, GVFSpec, create_horde_spec
from alberta_framework.core.independent_demon_horde import (
    IndependentDemonHorde,
    run_independent_horde_learning_loop,
)


class _ScanReached(Exception):
    pass


def test_independent_horde_scan_rejects_oversized_arrays_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    horde = IndependentDemonHorde(
        horde_spec=create_horde_spec(
            [
                GVFSpec(
                    name="d0",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=0,
                )
            ]
        ),
        hidden_sizes=(),
    )
    state = horde.init(feature_dim=2, key=jr.key(0))
    calls: list[bool] = []

    def scan_spy(*args: object, **kwargs: object) -> None:
        calls.append(True)
        raise _ScanReached

    monkeypatch.setattr(jax.lax, "scan", scan_spy)
    observations = jnp.zeros((10_001, 2), dtype=jnp.float32)
    cumulants = jnp.zeros((10_001, 1), dtype=jnp.float32)
    next_observations = jnp.zeros_like(observations)

    with pytest.raises(ValueError, match="between 1 and 10000 scan steps"):
        run_independent_horde_learning_loop(
            horde, state, observations, cumulants, next_observations
        )
    assert calls == []

    observations = observations[:10_000]
    cumulants = cumulants[:10_000]
    next_observations = next_observations[:10_000]
    with pytest.raises(_ScanReached):
        run_independent_horde_learning_loop(
            horde, state, observations, cumulants, next_observations
        )
    assert calls == [True]
