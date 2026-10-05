"""Hang guards for pre-collected Step 2 feature-discovery scans."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import Array

from alberta_framework import FixedBudgetFeatureLearner, run_feature_discovery_arrays
from alberta_framework.core import feature_discovery


_MAX_STEPS = 10_000


def _feature_inputs(num_steps: int) -> tuple[Any, Any, Array, Array]:
    observations = jnp.zeros((num_steps, 2), dtype=jnp.float32)
    targets = jnp.zeros((num_steps, 1), dtype=jnp.float32)
    learner = FixedBudgetFeatureLearner(
        n_features=2, n_tasks=1, candidate_count=1, replacement_interval=0
    )
    state = learner.init(feature_dim=2, key=jr.key(0))
    return learner, state, observations, targets


def test_oversized_precollected_scan_rejected_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner, state, observations, targets = _feature_inputs(_MAX_STEPS + 1)
    scans: list[int] = []

    def spy_scan(_step: Any, _state: Any, xs: tuple[Array, Array]) -> None:
        scans.append(int(xs[0].shape[0]))
        raise AssertionError("oversized sequence reached lax.scan")

    monkeypatch.setattr(feature_discovery.jax.lax, "scan", spy_scan)
    with pytest.raises(
        ValueError, match=rf"observations num_steps.*\[1, {_MAX_STEPS}\]"
    ):
        run_feature_discovery_arrays(learner, state, observations, targets)
    assert scans == []


def test_last_supported_length_reaches_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    learner, state, observations, targets = _feature_inputs(_MAX_STEPS)
    scans: list[int] = []

    class ScanReached(Exception):
        pass

    def spy_scan(_step: Any, _state: Any, xs: tuple[Array, Array]) -> None:
        scans.append(int(xs[0].shape[0]))
        raise ScanReached

    monkeypatch.setattr(feature_discovery.jax.lax, "scan", spy_scan)
    with pytest.raises(ScanReached):
        run_feature_discovery_arrays(learner, state, observations, targets)
    assert scans == [_MAX_STEPS]
