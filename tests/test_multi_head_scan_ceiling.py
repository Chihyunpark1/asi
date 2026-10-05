"""Regression tests for the multi-head learner array scan ceiling."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    run_multi_head_learning_loop,
)


class _ScanReached(Exception):
    pass


def test_multi_head_scan_rejects_oversized_arrays_before_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = MultiHeadMLPLearner(n_heads=1, hidden_sizes=(), sparsity=0.0)
    state = learner.init(feature_dim=2, key=jr.key(0))
    calls: list[bool] = []

    def scan_spy(*args: object, **kwargs: object) -> None:
        calls.append(True)
        raise _ScanReached

    monkeypatch.setattr(jax.lax, "scan", scan_spy)
    observations = jnp.zeros((10_001, 2), dtype=jnp.float32)
    targets = jnp.zeros((10_001, 1), dtype=jnp.float32)

    with pytest.raises(ValueError, match="between 1 and 10000 scan steps"):
        run_multi_head_learning_loop(learner, state, observations, targets)
    assert calls == []

    observations = observations[:10_000]
    targets = targets[:10_000]
    with pytest.raises(_ScanReached):
        run_multi_head_learning_loop(learner, state, observations, targets)
    assert calls == [True]