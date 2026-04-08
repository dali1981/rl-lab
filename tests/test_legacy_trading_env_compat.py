"""Compatibility-only checks for deprecated legacy TradingEnv."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rl_trading_lab.environment.trading_env import TradingEnv


def _sample_data(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.0, 0.4, rows))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "open": close + rng.normal(0.0, 0.2, rows),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + rng.normal(0.0, 20.0, rows),
            "feature_1": rng.normal(0.0, 1.0, rows),
            "feature_2": rng.normal(0.0, 1.0, rows),
        }
    )


def _build_legacy_env() -> TradingEnv:
    return TradingEnv(
        df=_sample_data(),
        lookback_window=20,
        min_episode_length=50,
        reward_type="returns",
        randomize_start=False,
        hold_closes_position=True,
    )


def test_legacy_trading_env_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="TradingEnv is deprecated"):
        _build_legacy_env()


def test_legacy_trading_env_basic_step_compatibility() -> None:
    # Compatibility smoke only: keep one minimal behavior check during migration.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        env = _build_legacy_env()
        obs, info = env.reset(seed=123)
        assert obs.shape == env.observation_space.shape
        assert "portfolio_value" in info

        obs, reward, terminated, truncated, info = env.step(1)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "num_trades" in info
