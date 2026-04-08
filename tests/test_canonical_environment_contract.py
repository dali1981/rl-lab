"""Canonical environment contract checks (Layer 2)."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from rl_trading_lab.domain.trading_domain import TradingDomain, TradingDomainConfig
from rl_trading_lab.infrastructure.adapters.gym_adapter import GymTradingEnvAdapter
from rl_trading_lab.infrastructure.adapters.market_data_adapter import ParquetMarketDataAdapter


ROOT = Path(__file__).resolve().parents[1]


def _build_canonical_env() -> GymTradingEnvAdapter:
    rows = 260
    rng = np.random.default_rng(11)
    close = 100 + np.cumsum(rng.normal(0.0, 0.3, rows))
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h"),
            "open": close + rng.normal(0.0, 0.2, rows),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 900 + rng.normal(0.0, 15.0, rows),
            "feature_1": rng.normal(0.0, 1.0, rows),
            "feature_2": rng.normal(0.0, 1.0, rows),
        }
    )
    market_data = ParquetMarketDataAdapter(df=df)
    domain = TradingDomain(
        market_data=market_data,
        observation_features=["close", "volume", "feature_1", "feature_2"],
        config=TradingDomainConfig(
            lookback_window=20,
            min_episode_length=60,
        ),
    )
    return GymTradingEnvAdapter(
        domain=domain,
        randomize_start=False,
        min_episode_length=60,
    )


def test_check_env_runs_on_canonical_environment_path() -> None:
    env = _build_canonical_env()
    check_env(env, warn=True, skip_render_check=True)


def test_core_entrypoints_do_not_import_legacy_trading_env() -> None:
    core_entrypoints = [
        ROOT / "experiments" / "train.py",
        ROOT / "run_pipeline.py",
        ROOT / "experiments" / "live_trading.py",
        ROOT / "examples" / "live_trading_example.py",
    ]
    for path in core_entrypoints:
        source = path.read_text(encoding="utf-8")
        assert "rl_trading_lab.environment.trading_env" not in source, (
            f"Core entrypoint must not import legacy TradingEnv: {path}"
        )


def test_only_compat_test_may_import_legacy_trading_env() -> None:
    allowed = {"test_legacy_trading_env_compat.py"}
    violations: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rl_trading_lab.environment.trading_env":
                violations.append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rl_trading_lab.environment.trading_env":
                        violations.append(f"{path}:{node.lineno}")
    assert not violations, (
        "Legacy TradingEnv import is allowed only in compatibility test file:\n"
        + "\n".join(violations)
    )
