"""Factory port contracts for environment assembly in application services."""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym

from rl_trading_lab.domain.ports.market_data import MarketDataPort
from rl_trading_lab.domain.trading_domain import TradingDomain


class MarketDataAdapterFactory(Protocol):
    """Build a MarketDataPort implementation from loaded data."""

    def __call__(self, data_frame: Any) -> MarketDataPort:
        ...


class EnvAdapterFactory(Protocol):
    """Build a gym-compatible environment from an assembled domain model."""

    def __call__(
        self,
        domain: TradingDomain,
        randomize_start: bool,
        min_episode_length: int,
    ) -> gym.Env:
        ...
