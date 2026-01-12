"""
Domain Layer - Pure business logic with no external dependencies.

This layer contains:
- Value Objects: Immutable domain concepts (Position, Bar, CompletedTrade)
- Ports: Interfaces for external dependencies (MarketDataPort)
- Services: Stateless domain operations (PositionSizing, RewardCalculation, RiskManagement)
- TradingDomain: Core domain class orchestrating trading logic
- Exceptions: Domain-specific errors

Dependencies flow inward: Infrastructure -> Application -> Domain
The domain layer has NO dependencies on external frameworks (pandas, gymnasium, etc.)
"""

from rl_trading_lab.domain.exceptions import (
    DomainError,
    InsufficientFundsError,
    InvalidPositionError,
    InvalidOrderError,
)
from rl_trading_lab.domain.trading_domain import (
    TradingDomain,
    TradingDomainConfig,
    TradingState,
    StepResult,
    OrderIntent,
)

__all__ = [
    # Exceptions
    "DomainError",
    "InsufficientFundsError",
    "InvalidPositionError",
    "InvalidOrderError",
    # Trading Domain
    "TradingDomain",
    "TradingDomainConfig",
    "TradingState",
    "StepResult",
    "OrderIntent",
]
