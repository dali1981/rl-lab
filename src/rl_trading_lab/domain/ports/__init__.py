"""
Domain Ports - Interfaces for external dependencies.

Ports define the contracts that infrastructure adapters must implement.
This follows the Ports & Adapters (Hexagonal) architecture pattern.

The domain defines what it needs; infrastructure provides implementations.
"""

from rl_trading_lab.domain.ports.market_data import MarketDataPort

__all__ = [
    "MarketDataPort",
]
