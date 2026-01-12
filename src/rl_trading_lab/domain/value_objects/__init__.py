"""
Value Objects - Immutable domain concepts.

Value objects are defined by their attributes, not by identity.
Two value objects with the same attributes are considered equal.

All value objects in this module are immutable (frozen dataclasses).
"""

from rl_trading_lab.domain.value_objects.bar import Bar
from rl_trading_lab.domain.value_objects.position import Position
from rl_trading_lab.domain.value_objects.trade import CompletedTrade, TradeSide
from rl_trading_lab.domain.value_objects.feature_window import FeatureWindow

__all__ = [
    "Bar",
    "Position",
    "CompletedTrade",
    "TradeSide",
    "FeatureWindow",
]
