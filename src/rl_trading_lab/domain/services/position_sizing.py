"""
Position Sizing Service - Domain service for calculating position sizes.

This service encapsulates the logic for determining how large a position
to take based on available capital, risk parameters, and signal strength.

Different strategies can be implemented by subclassing PositionSizingService.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


class PositionSizingService(Protocol):
    """
    Protocol for position sizing strategies.

    Position sizing determines how much capital to allocate to a trade.
    This is a critical risk management decision that affects:
    - Maximum potential loss per trade
    - Portfolio volatility
    - Compounding effects over time
    """

    def calculate_size(
        self,
        available_cash: float,
        current_price: float,
        signal_strength: float,
        max_position_pct: float,
    ) -> float:
        """
        Calculate position size for a trade.

        Args:
            available_cash: Cash available for trading
            current_price: Current asset price
            signal_strength: Signal strength (-1 to 1, where sign indicates direction)
            max_position_pct: Maximum position as fraction of available cash

        Returns:
            Position size (positive for long, negative for short)
        """
        ...


@dataclass(frozen=True)
class PositionSizingConfig:
    """Configuration for position sizing."""

    max_position_pct: float = 0.95
    min_position_value: float = 10.0  # Minimum trade value in base currency


class FixedPercentagePositionSizing(PositionSizingService):
    """
    Size positions as fixed percentage of available cash.

    This is the simplest position sizing strategy:
    - Uses a fixed percentage of available capital
    - Signal strength scales the position (optional)
    - Respects minimum position value

    Example:
        >>> sizing = FixedPercentagePositionSizing()
        >>> size = sizing.calculate_size(
        ...     available_cash=10000,
        ...     current_price=100,
        ...     signal_strength=1.0,
        ...     max_position_pct=0.95
        ... )
        >>> print(f"Position size: {size}")  # 95 units
    """

    def __init__(self, config: PositionSizingConfig = None):
        """
        Initialize the position sizing service.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        self._config = config or PositionSizingConfig()

    def calculate_size(
        self,
        available_cash: float,
        current_price: float,
        signal_strength: float,
        max_position_pct: float,
    ) -> float:
        """
        Calculate position size as percentage of available cash.

        Position value = available_cash * max_position_pct * |signal_strength|
        Position size = position_value / current_price
        Sign determined by signal_strength sign

        Args:
            available_cash: Cash available for trading
            current_price: Current asset price
            signal_strength: Signal strength (-1 to 1)
            max_position_pct: Maximum position as fraction of cash

        Returns:
            Position size (signed based on signal direction)
        """
        if signal_strength == 0 or available_cash <= 0 or current_price <= 0:
            return 0.0

        # Calculate maximum position value
        max_value = available_cash * max_position_pct

        # Scale by signal strength (absolute value)
        position_value = max_value * abs(signal_strength)

        # Check minimum position value
        if position_value < self._config.min_position_value:
            return 0.0

        # Calculate size in units
        size = position_value / current_price

        # Apply direction from signal
        if signal_strength < 0:
            size = -size

        return size


class KellyCriterionPositionSizing(PositionSizingService):
    """
    Position sizing based on Kelly Criterion.

    The Kelly Criterion calculates optimal bet size based on:
    - Win probability
    - Win/loss ratio

    Formula: f* = (bp - q) / b
    Where:
    - f* = fraction of bankroll to bet
    - b = odds received on bet (win/loss ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)

    Note: This implementation uses a fractional Kelly (half-Kelly by default)
    to reduce volatility.
    """

    def __init__(
        self,
        win_rate: float = 0.5,
        win_loss_ratio: float = 1.5,
        kelly_fraction: float = 0.5,
        config: PositionSizingConfig = None,
    ):
        """
        Initialize Kelly Criterion position sizing.

        Args:
            win_rate: Estimated probability of winning (0 to 1)
            win_loss_ratio: Average win / average loss
            kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
            config: Optional configuration
        """
        self._win_rate = win_rate
        self._win_loss_ratio = win_loss_ratio
        self._kelly_fraction = kelly_fraction
        self._config = config or PositionSizingConfig()

    def calculate_size(
        self,
        available_cash: float,
        current_price: float,
        signal_strength: float,
        max_position_pct: float,
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        if signal_strength == 0 or available_cash <= 0 or current_price <= 0:
            return 0.0

        # Calculate Kelly fraction
        p = self._win_rate
        q = 1 - p
        b = self._win_loss_ratio

        kelly_pct = (b * p - q) / b

        # Apply fractional Kelly and cap at max_position_pct
        if kelly_pct <= 0:
            return 0.0

        position_pct = min(kelly_pct * self._kelly_fraction, max_position_pct)

        # Scale by signal strength
        position_value = available_cash * position_pct * abs(signal_strength)

        # Check minimum
        if position_value < self._config.min_position_value:
            return 0.0

        # Calculate size
        size = position_value / current_price

        # Apply direction
        if signal_strength < 0:
            size = -size

        return size

    def update_statistics(self, win_rate: float, win_loss_ratio: float):
        """
        Update win rate and win/loss ratio based on recent trades.

        Args:
            win_rate: New estimated win rate
            win_loss_ratio: New win/loss ratio
        """
        self._win_rate = win_rate
        self._win_loss_ratio = win_loss_ratio
