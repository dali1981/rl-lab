"""
Position Value Object - Immutable representation of a trading position.

A position represents a current holding in a trading instrument.
Positions are immutable - any change creates a new Position instance.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class Position:
    """
    Immutable trading position.

    Attributes:
        size: Position size (positive=long, negative=short, 0=flat)
        entry_price: Average entry price
        entry_bar: Bar index when position was opened

    Example:
        >>> pos = Position.open_long(size=1.0, price=100.0, bar=42)
        >>> pos.is_long
        True
        >>> pos.unrealized_pnl(current_price=110.0)
        10.0
        >>> pos.return_pct(current_price=110.0)
        0.1
    """

    size: float = 0.0
    entry_price: float = 0.0
    entry_bar: int = 0

    # --- Factory Methods ---

    @classmethod
    def flat(cls) -> "Position":
        """Create a flat (no position) state."""
        return cls(size=0.0, entry_price=0.0, entry_bar=0)

    @classmethod
    def open_long(cls, size: float, price: float, bar: int) -> "Position":
        """
        Create a long position.

        Args:
            size: Position size (will be made positive)
            price: Entry price
            bar: Entry bar index

        Returns:
            New Position with positive size
        """
        if size <= 0:
            raise ValueError(f"Long position size must be positive, got {size}")
        return cls(size=abs(size), entry_price=price, entry_bar=bar)

    @classmethod
    def open_short(cls, size: float, price: float, bar: int) -> "Position":
        """
        Create a short position.

        Args:
            size: Position size (will be made negative)
            price: Entry price
            bar: Entry bar index

        Returns:
            New Position with negative size
        """
        if size <= 0:
            raise ValueError(f"Short position size must be positive, got {size}")
        return cls(size=-abs(size), entry_price=price, entry_bar=bar)

    # --- Properties ---

    @property
    def is_flat(self) -> bool:
        """True if no position (size == 0)."""
        return self.size == 0.0

    @property
    def is_long(self) -> bool:
        """True if long position (size > 0)."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """True if short position (size < 0)."""
        return self.size < 0

    @property
    def direction(self) -> int:
        """Position direction: 1 (long), -1 (short), or 0 (flat)."""
        if self.size > 0:
            return 1
        elif self.size < 0:
            return -1
        return 0

    @property
    def abs_size(self) -> float:
        """Absolute position size."""
        return abs(self.size)

    # --- Calculations ---

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current price.

        For long positions: (current - entry) * size
        For short positions: (entry - current) * |size| = (current - entry) * size

        Args:
            current_price: Current market price

        Returns:
            Unrealized profit/loss (positive = profit, negative = loss)
        """
        if self.is_flat:
            return 0.0
        return self.size * (current_price - self.entry_price)

    def return_pct(self, current_price: float) -> float:
        """
        Calculate return as percentage of entry value.

        Args:
            current_price: Current market price

        Returns:
            Return percentage (0.1 = 10% gain)
        """
        if self.is_flat or self.entry_price == 0:
            return 0.0
        pnl = self.unrealized_pnl(current_price)
        entry_value = abs(self.size * self.entry_price)
        return pnl / entry_value if entry_value > 0 else 0.0

    def market_value(self, current_price: float) -> float:
        """
        Calculate current market value of position.

        Args:
            current_price: Current market price

        Returns:
            Market value (can be negative for short positions)
        """
        return self.size * current_price

    def holding_period(self, current_bar: int) -> int:
        """
        Calculate how many bars position has been held.

        Args:
            current_bar: Current bar index

        Returns:
            Number of bars held (0 if flat)
        """
        if self.is_flat:
            return 0
        return current_bar - self.entry_bar

    # --- String Representation ---

    def __str__(self) -> str:
        if self.is_flat:
            return "Position(FLAT)"
        direction = "LONG" if self.is_long else "SHORT"
        return f"Position({direction} {self.abs_size:.4f} @ {self.entry_price:.2f})"

    def __repr__(self) -> str:
        return f"Position(size={self.size}, entry_price={self.entry_price}, entry_bar={self.entry_bar})"
