"""
Trade Value Objects - Immutable trade records.

CompletedTrade represents a round-trip trade (open + close).
These are immutable facts that can be used for analytics and audit.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class TradeSide(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True, slots=True)
class CompletedTrade:
    """
    Immutable record of a completed round-trip trade.

    A completed trade represents a position that was opened and then closed.
    This is an immutable fact suitable for trade history and analytics.

    Attributes:
        trade_id: Unique identifier for this trade
        side: Trade direction (LONG or SHORT)
        entry_price: Price at which position was opened
        exit_price: Price at which position was closed
        position_size: Size of the position (always positive)
        entry_bar: Bar index when opened
        exit_bar: Bar index when closed
        entry_commission: Commission paid on entry
        exit_commission: Commission paid on exit
        entry_timestamp: Timestamp when opened (optional)
        exit_timestamp: Timestamp when closed (optional)

    Example:
        >>> trade = CompletedTrade(
        ...     trade_id=1,
        ...     side=TradeSide.LONG,
        ...     entry_price=100.0,
        ...     exit_price=110.0,
        ...     position_size=1.0,
        ...     entry_bar=0,
        ...     exit_bar=10,
        ...     entry_commission=0.1,
        ...     exit_commission=0.11,
        ... )
        >>> trade.gross_pnl
        10.0
        >>> trade.net_pnl
        9.79
        >>> trade.return_pct
        0.0979
    """

    trade_id: int
    side: TradeSide
    entry_price: float
    exit_price: float
    position_size: float
    entry_bar: int
    exit_bar: int
    entry_commission: float = 0.0
    exit_commission: float = 0.0
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Validate trade data."""
        if self.position_size <= 0:
            raise ValueError(f"Position size must be positive, got {self.position_size}")
        if self.entry_price <= 0:
            raise ValueError(f"Entry price must be positive, got {self.entry_price}")
        if self.exit_price <= 0:
            raise ValueError(f"Exit price must be positive, got {self.exit_price}")
        if self.exit_bar < self.entry_bar:
            raise ValueError(
                f"Exit bar ({self.exit_bar}) cannot be before entry bar ({self.entry_bar})"
            )

    @property
    def gross_pnl(self) -> float:
        """
        Profit/Loss before commissions.

        For LONG: (exit - entry) * size
        For SHORT: (entry - exit) * size
        """
        if self.side == TradeSide.LONG:
            return (self.exit_price - self.entry_price) * self.position_size
        else:  # SHORT
            return (self.entry_price - self.exit_price) * self.position_size

    @property
    def total_commission(self) -> float:
        """Total commission paid (entry + exit)."""
        return self.entry_commission + self.exit_commission

    @property
    def net_pnl(self) -> float:
        """Profit/Loss after commissions."""
        return self.gross_pnl - self.total_commission

    @property
    def return_pct(self) -> float:
        """
        Return as percentage of entry value.

        Calculated as net_pnl / entry_value.
        """
        entry_value = self.position_size * self.entry_price
        if entry_value == 0:
            return 0.0
        return self.net_pnl / entry_value

    @property
    def hold_bars(self) -> int:
        """Number of bars position was held."""
        return self.exit_bar - self.entry_bar

    @property
    def is_winner(self) -> bool:
        """True if trade was profitable (net_pnl > 0)."""
        return self.net_pnl > 0

    @property
    def is_loser(self) -> bool:
        """True if trade was unprofitable (net_pnl < 0)."""
        return self.net_pnl < 0

    @property
    def entry_value(self) -> float:
        """Total value at entry (size * entry_price)."""
        return self.position_size * self.entry_price

    @property
    def exit_value(self) -> float:
        """Total value at exit (size * exit_price)."""
        return self.position_size * self.exit_price

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the trade
        """
        return {
            "trade_id": self.trade_id,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "entry_bar": self.entry_bar,
            "exit_bar": self.exit_bar,
            "entry_commission": self.entry_commission,
            "exit_commission": self.exit_commission,
            "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "hold_bars": self.hold_bars,
        }

    def __str__(self) -> str:
        return (
            f"Trade #{self.trade_id}: {self.side.value} "
            f"{self.position_size:.4f} @ {self.entry_price:.2f} -> {self.exit_price:.2f} "
            f"(PnL: ${self.net_pnl:.2f}, {self.return_pct:.2%})"
        )
