"""
Bar Value Object - Immutable OHLCV price bar.

Represents a single candlestick/bar of price data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class Bar:
    """
    Immutable OHLCV price bar.

    Attributes:
        open: Opening price
        high: Highest price in period
        low: Lowest price in period
        close: Closing price
        volume: Trading volume
        timestamp: Bar timestamp (optional)

    Example:
        >>> bar = Bar(open=100.0, high=105.0, low=99.0, close=103.0, volume=1000.0)
        >>> bar.range
        6.0
        >>> bar.is_bullish
        True
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        """Validate bar data."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= open and close")
        if self.volume < 0:
            raise ValueError(f"Volume ({self.volume}) cannot be negative")

    @property
    def range(self) -> float:
        """Price range (high - low)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Candle body size (absolute difference between open and close)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True if close > open (green candle)."""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """True if close < open (red candle)."""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """True if open == close (neutral candle)."""
        return self.close == self.open

    @property
    def upper_shadow(self) -> float:
        """Upper shadow/wick length."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Lower shadow/wick length."""
        return min(self.open, self.close) - self.low

    @property
    def typical_price(self) -> float:
        """Typical price: (high + low + close) / 3."""
        return (self.high + self.low + self.close) / 3

    @property
    def vwap_approx(self) -> float:
        """Approximate VWAP using typical price (actual VWAP needs tick data)."""
        return self.typical_price
