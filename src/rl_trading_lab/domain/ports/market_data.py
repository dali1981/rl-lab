"""
Market Data Port - Interface for accessing market data.

This port abstracts the data source from the domain, allowing:
- DataFrame-based adapters for backtesting
- Live streaming adapters for real-time trading
- Mock adapters for testing

The domain never knows about pandas, parquet, or any specific storage format.
"""

from typing import Protocol, List, Optional, Tuple
from datetime import datetime

from rl_trading_lab.domain.value_objects.bar import Bar
from rl_trading_lab.domain.value_objects.feature_window import FeatureWindow


class MarketDataPort(Protocol):
    """
    Port for accessing market data.

    Implementations must provide price and feature data without
    exposing the underlying storage mechanism.

    This follows the Dependency Inversion Principle:
    - High-level domain depends on this abstraction
    - Low-level infrastructure implements this abstraction
    """

    def get_price(self, index: int, column: str = "close") -> float:
        """
        Get price at specific index.

        Args:
            index: Bar index (0-based)
            column: Price column name (default: "close")

        Returns:
            Price value as float

        Raises:
            IndexError: If index is out of bounds
            KeyError: If column doesn't exist
        """
        ...

    def get_bar(self, index: int) -> Bar:
        """
        Get OHLCV bar at index.

        Args:
            index: Bar index (0-based)

        Returns:
            Bar value object with OHLCV data

        Raises:
            IndexError: If index is out of bounds
        """
        ...

    def get_feature_window(
        self,
        start_index: int,
        end_index: int,
        features: List[str],
    ) -> FeatureWindow:
        """
        Get window of features for observation.

        Args:
            start_index: Start index (inclusive)
            end_index: End index (exclusive)
            features: List of feature column names

        Returns:
            FeatureWindow containing the requested features

        Raises:
            IndexError: If indices are out of bounds
            KeyError: If any feature doesn't exist
        """
        ...

    def get_timestamp(self, index: int) -> Optional[datetime]:
        """
        Get timestamp at index if available.

        Args:
            index: Bar index (0-based)

        Returns:
            Timestamp or None if not available
        """
        ...

    def __len__(self) -> int:
        """Total number of bars available."""
        ...

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """All available feature column names."""
        ...
