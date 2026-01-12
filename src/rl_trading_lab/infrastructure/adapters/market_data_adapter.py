"""
Market Data Adapter - DataFrame-based implementation of MarketDataPort.

This adapter wraps a pandas DataFrame to provide market data to the domain
without exposing pandas internals.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from rl_trading_lab.domain.ports.market_data import MarketDataPort
from rl_trading_lab.domain.value_objects.bar import Bar
from rl_trading_lab.domain.value_objects.feature_window import FeatureWindow

logger = logging.getLogger(__name__)


class ParquetMarketDataAdapter(MarketDataPort):
    """
    Adapts pandas DataFrame to MarketDataPort interface.

    This adapter:
    - Keeps pandas in the infrastructure layer
    - Provides immutable value objects to the domain
    - Validates data on construction

    Example:
        >>> df = pd.read_parquet("data.parquet")
        >>> adapter = ParquetMarketDataAdapter(df)
        >>> bar = adapter.get_bar(0)
        >>> price = adapter.get_price(100, "close")
    """

    # Required OHLCV columns
    REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")

    def __init__(
        self,
        df: pd.DataFrame,
        price_column: str = "close",
        timestamp_column: str = "timestamp",
    ):
        """
        Initialize the adapter with a DataFrame.

        Args:
            df: DataFrame with OHLCV and feature data
            price_column: Default price column name
            timestamp_column: Timestamp column name (optional)

        Raises:
            ValueError: If required columns are missing
        """
        self._df = df
        self._price_column = price_column
        self._timestamp_column = timestamp_column
        self._validate()

        # Cache feature names (excluding non-feature columns)
        self._feature_names = self._compute_feature_names()

        logger.debug(
            f"ParquetMarketDataAdapter initialized: {len(df)} bars, "
            f"{len(self._feature_names)} features"
        )

    def _validate(self) -> None:
        """Validate DataFrame has required columns."""
        missing = [c for c in self.REQUIRED_COLUMNS if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {missing}. "
                f"Available columns: {sorted(self._df.columns.tolist())}"
            )

        if len(self._df) == 0:
            raise ValueError("DataFrame is empty")

    def _compute_feature_names(self) -> Tuple[str, ...]:
        """Compute list of feature column names."""
        exclude = {"timestamp", "date", "bar_id", "index"}
        numeric_cols = self._df.select_dtypes(include=["float64", "int64"]).columns
        features = [c for c in numeric_cols if c not in exclude]
        return tuple(sorted(features))

    def get_price(self, index: int, column: str = "close") -> float:
        """Get price at specific index."""
        if index < 0 or index >= len(self._df):
            raise IndexError(f"Index {index} out of bounds [0, {len(self._df)})")
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found")
        return float(self._df.iloc[index][column])

    def get_bar(self, index: int) -> Bar:
        """Get OHLCV bar at index."""
        if index < 0 or index >= len(self._df):
            raise IndexError(f"Index {index} out of bounds [0, {len(self._df)})")

        row = self._df.iloc[index]

        # Get timestamp if available
        timestamp = None
        if self._timestamp_column in self._df.columns:
            ts = row[self._timestamp_column]
            if pd.notna(ts):
                if isinstance(ts, datetime):
                    timestamp = ts
                elif isinstance(ts, pd.Timestamp):
                    timestamp = ts.to_pydatetime()

        return Bar(
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
            timestamp=timestamp,
        )

    def get_feature_window(
        self,
        start_index: int,
        end_index: int,
        features: List[str],
    ) -> FeatureWindow:
        """Get window of features for observation."""
        # Validate indices
        if start_index < 0:
            raise IndexError(f"Start index {start_index} cannot be negative")
        if end_index > len(self._df):
            raise IndexError(f"End index {end_index} exceeds data length {len(self._df)}")
        if start_index >= end_index:
            raise IndexError(f"Start index {start_index} must be less than end index {end_index}")

        # Validate features exist
        missing = [f for f in features if f not in self._df.columns]
        if missing:
            raise KeyError(f"Features not found: {missing}")

        # Extract window
        window_df = self._df.iloc[start_index:end_index][features]

        # Convert to nested tuples (immutable)
        values = tuple(tuple(float(v) for v in row) for row in window_df.values)

        return FeatureWindow(
            values=values,
            feature_names=tuple(features),
        )

    def get_timestamp(self, index: int) -> Optional[datetime]:
        """Get timestamp at index if available."""
        if self._timestamp_column not in self._df.columns:
            return None

        if index < 0 or index >= len(self._df):
            raise IndexError(f"Index {index} out of bounds [0, {len(self._df)})")

        ts = self._df.iloc[index][self._timestamp_column]
        if pd.isna(ts):
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        return None

    def __len__(self) -> int:
        """Total number of bars available."""
        return len(self._df)

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """All available feature column names."""
        return self._feature_names

    @property
    def columns(self) -> List[str]:
        """All column names (for debugging)."""
        return self._df.columns.tolist()

    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the date range of the data.

        Returns:
            Tuple of (start_date, end_date) or (None, None) if no timestamps
        """
        if self._timestamp_column not in self._df.columns:
            return (None, None)

        start = self.get_timestamp(0)
        end = self.get_timestamp(len(self._df) - 1)
        return (start, end)
