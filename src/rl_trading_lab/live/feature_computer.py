"""
Real-time feature computation for live trading.

This module maintains a rolling window of bars and computes features incrementally
as new bars arrive, ensuring features match the training data format exactly.
"""

import logging
from typing import Dict, Optional, List
from collections import deque
import pandas as pd
import numpy as np
from pathlib import Path

from ..data.feature_pipeline import FeaturePipeline

logger = logging.getLogger(__name__)


class FeatureComputer:
    """
    Computes features in real-time from streaming bars.

    Maintains a rolling window of bars to compute indicators that require
    historical data (e.g., 20-bar SMA). Features are normalized using
    statistics saved from training data to avoid distribution shift.

    Example:
        >>> computer = FeatureComputer(
        ...     symbol="BTCUSDT",
        ...     lookback_window=100,
        ...     feature_stats_path="feature_stats.json"
        ... )
        >>>
        >>> # Add bars as they arrive
        >>> computer.add_bar(new_bar)
        >>>
        >>> # Get features when ready
        >>> if computer.is_ready():
        ...     features = computer.get_latest_features()
        ...     print(features)
    """

    def __init__(
        self,
        symbol: str,
        lookback_window: int = 100,
        feature_stats_path: Optional[str] = None,
        min_bars_required: int = 21,  # Need 20 bars for SMA(20) + 1 for current
    ):
        """
        Initialize the feature computer.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            lookback_window: Number of bars to keep in rolling window
            feature_stats_path: Path to saved feature statistics (mean/std)
            min_bars_required: Minimum bars needed before features can be computed
        """
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.min_bars_required = min_bars_required

        # Rolling window of bars (OHLCV)
        self.bar_window: deque = deque(maxlen=lookback_window)

        # Feature pipeline
        self.feature_pipeline = FeaturePipeline()

        # Load feature statistics if provided
        if feature_stats_path:
            self.feature_pipeline.load_stats(feature_stats_path)
            logger.info(f"Loaded feature statistics from {feature_stats_path}")

        # Track state
        self.total_bars_received = 0
        self.features_computed = 0

        logger.info(
            f"Initialized FeatureComputer for {symbol} "
            f"(lookback={lookback_window}, min_bars={min_bars_required})"
        )

    def add_bar(self, bar: pd.DataFrame) -> bool:
        """
        Add a new bar to the rolling window.

        Args:
            bar: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            True if features can now be computed (enough bars accumulated)
        """
        if bar.empty:
            return False

        # Validate bar has required columns
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in bar.columns]
        if missing_cols:
            logger.error(f"Bar missing required columns: {missing_cols}")
            return False

        # Add to window
        bar_dict = bar.iloc[0].to_dict()
        self.bar_window.append(bar_dict)
        self.total_bars_received += 1

        ready = self.is_ready()
        if ready and self.total_bars_received == self.min_bars_required:
            logger.info(
                f"{self.symbol}: Accumulated {self.min_bars_required} bars, "
                "feature computation ready"
            )

        return ready

    def is_ready(self) -> bool:
        """Check if enough bars have been accumulated to compute features."""
        return len(self.bar_window) >= self.min_bars_required

    def get_latest_features(self) -> Optional[pd.DataFrame]:
        """
        Compute and return features for the current state.

        Returns:
            DataFrame with features, or None if not enough bars
        """
        if not self.is_ready():
            logger.warning(
                f"{self.symbol}: Not enough bars for features "
                f"({len(self.bar_window)}/{self.min_bars_required})"
            )
            return None

        try:
            # Convert window to DataFrame
            bars_df = pd.DataFrame(list(self.bar_window))

            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(bars_df["timestamp"]):
                bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])

            # Compute features using the pipeline
            features_df = self.feature_pipeline.transform(bars_df)

            self.features_computed += 1

            # Return only the latest row (most recent features)
            return features_df.iloc[[-1]].copy()

        except Exception as e:
            logger.error(f"Error computing features for {self.symbol}: {e}")
            return None

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        dummy_bar = pd.DataFrame([{
            "timestamp": pd.Timestamp.now(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }] * self.min_bars_required)

        features = self.feature_pipeline.transform(dummy_bar)
        return features.columns.tolist()

    def reset(self):
        """Clear the bar window and reset state."""
        self.bar_window.clear()
        self.total_bars_received = 0
        self.features_computed = 0
        logger.info(f"Reset FeatureComputer for {self.symbol}")

    def get_stats(self) -> Dict:
        """Get statistics about feature computation."""
        return {
            "symbol": self.symbol,
            "bars_in_window": len(self.bar_window),
            "total_bars_received": self.total_bars_received,
            "features_computed": self.features_computed,
            "is_ready": self.is_ready(),
            "lookback_window": self.lookback_window,
            "min_bars_required": self.min_bars_required,
        }


class MultiSymbolFeatureComputer:
    """
    Manages feature computation for multiple symbols independently.

    Each symbol maintains its own rolling window and computes features
    independently using shared or symbol-specific feature statistics.

    Example:
        >>> manager = MultiSymbolFeatureComputer(
        ...     symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        ...     lookback_window=100
        ... )
        >>>
        >>> # Add bars as they arrive
        >>> manager.add_bar("BTCUSDT", btc_bar)
        >>> manager.add_bar("ETHUSDT", eth_bar)
        >>>
        >>> # Get features for specific symbol
        >>> btc_features = manager.get_features("BTCUSDT")
    """

    def __init__(
        self,
        symbols: List[str],
        lookback_window: int = 100,
        feature_stats_dir: Optional[str] = None,
        shared_stats: bool = True,
    ):
        """
        Initialize multi-symbol feature computer.

        Args:
            symbols: List of trading symbols
            lookback_window: Number of bars to keep per symbol
            feature_stats_dir: Directory with feature statistics files
            shared_stats: If True, use shared stats; if False, use symbol-specific stats
        """
        self.symbols = symbols
        self.lookback_window = lookback_window
        self.shared_stats = shared_stats

        # Create feature computer for each symbol
        self.computers: Dict[str, FeatureComputer] = {}

        for symbol in symbols:
            # Determine stats file path
            stats_path = None
            if feature_stats_dir:
                stats_dir = Path(feature_stats_dir)
                if shared_stats:
                    stats_path = str(stats_dir / "feature_stats.json")
                else:
                    stats_path = str(stats_dir / f"feature_stats_{symbol}.json")

            self.computers[symbol] = FeatureComputer(
                symbol=symbol,
                lookback_window=lookback_window,
                feature_stats_path=stats_path,
            )

        logger.info(
            f"Initialized MultiSymbolFeatureComputer for {len(symbols)} symbols"
        )

    def add_bar(self, symbol: str, bar: pd.DataFrame) -> bool:
        """
        Add a bar for a specific symbol.

        Args:
            symbol: Trading symbol
            bar: Bar DataFrame

        Returns:
            True if features can be computed for this symbol
        """
        if symbol not in self.computers:
            logger.error(f"Unknown symbol: {symbol}")
            return False

        return self.computers[symbol].add_bar(bar)

    def get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get latest features for a symbol."""
        if symbol not in self.computers:
            logger.error(f"Unknown symbol: {symbol}")
            return None

        return self.computers[symbol].get_latest_features()

    def is_ready(self, symbol: str) -> bool:
        """Check if features can be computed for a symbol."""
        if symbol not in self.computers:
            return False
        return self.computers[symbol].is_ready()

    def all_ready(self) -> bool:
        """Check if all symbols are ready for feature computation."""
        return all(computer.is_ready() for computer in self.computers.values())

    def get_all_features(self) -> Dict[str, pd.DataFrame]:
        """
        Get features for all symbols that are ready.

        Returns:
            Dict mapping symbol to features DataFrame
        """
        features = {}
        for symbol, computer in self.computers.items():
            if computer.is_ready():
                feat = computer.get_latest_features()
                if feat is not None:
                    features[symbol] = feat
        return features

    def reset(self, symbol: Optional[str] = None):
        """
        Reset feature computers.

        Args:
            symbol: If specified, reset only this symbol; otherwise reset all
        """
        if symbol:
            if symbol in self.computers:
                self.computers[symbol].reset()
        else:
            for computer in self.computers.values():
                computer.reset()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all symbols."""
        return {
            symbol: computer.get_stats()
            for symbol, computer in self.computers.items()
        }