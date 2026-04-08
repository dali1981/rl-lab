"""
Feature Engineering Pipeline - Create ML features from OHLCV bars.

This module creates features that match the training data format exactly.
Features must be computed identically to avoid distribution shift.

Training data features:
- ratio_sma_5_close: SMA(5) / close ratio
- ratio_sma_20_close: SMA(20) / close ratio
- ratio_range_close: (high - low) / close ratio
- fracdiff_0.4: Fractionally differentiated series (d=0.4)
- Z-score normalized versions of all indicators
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List
from pathlib import Path
import sys

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Feature engineering pipeline for trading data.

    Creates technical indicators and normalized features that match
    the format expected by trained RL models.

    Example:
        >>> pipeline = FeaturePipeline()
        >>> features_df = pipeline.transform(ohlcv_bars)
        >>> print(features_df.columns)
    """

    def __init__(
        self,
        lookback_periods: Dict[str, int] = None,
        fracdiff_d: float = 0.4,
        zscore_window: int = 252,  # ~1 year of daily bars
        compute_zscore: bool = True,
    ):
        """
        Initialize feature pipeline.

        Args:
            lookback_periods: Dict of indicator names to window sizes
                Default: {"sma_5": 5, "sma_20": 20}
            fracdiff_d: Fractional differentiation parameter (0 < d < 1)
                Higher d = more differencing, lower d = more memory
            zscore_window: Rolling window for z-score normalization
            compute_zscore: Whether to compute z-score normalized features
        """
        if lookback_periods is None:
            lookback_periods = {
                "sma_5": 5,
                "sma_20": 20,
            }

        self.lookback_periods = lookback_periods
        self.fracdiff_d = fracdiff_d
        self.zscore_window = zscore_window
        self.compute_zscore = compute_zscore

        # Import fractional differentiation from tools if available
        self._import_fracdiff()

        logger.info(f"Initialized FeaturePipeline with lookbacks: {lookback_periods}")

    def _import_fracdiff(self):
        """Import fractional differentiation function from tools."""
        try:
            # Try direct import if tools package installed
            from tools.indicators import fracdiff
            self.fracdiff_fn = fracdiff
            logger.info("Using fracdiff from installed tools package")
        except ImportError:
            # Try to add tools to path
            tools_path = Path(__file__).parents[4] / "tools"
            if tools_path.exists():
                sys.path.insert(0, str(tools_path))
                try:
                    from indicators import fracdiff
                    self.fracdiff_fn = fracdiff
                    logger.info(f"Using fracdiff from {tools_path}")
                except ImportError:
                    logger.warning("Could not import fracdiff, using fallback")
                    self.fracdiff_fn = None
            else:
                logger.warning(f"tools directory not found at {tools_path}")
                self.fracdiff_fn = None

    @property
    def feature_names(self) -> List[str]:
        """Names of engineered feature columns produced by this pipeline."""
        names: List[str] = []
        for name in self.lookback_periods:
            if "sma" in name:
                names.append(f"ratio_{name}_close")
        names.append("ratio_range_close")
        names.append(f"fracdiff_{self.fracdiff_d}")
        if self.compute_zscore:
            names.extend(f"{base}_zscore" for base in list(names))
        return names

    def transform(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
    ) -> pd.DataFrame:
        """
        Transform OHLCV bars into ML features.

        Args:
            df: DataFrame with OHLCV data
            price_col: Name of price column (default: 'close')
            high_col: Name of high price column
            low_col: Name of low price column

        Returns:
            DataFrame with original columns plus engineered features:
                - ratio_sma_5_close
                - ratio_sma_20_close
                - ratio_range_close
                - fracdiff_0.4
                - *_zscore versions if compute_zscore=True
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df

        # Validate required columns
        required = [price_col, high_col, low_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Create copy to avoid modifying original
        df_feat = df.copy()

        logger.info(f"Computing features for {len(df)} bars...")

        # 1. SMA Ratios
        for name, window in self.lookback_periods.items():
            if "sma" in name:
                feature_name = f"ratio_{name}_{price_col}"
                sma = df_feat[price_col].rolling(window=window, min_periods=1).mean()
                df_feat[feature_name] = sma / df_feat[price_col]
                logger.debug(f"Computed {feature_name}")

        # 2. Range Ratio
        df_feat[f'ratio_range_{price_col}'] = (
            (df_feat[high_col] - df_feat[low_col]) / df_feat[price_col]
        )

        # 3. Fractional Differentiation
        if self.fracdiff_fn is not None:
            try:
                df_feat[f'fracdiff_{self.fracdiff_d}'] = self.fracdiff_fn(
                    df_feat[price_col],
                    d=self.fracdiff_d
                )
                logger.debug(f"Computed fracdiff_{self.fracdiff_d}")
            except Exception as e:
                logger.warning(f"Error computing fracdiff: {e}")
                # Use fallback
                df_feat[f'fracdiff_{self.fracdiff_d}'] = self._fracdiff_fallback(
                    df_feat[price_col],
                    d=self.fracdiff_d
                )
        else:
            # Use fallback implementation
            df_feat[f'fracdiff_{self.fracdiff_d}'] = self._fracdiff_fallback(
                df_feat[price_col],
                d=self.fracdiff_d
            )

        # 4. Z-Score Normalization
        if self.compute_zscore:
            # Get list of indicator columns (exclude OHLCV)
            indicator_cols = [
                col for col in df_feat.columns
                if col.startswith('ratio_') or col.startswith('fracdiff_')
            ]

            for col in indicator_cols:
                zscore_col = f"{col}_zscore"
                df_feat[zscore_col] = self._compute_zscore(
                    df_feat[col],
                    window=self.zscore_window
                )
                logger.debug(f"Computed {zscore_col}")

        # Drop NaN rows from indicators
        initial_len = len(df_feat)
        df_feat = df_feat.dropna()
        dropped = initial_len - len(df_feat)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values (from lookback windows)")

        logger.info(f"Feature engineering complete: {len(df_feat.columns)} total columns")

        return df_feat

    def _compute_zscore(
        self,
        series: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling z-score normalization.

        Z-score = (value - rolling_mean) / rolling_std

        Args:
            series: Input series
            window: Rolling window size
            min_periods: Minimum periods required (default: window // 2)

        Returns:
            Z-score normalized series
        """
        if min_periods is None:
            min_periods = max(1, window // 2)

        rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=min_periods).std()

        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)

        zscore = (series - rolling_mean) / rolling_std

        return zscore

    def _fracdiff_fallback(
        self,
        series: pd.Series,
        d: float,
        threshold: float = 1e-5
    ) -> pd.Series:
        """
        Fallback fractional differentiation implementation.

        Uses fixed-width window approximation of fractional differentiation.

        Args:
            series: Input series (e.g., prices)
            d: Differentiation order (0 < d < 1)
            threshold: Threshold for weight truncation

        Returns:
            Fractionally differentiated series
        """
        if not 0 < d < 1:
            raise ValueError(f"d must be in (0, 1), got {d}")

        logger.info(f"Using fallback fracdiff implementation (d={d})")

        # Compute weights
        weights = self._compute_fracdiff_weights(d, threshold)
        width = len(weights)

        # Pad series for convolution
        series_padded = pd.Series([np.nan] * (width - 1) + list(series))

        # Apply fractional differentiation via convolution
        fracdiff = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= width - 1:
                # Get window of values
                window = series_padded.iloc[i:i+width].values
                # Apply weights (reversed for convolution)
                fracdiff.iloc[i] = np.dot(window[::-1], weights)
            else:
                fracdiff.iloc[i] = np.nan

        return fracdiff

    def _compute_fracdiff_weights(
        self,
        d: float,
        threshold: float = 1e-5
    ) -> np.ndarray:
        """
        Compute fractional differentiation weights.

        Weights follow: w_k = (-1)^k * binom(d, k)
        where binom(d, k) = d * (d-1) * ... * (d-k+1) / k!

        Args:
            d: Differentiation order
            threshold: Stop when |weight| < threshold

        Returns:
            Array of weights
        """
        weights = [1.0]  # w_0 = 1
        k = 1

        while True:
            # Compute next weight: w_k = w_{k-1} * (d - k + 1) / k * (-1)
            weight = -weights[-1] * (d - k + 1) / k
            weights.append(weight)

            if abs(weight) < threshold:
                break

            k += 1

            # Safety limit
            if k > 1000:
                logger.warning(f"Fracdiff weights exceeded 1000 terms, truncating")
                break

        return np.array(weights)

    def fit_save_stats(
        self,
        df: pd.DataFrame,
        save_path: str
    ) -> Dict:
        """
        Compute and save feature statistics for later normalization.

        This should be called on training data, then the saved stats
        can be used to normalize live data identically.

        Args:
            df: Training data DataFrame (after transform)
            save_path: Path to save statistics JSON

        Returns:
            Dictionary of feature statistics
        """
        import json

        # Get indicator columns
        indicator_cols = [
            col for col in df.columns
            if col.startswith('ratio_') or col.startswith('fracdiff_')
        ]

        stats = {}
        for col in indicator_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
            }

        # Save to file
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved feature statistics to {save_path}")

        return stats

    def load_and_apply_stats(
        self,
        df: pd.DataFrame,
        stats_path: str,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Load saved statistics and normalize features.

        Args:
            df: DataFrame with features to normalize
            stats_path: Path to saved statistics JSON
            method: Normalization method ('zscore', 'minmax', or 'none')

        Returns:
            DataFrame with normalized features
        """
        import json

        with open(stats_path, 'r') as f:
            stats = json.load(f)

        df_norm = df.copy()

        if method == 'zscore':
            for col, col_stats in stats.items():
                if col in df_norm.columns:
                    df_norm[col] = (df_norm[col] - col_stats['mean']) / col_stats['std']
        elif method == 'minmax':
            for col, col_stats in stats.items():
                if col in df_norm.columns:
                    range_val = col_stats['max'] - col_stats['min']
                    df_norm[col] = (df_norm[col] - col_stats['min']) / range_val
        elif method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(f"Applied {method} normalization using stats from {stats_path}")

        return df_norm
