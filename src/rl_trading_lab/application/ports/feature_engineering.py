"""
FeatureEngineeringPort - Interface for feature transformation pipelines.

This port decouples the feature engineering strategy from the training
pipeline, allowing different feature sets for different asset classes.
"""

from typing import List, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FeatureEngineeringPort(Protocol):
    """
    Port for transforming raw OHLCV data into ML features.

    Implementations handle:
    - Computing technical indicators
    - Normalization (z-score, min-max, etc.)
    - Feature selection

    Example implementations:
    - CryptoFeaturePipeline: SMA ratios, fracdiff, z-scores (current default)
    - PassthroughFeatures: No-op for pre-computed features
    - EquityFeaturePipeline: RSI, MACD, Bollinger Bands, etc.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature columns to a DataFrame.

        Args:
            df: DataFrame with at minimum OHLCV columns

        Returns:
            DataFrame with original columns plus engineered features
        """
        ...

    @property
    def feature_names(self) -> List[str]:
        """Names of features this pipeline produces (excluding raw OHLCV)."""
        ...


class PassthroughFeatures:
    """
    No-op feature engineering for pre-computed feature sets.

    Use this when the data already contains all needed features
    (e.g., loaded from a parquet file that was pre-processed).

    Example:
        >>> pipeline = PassthroughFeatures(feature_names=["rsi", "macd", "bb_width"])
        >>> df_out = pipeline.transform(df)  # returns df unchanged
        >>> pipeline.feature_names  # ["rsi", "macd", "bb_width"]
    """

    def __init__(self, feature_names: List[str]):
        self._feature_names = list(feature_names)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame unchanged."""
        missing = [f for f in self._feature_names if f not in df.columns]
        if missing:
            raise ValueError(
                f"Expected features not found in data: {missing}. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )
        return df

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)
