"""
DataLoaderPort - Interface for loading market data.

This port defines the contract for data loading implementations.
It abstracts away the data source (parquet files, databases, APIs)
from the application logic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import pandas as pd


class DataLoaderPort(Protocol):
    """
    Port for loading market data.

    Implementations handle:
    - Loading data from various sources (parquet, CSV, databases)
    - Splitting data into train/validation/test sets
    - Basic data validation

    Example implementations:
    - ParquetDataLoader: Loads from parquet files
    - CSVDataLoader: Loads from CSV files
    - DatabaseDataLoader: Loads from SQL databases
    """

    def load(
        self,
        data_path: Path,
        mode: str = "train",
    ) -> pd.DataFrame:
        """
        Load data for the specified mode.

        Args:
            data_path: Path to the data source
            mode: One of 'train', 'eval', 'test'

        Returns:
            DataFrame with market data
        """
        ...

    def load_with_splits(
        self,
        data_path: Path,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data and split into train/val/test sets.

        Args:
            data_path: Path to the data source
            val_split: Fraction for validation set
            test_split: Fraction for test set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        ...

    def get_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names from data.

        Args:
            df: DataFrame to inspect

        Returns:
            List of feature column names
        """
        ...


class ParquetDataLoader:
    """
    Implementation of DataLoaderPort for parquet files.

    Handles:
    - Loading parquet files with pandas
    - Chronological train/val/test splitting
    - Feature detection
    """

    def __init__(
        self,
        val_split: float = 0.1,
        test_split: float = 0.1,
        required_columns: Optional[List[str]] = None,
    ):
        """
        Initialize the parquet data loader.

        Args:
            val_split: Fraction of data for validation
            test_split: Fraction of data for test
            required_columns: Columns that must be present
        """
        self._val_split = val_split
        self._test_split = test_split
        self._required_columns = required_columns or ["open", "high", "low", "close", "volume"]
        self._cached_splits: Optional[Dict[str, pd.DataFrame]] = None
        self._cached_path: Optional[Path] = None

    def load(
        self,
        data_path: Path,
        mode: str = "train",
    ) -> pd.DataFrame:
        """
        Load data for the specified mode.

        Uses caching to avoid reloading and re-splitting on every call.
        """
        data_path = Path(data_path)

        # Check cache
        if self._cached_path != data_path or self._cached_splits is None:
            self._load_and_split(data_path)

        if mode not in self._cached_splits:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: train, eval, test")

        return self._cached_splits[mode].copy()

    def load_with_splits(
        self,
        data_path: Path,
        val_split: Optional[float] = None,
        test_split: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and split data, optionally overriding default splits.
        """
        data_path = Path(data_path)

        # Use provided splits or defaults
        val_pct = val_split if val_split is not None else self._val_split
        test_pct = test_split if test_split is not None else self._test_split

        # Force reload if splits changed
        if val_pct != self._val_split or test_pct != self._test_split:
            self._val_split = val_pct
            self._test_split = test_pct
            self._cached_splits = None

        if self._cached_path != data_path or self._cached_splits is None:
            self._load_and_split(data_path)

        return (
            self._cached_splits["train"].copy(),
            self._cached_splits["eval"].copy(),
            self._cached_splits["test"].copy(),
        )

    def get_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get numeric feature columns (excluding timestamp and bar_id).
        """
        exclude_cols = {"timestamp", "bar_id", "date", "datetime"}
        features = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in ["float64", "int64", "float32", "int32"]
        ]
        return features

    def _load_and_split(self, data_path: Path) -> None:
        """
        Load parquet file and split chronologically.
        """
        # Load data
        df = pd.read_parquet(data_path)

        # Validate required columns
        missing = set(self._required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Calculate split indices (chronological)
        n = len(df)
        test_start = int(n * (1 - self._test_split))
        val_start = int(n * (1 - self._test_split - self._val_split))

        # Split data
        train_df = df.iloc[:val_start].copy()
        val_df = df.iloc[val_start:test_start].copy()
        test_df = df.iloc[test_start:].copy()

        # Cache results
        self._cached_splits = {
            "train": train_df,
            "eval": val_df,
            "test": test_df,
        }
        self._cached_path = data_path

    def clear_cache(self) -> None:
        """Clear the cached data."""
        self._cached_splits = None
        self._cached_path = None
