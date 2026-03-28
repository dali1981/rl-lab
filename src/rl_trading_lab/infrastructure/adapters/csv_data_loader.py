"""
CSV Data Loader - Implementation of DataLoaderPort for CSV files.

Simple loader for quick testing with CSV data files.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CsvDataLoader:
    """
    DataLoaderPort implementation for CSV files.

    Handles:
    - Loading CSV files with pandas
    - Chronological train/val/test splitting
    - Feature detection

    Example:
        >>> loader = CsvDataLoader(val_split=0.1, test_split=0.1)
        >>> train_df, val_df, test_df = loader.load_with_splits(Path("data/prices.csv"))
    """

    def __init__(
        self,
        val_split: float = 0.1,
        test_split: float = 0.1,
        required_columns: Optional[List[str]] = None,
        timestamp_column: Optional[str] = "timestamp",
        parse_dates: bool = True,
    ):
        self._val_split = val_split
        self._test_split = test_split
        self._required_columns = required_columns or ["open", "high", "low", "close", "volume"]
        self._timestamp_column = timestamp_column
        self._parse_dates = parse_dates
        self._cached_splits: Optional[Dict[str, pd.DataFrame]] = None
        self._cached_path: Optional[Path] = None

    def load(
        self,
        data_path: Path,
        mode: str = "train",
    ) -> pd.DataFrame:
        """Load data for the specified mode."""
        data_path = Path(data_path)

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
        """Load and split data."""
        data_path = Path(data_path)

        val_pct = val_split if val_split is not None else self._val_split
        test_pct = test_split if test_split is not None else self._test_split

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
        """Get numeric feature columns."""
        exclude_cols = {"timestamp", "bar_id", "date", "datetime"}
        return [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in ["float64", "int64", "float32", "int32"]
        ]

    def _load_and_split(self, data_path: Path) -> None:
        """Load CSV file and split chronologically."""
        parse_dates = [self._timestamp_column] if self._parse_dates and self._timestamp_column else False
        df = pd.read_csv(data_path, parse_dates=parse_dates)

        missing = set(self._required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        n = len(df)
        test_start = int(n * (1 - self._test_split))
        val_start = int(n * (1 - self._test_split - self._val_split))

        self._cached_splits = {
            "train": df.iloc[:val_start].copy(),
            "eval": df.iloc[val_start:test_start].copy(),
            "test": df.iloc[test_start:].copy(),
        }
        self._cached_path = data_path

        logger.info(
            f"Loaded CSV: {len(df)} rows -> "
            f"train={val_start}, eval={test_start - val_start}, test={n - test_start}"
        )

    def clear_cache(self) -> None:
        """Clear cached data."""
        self._cached_splits = None
        self._cached_path = None
