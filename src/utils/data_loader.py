"""
Data loader for trading data.
Supports loading from Kedro outputs or standalone parquet files.
"""

import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TradingDataLoader:
    """
    Load and prepare trading data for RL training.

    Supports:
    - Kedro catalog outputs
    - Parquet files (from your indicators pipeline)
    - Train/validation/test splits
    - Feature selection and preprocessing
    """

    def __init__(
        self,
        data_path: str,
        features_config: Dict[str, Any],
        val_split: float = 0.2,
        test_split: float = 0.1,
        use_polars: bool = False,
    ):
        """
        Initialize data loader.

        Args:
            data_path: Path to parquet file or Kedro output
            features_config: Feature configuration dict
            val_split: Validation split ratio
            test_split: Test split ratio
            use_polars: Use Polars instead of Pandas
        """
        self.data_path = Path(data_path)
        self.features_config = features_config
        self.val_split = val_split
        self.test_split = test_split
        self.use_polars = use_polars

        # Check if path exists
        if not self.data_path.exists():
            # Try relative to kedro project
            kedro_path = Path("../kedro-crypto-ind") / self.data_path
            if kedro_path.exists():
                self.data_path = kedro_path
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading data from: {self.data_path}")

    def load_data(self) -> pd.DataFrame:
        """Load raw data from file"""
        if self.use_polars:
            df = pl.read_parquet(self.data_path)
            return df.to_pandas()
        else:
            return pd.read_parquet(self.data_path)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features based on configuration.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with selected and engineered features
        """
        df = df.copy()

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Select base features
        features_to_use = []

        # Price features
        if 'price_features' in self.features_config:
            for feat in self.features_config['price_features']:
                if feat in df.columns:
                    features_to_use.append(feat)

        # Technical indicators
        if self.features_config.get('use_zscore', False):
            # Use z-score normalized versions
            for feat in self.features_config.get('zscore_indicators', []):
                if feat in df.columns:
                    features_to_use.append(feat)
                    # Handle NaN values in z-scores (beginning of data)
                    fill_value = self.features_config.get('missing_values', {}).get('initial_zscore_fill', 0.0)
                    df[feat] = df[feat].fillna(fill_value)
        else:
            # Use raw indicators
            for feat in self.features_config.get('technical_indicators', []):
                if feat in df.columns:
                    features_to_use.append(feat)

        # Feature engineering
        if self.features_config.get('feature_engineering', {}).get('add_returns', False):
            # Add returns
            for period in self.features_config['feature_engineering'].get('return_periods', [1]):
                ret_col = f'returns_{period}'
                df[ret_col] = df['close'].pct_change(period)
                features_to_use.append(ret_col)

        if self.features_config.get('feature_engineering', {}).get('add_log_returns', False):
            # Add log returns
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features_to_use.append('log_returns')

        # Handle missing values
        strategy = self.features_config.get('missing_values', {}).get('strategy', 'forward_fill')
        if strategy == 'forward_fill':
            df[features_to_use] = df[features_to_use].fillna(method='ffill')
        elif strategy == 'interpolate':
            df[features_to_use] = df[features_to_use].interpolate()
        elif strategy == 'drop':
            df = df.dropna(subset=features_to_use)

        # Store selected features
        self.selected_features = features_to_use

        logger.info(f"Selected {len(features_to_use)} features: {features_to_use[:5]}...")

        return df

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.

        Args:
            df: Prepared dataframe

        Returns:
            train_df, val_df, test_df
        """
        n = len(df)

        # Calculate split points
        test_size = int(n * self.test_split)
        val_size = int(n * self.val_split)
        train_size = n - test_size - val_size

        # Create splits (time-based, no shuffling)
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()

        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

        # Add date ranges if timestamp available
        if 'timestamp' in df.columns:
            logger.info(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
            logger.info(f"Val: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
            logger.info(f"Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        return train_df, val_df, test_df

    def load_and_prepare(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Full pipeline: load, prepare, and split data.

        Returns:
            train_df, val_df, test_df, feature_names
        """
        # Load raw data
        df = self.load_data()
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Prepare features
        df = self.prepare_features(df)

        # Create splits
        train_df, val_df, test_df = self.create_splits(df)

        return train_df, val_df, test_df, self.selected_features


def load_kedro_catalog_data(
    catalog_name: str,
    kedro_project_path: str = "../kedro-crypto-ind"
) -> pd.DataFrame:
    """
    Load data directly from Kedro catalog.

    Args:
        catalog_name: Name in Kedro catalog (e.g., "ml_ready_features")
        kedro_project_path: Path to Kedro project

    Returns:
        DataFrame from Kedro catalog
    """
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(kedro_project_path).resolve()))

        from kedro.framework.session import KedroSession
        from kedro.framework.startup import bootstrap_project

        # Bootstrap Kedro project
        bootstrap_project(Path(kedro_project_path))

        # Create session and load data
        with KedroSession.create() as session:
            df = session.load_context().catalog.load(catalog_name)

        logger.info(f"Loaded {catalog_name} from Kedro catalog")
        return df

    except ImportError:
        logger.warning("Kedro not available, falling back to direct parquet loading")
        # Fall back to direct file loading
        data_path = Path(kedro_project_path) / f"data/08_reporting/{catalog_name}.parquet"
        return pd.read_parquet(data_path)


def get_feature_statistics(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Get statistics for features (useful for debugging).

    Args:
        df: DataFrame
        features: List of feature names

    Returns:
        DataFrame with statistics
    """
    stats = []
    for feat in features:
        if feat in df.columns:
            stats.append({
                'feature': feat,
                'dtype': str(df[feat].dtype),
                'null_count': df[feat].isna().sum(),
                'null_pct': df[feat].isna().mean() * 100,
                'mean': df[feat].mean() if df[feat].dtype in ['float64', 'int64'] else None,
                'std': df[feat].std() if df[feat].dtype in ['float64', 'int64'] else None,
                'min': df[feat].min() if df[feat].dtype in ['float64', 'int64'] else None,
                'max': df[feat].max() if df[feat].dtype in ['float64', 'int64'] else None,
            })

    return pd.DataFrame(stats)