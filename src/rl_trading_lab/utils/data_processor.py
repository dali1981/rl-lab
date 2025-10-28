"""
Data processing for RL training.
Separates concerns: loading, feature engineering, observation selection, splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import logging

from rl_trading_lab.config.observation import ObservationConfig
from rl_trading_lab.config.feature_engineering import FeatureEngineeringConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Load and process trading data for RL training.

    Separates concerns:
    1. Load raw data
    2. Engineer features (optional)
    3. Select observation features
    4. Split into train/val/test
    """

    def __init__(
        self,
        data_path: str,
        observation_config: ObservationConfig,
        feature_engineering_config: FeatureEngineeringConfig,
        val_split: float = 0.2,
        test_split: float = 0.1,
    ):
        """
        Initialize data processor.

        Args:
            data_path: Path to parquet file
            observation_config: Observation space configuration
            feature_engineering_config: Feature engineering configuration
            val_split: Validation split ratio
            test_split: Test split ratio
        """
        self.data_path = Path(data_path)
        self.observation_config = observation_config
        self.feature_engineering_config = feature_engineering_config
        self.val_split = val_split
        self.test_split = test_split

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
        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features based on configuration.

        Args:
            df: Raw dataframe

        Returns:
            DataFrame with engineered features added
        """
        if not self.feature_engineering_config.enabled:
            logger.info("Feature engineering disabled")
            return df

        df = df.copy()
        logger.info("Engineering features...")

        # Add returns
        if self.feature_engineering_config.add_returns:
            for period in self.feature_engineering_config.return_periods:
                ret_col = f'returns_{period}'
                df[ret_col] = df['close'].pct_change(period)
                logger.debug(f"  Created: {ret_col}")

        # Add log returns
        if self.feature_engineering_config.add_log_returns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            logger.debug(f"  Created: log_returns")

        # Add rolling statistics
        if self.feature_engineering_config.add_rolling_stats:
            window = self.feature_engineering_config.rolling_window
            for stat in self.feature_engineering_config.rolling_stats:
                col_name = f'rolling_{stat}_{window}'
                if stat == "mean":
                    df[col_name] = df['close'].rolling(window).mean()
                elif stat == "std":
                    df[col_name] = df['close'].rolling(window).std()
                elif stat == "min":
                    df[col_name] = df['close'].rolling(window).min()
                elif stat == "max":
                    df[col_name] = df['close'].rolling(window).max()
                logger.debug(f"  Created: {col_name}")

        # Handle missing values in engineered features
        if self.feature_engineering_config.missing_values:
            strategy = self.feature_engineering_config.missing_values.strategy
            if strategy == 'forward_fill':
                df = df.ffill()
            elif strategy == 'interpolate':
                df = df.interpolate()
            elif strategy == 'drop':
                df = df.dropna()

            # Fill initial NaN with specified value
            fill_value = self.feature_engineering_config.missing_values.initial_fill
            df = df.fillna(fill_value)

        logger.info(f"✓ Feature engineering complete. Total columns: {len(df.columns)}")
        return df

    def select_observation_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select and validate observation features.

        Args:
            df: DataFrame with all features

        Returns:
            List of selected feature names
        """
        features = self.observation_config.input_features

        # Validate features exist
        if self.observation_config.validate_features:
            missing = [f for f in features if f not in df.columns]
            if missing:
                available_cols = sorted(df.columns.tolist())
                raise ValueError(
                    f"Features specified in observation.input_features not found in data:\n"
                    f"  Missing: {missing}\n"
                    f"  Available columns: {available_cols}"
                )
            logger.info(f"✓ All {len(features)} observation features validated")

        # Log features
        if self.observation_config.log_all_features:
            logger.info(f"Observation features ({len(features)} total):")
            for feat in features:
                logger.info(f"  - {feat}")
        else:
            logger.info(f"Observation features: {len(features)} total: {features[:5]}...")

        return features

    def create_splits(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.

        Args:
            df: Full dataframe

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

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Full pipeline: load, engineer, select, clean, and split.

        Returns:
            train_df, val_df, test_df, observation_features
        """
        # 1. Load raw data
        df = self.load_data()

        # 2. Engineer features (if enabled)
        df = self.engineer_features(df)

        # 3. Select observation features
        observation_features = self.select_observation_features(df)

        # 4. Clean NaN values BEFORE splitting
        # Only keep rows where observation features and price column are valid
        initial_rows = len(df)
        required_columns = observation_features + ['close']  # Need close for price
        df = df.dropna(subset=required_columns).reset_index(drop=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            logger.info(f"✓ Dropped {rows_dropped} rows with NaN in observation features (kept {len(df)})")

        # 5. Create splits
        train_df, val_df, test_df = self.create_splits(df)

        return train_df, val_df, test_df, observation_features
