"""
Environment factory utilities for creating trading environments.

Provides factory functions to create environment instances from configuration,
handling data loading, splitting, and feature engineering.
"""

import logging
from typing import Callable
from pathlib import Path

import pandas as pd

from rl_trading_lab.environment.trading_env import TradingEnv
from rl_trading_lab.utils.data_processor import DataProcessor

logger = logging.getLogger(__name__)


def create_make_env(
    data_path: str,
    observation_config,
    feature_engineering_config,
    env_config,
    val_split: float,
    test_split: float,
) -> Callable[[str], TradingEnv]:
    """
    Create a make_env factory function that encapsulates data loading and environment creation.

    This function loads and processes data once, then returns a factory function that can
    create training, validation, or test environments on demand.

    Args:
        data_path: Path to training data file
        observation_config: Configuration for observation space
        feature_engineering_config: Configuration for feature engineering
        env_config: Environment configuration (contains environment_params, required_columns, price_column)
        val_split: Validation set split ratio
        test_split: Test set split ratio

    Returns:
        Factory function that takes mode ('train', 'eval', 'test') and returns TradingEnv

    Example:
        ```python
        make_env = create_make_env(
            data_path="data/btc_bars.parquet",
            observation_config=config.observation,
            feature_engineering_config=config.feature_engineering,
            env_config=config.env,
            val_split=0.2,
            test_split=0.1,
        )

        # Create environments for different modes
        train_env = make_env('train')
        eval_env = make_env('eval')
        test_env = make_env('test')
        ```
    """
    logger.info("Loading data...")

    # Initialize data processor
    data_processor = DataProcessor(
        data_path=data_path,
        observation_config=observation_config,
        feature_engineering_config=feature_engineering_config,
        val_split=val_split,
        test_split=test_split,
    )

    # Load and process data
    train_df, val_df, test_df, observation_features = data_processor.process()

    # Validate required columns exist
    for col in env_config.required_columns:
        if col not in train_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in data. "
                f"Available columns: {sorted(train_df.columns.tolist())}"
            )

    logger.info(f"Loaded data: Train={len(train_df)} bars, Val={len(val_df)} bars, Test={len(test_df)} bars")
    logger.info(f"Observation features: {len(observation_features)} selected")

    # Extract environment parameters from config
    env_params = env_config.environment_params

    # Create factory function
    def make_env(mode: str) -> TradingEnv:
        """
        Create environment for specified mode.

        Args:
            mode: 'train', 'eval', or 'test'

        Returns:
            TradingEnv instance
        """
        if mode == 'train':
            df = train_df
            randomize = env_params.randomize_start
        elif mode == 'eval':
            df = val_df
            randomize = True  # Enable to get proper variance in eval metrics
        elif mode == 'test':
            df = test_df
            randomize = True  # Enable to assess performance across diverse market conditions
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'train', 'eval', or 'test'")

        logger.info(f"Creating {mode} environment: one_trade_mode={env_params.one_trade_mode}, data_length={len(df)}")

        return TradingEnv(
            df=df,
            lookback_window=env_params.lookback_window,
            initial_balance=env_params.initial_balance,
            commission_rate=env_params.commission_rate,
            slippage_rate=env_params.slippage_rate,
            reward_type=env_params.reward_type,
            discrete_actions=env_params.discrete_actions,
            max_position_pct=env_params.max_position_pct,
            features_to_use=observation_features,
            randomize_start=randomize,
            min_episode_length=env_params.min_episode_length,
            min_holding_period=env_params.min_holding_period,
            hold_closes_position=env_params.hold_closes_position,
            price_column=env_config.price_column,
            one_trade_mode=env_params.one_trade_mode,
        )

    return make_env
