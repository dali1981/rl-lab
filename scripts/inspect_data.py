#!/usr/bin/env python
"""
Inspect available features in data and show configuration.

Usage:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --show-data  # Also show sample data
    python scripts/inspect_data.py --show-stats  # Show statistics
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from rl_trading_lab.config import load_config
from rl_trading_lab.utils.data_processor import DataProcessor
from hydra import compose, initialize_config_dir


def main():
    parser = argparse.ArgumentParser(description="Inspect data and configuration")
    parser.add_argument('--show-data', action='store_true', help="Show sample data rows")
    parser.add_argument('--show-stats', action='store_true', help="Show basic statistics")
    parser.add_argument('--observation', type=str, default=None, help="Observation config to use")
    parser.add_argument('--feature-engineering', type=str, default=None, help="Feature engineering config to use")
    args = parser.parse_args()

    # Load config
    print("Loading configuration...")
    config_dir = str(Path(__file__).parent.parent / "configs")

    overrides = []
    if args.observation:
        overrides.append(f"observation={args.observation}")
    if args.feature_engineering:
        overrides.append(f"feature_engineering={args.feature_engineering}")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
        config = load_config(cfg)

    print(f"Data path: {config.data.train_data_path}\n")

    # Process data
    print("Processing data...")
    processor = DataProcessor(
        data_path=config.data.train_data_path,
        observation_config=config.observation,
        feature_engineering_config=config.feature_engineering,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
    )

    train_df, val_df, test_df, observation_features = processor.process()

    # Show available columns
    print("\n" + "="*80)
    print(f"AVAILABLE COLUMNS IN DATA ({len(train_df.columns)} total)")
    print("="*80)
    for col in sorted(train_df.columns):
        print(f"  - {col}")

    # Show observation features
    print("\n" + "="*80)
    print(f"OBSERVATION FEATURES ({len(observation_features)} total)")
    print("="*80)
    print("These are the features the RL agent observes:")
    for feat in observation_features:
        print(f"  - {feat}")

    # Configuration summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Observation config:     {args.observation or 'default'}")
    print(f"Feature engineering:    {'enabled' if config.feature_engineering.enabled else 'disabled'}")
    if config.feature_engineering.enabled:
        if config.feature_engineering.add_returns:
            print(f"  • Returns periods:    {config.feature_engineering.return_periods}")
        if config.feature_engineering.add_log_returns:
            print(f"  • Log returns:        enabled")
        if config.feature_engineering.add_rolling_stats:
            print(f"  • Rolling stats:      {config.feature_engineering.rolling_stats}")

    print(f"\nEnvironment config:")
    print(f"  • Price column:       {config.env.price_column}")
    print(f"  • Required columns:   {config.env.required_columns}")
    print(f"  • Reward type:        {config.env.environment_params.reward_type}")
    print(f"  • Lookback window:    {config.env.environment_params.lookback_window}")
    print(f"  • Initial balance:    ${config.env.environment_params.initial_balance:,.2f}")

    # Data info
    print("\n" + "="*80)
    print("DATA INFORMATION")
    print("="*80)
    print(f"Total rows: {len(train_df) + len(val_df) + len(test_df):,}")
    print(f"Train size: {len(train_df):,} rows")
    print(f"Val size: {len(val_df):,} rows")
    print(f"Test size: {len(test_df):,} rows")

    if 'timestamp' in train_df.columns:
        print(f"\nTime ranges:")
        print(f"  Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Val:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
        print(f"  Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Show sample data if requested
    if args.show_data:
        print("\n" + "="*80)
        print("SAMPLE DATA (first 5 rows, observation features only)")
        print("="*80)
        print(train_df[observation_features].head())

    # Show statistics if requested
    if args.show_stats:
        print("\n" + "="*80)
        print("FEATURE STATISTICS (observation features)")
        print("="*80)
        print(train_df[observation_features].describe())

    # Usage examples
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    print("\n1. Inspect with different observation config:")
    print("   python scripts/inspect_data.py --observation minimal")
    print("\n2. Inspect with feature engineering:")
    print("   python scripts/inspect_data.py --feature-engineering returns")
    print("\n3. Show sample data and statistics:")
    print("   python scripts/inspect_data.py --show-data --show-stats")
    print("\n4. Use specific configs:")
    print("   python scripts/inspect_data.py --observation indicators_only --feature-engineering none")
    print()


if __name__ == "__main__":
    main()
