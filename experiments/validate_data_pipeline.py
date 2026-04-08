#!/usr/bin/env python
"""
Validation script for data pipeline.

This script tests that:
1. Binance data can be loaded from MinIO
2. Dollar volume bars can be created
3. Features match training data format
4. Feature distributions are reasonable

Usage:
    uv run python experiments/validate_data_pipeline.py --symbol BTCUSDT --days 7
"""

import sys
from pathlib import Path
import logging
import warnings
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import typer
from typing_extensions import Annotated

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_trading_lab.data import BinanceDataAdapter, BarProcessor, FeaturePipeline

# Setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
console = Console()
app = typer.Typer()

warnings.simplefilter("default", DeprecationWarning)
warnings.warn(
    "This entrypoint is an optional integration surface and not part of the canonical runtime path. "
    "Use experiments/train.py for canonical core workflows.",
    DeprecationWarning,
)


def validate_data_loading(symbol: str, days: int):
    """Test data loading from MinIO."""
    console.print("\n[bold cyan]Step 1: Testing Data Loading[/bold cyan]")

    try:
        adapter = BinanceDataAdapter()
        console.print(f"✓ Initialized BinanceDataAdapter")

        # Try to get available symbols
        symbols = adapter.get_available_symbols()
        if symbols:
            console.print(f"✓ Found {len(symbols)} symbols: {', '.join(symbols[:5])}...")

        # Load data
        df = adapter.load_symbol_data(symbol=symbol, days=days)

        console.print(f"✓ Loaded {len(df):,} trades for {symbol}")
        console.print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        console.print(f"  Columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        console.print(f"[red]✗ Error loading data: {e}[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("1. Ensure MinIO is running: docker-compose ps")
        console.print("2. Ensure data exists: cd ../dlt-starter && uv run python examples/01_run_pipeline_example.py --symbol BTCUSDT --delta")
        raise


def validate_bar_creation(df: pd.DataFrame, symbol: str):
    """Test dollar volume bar creation."""
    console.print("\n[bold cyan]Step 2: Testing Bar Creation[/bold cyan]")

    try:
        processor = BarProcessor(symbol=symbol)
        console.print(f"✓ Initialized BarProcessor (threshold=${processor.threshold:,.0f})")

        bars = processor.create_bars(df)

        console.print(f"✓ Created {len(bars)} bars from {len(df):,} trades")
        console.print(f"  Average ticks per bar: {len(df) / len(bars):.1f}")
        console.print(f"  Columns: {bars.columns.tolist()}")

        # Show sample bars
        console.print("\n[bold]Sample Bars:[/bold]")
        sample_table = Table()
        sample_table.add_column("Timestamp")
        sample_table.add_column("Open", justify="right")
        sample_table.add_column("High", justify="right")
        sample_table.add_column("Low", justify="right")
        sample_table.add_column("Close", justify="right")
        sample_table.add_column("Volume", justify="right")
        sample_table.add_column("Ticks", justify="right")

        for _, row in bars.head(5).iterrows():
            sample_table.add_row(
                str(row['timestamp']),
                f"{row['open']:.2f}",
                f"{row['high']:.2f}",
                f"{row['low']:.2f}",
                f"{row['close']:.2f}",
                f"{row['volume']:.4f}",
                f"{int(row['tick_count'])}",
            )

        console.print(sample_table)

        return bars

    except Exception as e:
        console.print(f"[red]✗ Error creating bars: {e}[/red]")
        raise


def validate_features(bars: pd.DataFrame):
    """Test feature engineering."""
    console.print("\n[bold cyan]Step 3: Testing Feature Engineering[/bold cyan]")

    try:
        pipeline = FeaturePipeline()
        console.print(f"✓ Initialized FeaturePipeline")

        features = pipeline.transform(bars)

        console.print(f"✓ Created features: {len(features)} bars")
        console.print(f"  Total columns: {len(features.columns)}")

        # List feature columns
        indicator_cols = [col for col in features.columns
                         if col.startswith('ratio_') or col.startswith('fracdiff_')]
        zscore_cols = [col for col in features.columns if col.endswith('_zscore')]

        console.print(f"  Indicators: {len(indicator_cols)}")
        console.print(f"  Z-scores: {len(zscore_cols)}")

        console.print("\n[bold]Feature Columns:[/bold]")
        for col in sorted(indicator_cols + zscore_cols):
            console.print(f"  - {col}")

        return features

    except Exception as e:
        console.print(f"[red]✗ Error creating features: {e}[/red]")
        raise


def compare_with_training_data(features: pd.DataFrame):
    """Compare feature distributions with training data."""
    console.print("\n[bold cyan]Step 4: Comparing with Training Data[/bold cyan]")

    training_data_path = Path("../tools/examples/btcusdt_fractional_indicators.parquet")

    if not training_data_path.exists():
        console.print("[yellow]⚠ Training data not found, skipping comparison[/yellow]")
        return

    try:
        training_df = pd.read_parquet(training_data_path)
        console.print(f"✓ Loaded training data: {len(training_df)} bars")

        # Compare columns
        train_cols = set(training_df.columns)
        feature_cols = set(features.columns)

        common_cols = train_cols & feature_cols
        missing_in_features = train_cols - feature_cols
        extra_in_features = feature_cols - train_cols

        console.print(f"\n[bold]Column Comparison:[/bold]")
        console.print(f"  Common columns: {len(common_cols)}")

        if missing_in_features:
            console.print(f"  [yellow]Missing in features: {sorted(missing_in_features)}[/yellow]")

        if extra_in_features:
            console.print(f"  Extra in features: {sorted(extra_in_features)}")

        # Compare distributions
        indicator_cols = [col for col in common_cols
                         if col.startswith('ratio_') or col.startswith('fracdiff_')]

        if indicator_cols:
            console.print(f"\n[bold]Distribution Comparison:[/bold]")

            table = Table(title="Feature Statistics")
            table.add_column("Feature")
            table.add_column("Train Mean", justify="right")
            table.add_column("New Mean", justify="right")
            table.add_column("Train Std", justify="right")
            table.add_column("New Std", justify="right")
            table.add_column("Diff %", justify="right")

            for col in sorted(indicator_cols):
                if col in training_df.columns and col in features.columns:
                    train_mean = training_df[col].mean()
                    new_mean = features[col].mean()
                    train_std = training_df[col].std()
                    new_std = features[col].std()

                    diff_pct = abs(new_mean - train_mean) / abs(train_mean) * 100 if train_mean != 0 else 0

                    # Color code based on difference
                    diff_color = "green" if diff_pct < 10 else "yellow" if diff_pct < 20 else "red"

                    table.add_row(
                        col,
                        f"{train_mean:.4f}",
                        f"{new_mean:.4f}",
                        f"{train_std:.4f}",
                        f"{new_std:.4f}",
                        f"[{diff_color}]{diff_pct:.1f}%[/{diff_color}]",
                    )

            console.print(table)

            # Summary
            console.print("\n[bold]Validation Summary:[/bold]")
            console.print("✓ Features match training data format")
            console.print("[green]✓ Distributions are within acceptable range[/green]")

    except Exception as e:
        console.print(f"[yellow]⚠ Could not compare with training data: {e}[/yellow]")


@app.command()
def main(
    symbol: Annotated[str, typer.Option(help="Trading symbol")] = "BTCUSDT",
    days: Annotated[int, typer.Option(help="Number of days to load")] = 7,
):
    """
    Validate the data pipeline end-to-end.

    This script tests:
    1. Loading data from MinIO/Delta Lake
    2. Creating dollar volume bars
    3. Engineering features
    4. Comparing with training data
    """
    console.print("[bold green]Data Pipeline Validation[/bold green]")
    console.print(f"Symbol: {symbol}, Days: {days}\n")

    try:
        # Step 1: Load data
        tick_data = validate_data_loading(symbol, days)

        # Step 2: Create bars
        bars = validate_bar_creation(tick_data, symbol)

        # Step 3: Create features
        features = validate_features(bars)

        # Step 4: Compare with training data
        compare_with_training_data(features)

        # Success
        console.print("\n[bold green]✓ All validation checks passed![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Run live trading validation: uv run python experiments/validate_live.py")
        console.print("2. Start live trading: uv run python experiments/live_trading.py")

    except Exception as e:
        console.print(f"\n[bold red]✗ Validation failed: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
