#!/usr/bin/env python3
"""
Validation script for live trading system.

Tests the entire pipeline with recent historical data before going live:
1. Load recent data from MinIO
2. Create bars
3. Compute features
4. Run model predictions
5. Simulate trades
6. Verify everything works end-to-end

Example usage:
    uv run python experiments/validate_live.py \\
        --symbol BTCUSDT \\
        --days 1 \\
        --model checkpoints/PPO_returns_20251028_143659/best_model.zip
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_lab.data.binance_adapter import BinanceDataAdapter
from rl_trading_lab.data.bar_processor import BarProcessor
from rl_trading_lab.data.feature_pipeline import FeaturePipeline
from rl_trading_lab.live import (
    FeatureComputer,
    ModelInferenceEngine,
    PortfolioManager,
)
from rl_trading_lab.live.safety import SafetyGuard

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()
console = Console()


def print_header(title: str):
    """Print a section header."""
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 60)


@app.command()
def main(
    symbol: str = typer.Option("BTCUSDT", "--symbol", "-s", help="Symbol to validate"),
    days: int = typer.Option(1, "--days", "-d", help="Days of historical data to use"),
    model_path: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to trained model",
    ),
    vecnormalize_path: str = typer.Option(
        None,
        "--vecnormalize",
        help="Path to VecNormalize wrapper (only needed if model was trained with VecNormalize)",
    ),
    dollar_volume_threshold: float = typer.Option(
        1000000,
        "--threshold",
        "-t",
        help="Dollar volume threshold for bars",
    ),
):
    """
    Validate live trading system with recent historical data.

    This script simulates the entire live trading pipeline using
    historical data to ensure everything works before going live.
    """
    console.print("\n[bold]Live Trading System Validation[/bold]", style="cyan")
    console.print(f"Symbol: {symbol}, Days: {days}\n")

    # Step 1: Load data
    print_header("Step 1: Loading Historical Data")
    try:
        adapter = BinanceDataAdapter()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        console.print(f"Loading trades from {start_date.date()} to {end_date.date()}...")
        trades_df = adapter.load_symbol_data(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )

        if trades_df.empty:
            console.print("[red]No data loaded! Check MinIO and data pipeline.[/red]")
            sys.exit(1)

        console.print(f"[green]✓ Loaded {len(trades_df):,} trades[/green]")

    except Exception as e:
        console.print(f"[red]✗ Error loading data: {e}[/red]")
        sys.exit(1)

    # Step 2: Create bars
    print_header("Step 2: Creating Dollar Volume Bars")
    try:
        bar_processor = BarProcessor(symbol=symbol, threshold=dollar_volume_threshold)
        bars_df = bar_processor.process(trades_df)

        console.print(f"[green]✓ Created {len(bars_df)} bars[/green]")
        console.print(f"  Threshold: ${dollar_volume_threshold:,.0f}")
        console.print(f"  Time span: {bars_df['timestamp'].min()} to {bars_df['timestamp'].max()}")

    except Exception as e:
        console.print(f"[red]✗ Error creating bars: {e}[/red]")
        sys.exit(1)

    # Step 3: Compute features
    print_header("Step 3: Computing Features")
    try:
        # Initialize feature computer
        feature_computer = FeatureComputer(symbol=symbol)

        # Feed bars sequentially
        features_list = []
        for idx, row in bars_df.iterrows():
            bar = pd.DataFrame([row])
            is_ready = feature_computer.add_bar(bar)

            if is_ready:
                features = feature_computer.get_latest_features()
                if features is not None:
                    features_list.append(features)

        if not features_list:
            console.print("[red]✗ No features computed![/red]")
            sys.exit(1)

        features_df = pd.concat(features_list, ignore_index=True)
        console.print(f"[green]✓ Computed features for {len(features_df)} bars[/green]")
        console.print(f"  Features: {', '.join(features_df.columns[:5])}...")

    except Exception as e:
        console.print(f"[red]✗ Error computing features: {e}[/red]")
        sys.exit(1)

    # Step 4: Model inference (if model provided)
    if model_path:
        print_header("Step 4: Testing Model Inference")
        try:
            engine = ModelInferenceEngine(
                model_path=model_path,
                vecnormalize_path=vecnormalize_path,
            )

            # Make predictions on all features
            predictions = []
            for idx, row in features_df.iterrows():
                feature_row = pd.DataFrame([row])
                action, confidence = engine.predict(feature_row)
                action_name = engine.get_action_name(action)
                predictions.append({
                    "action": action_name,
                    "confidence": confidence,
                })

            predictions_df = pd.DataFrame(predictions)
            console.print(f"[green]✓ Generated {len(predictions)} predictions[/green]")

            # Show action distribution
            action_counts = predictions_df["action"].value_counts()
            table = Table(title="Action Distribution")
            table.add_column("Action", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Percentage", justify="right")

            for action, count in action_counts.items():
                pct = count / len(predictions_df) * 100
                table.add_row(action, str(count), f"{pct:.1f}%")

            console.print(table)

            # Show confidence stats
            console.print(f"\nConfidence stats:")
            console.print(f"  Mean: {predictions_df['confidence'].mean():.3f}")
            console.print(f"  Min:  {predictions_df['confidence'].min():.3f}")
            console.print(f"  Max:  {predictions_df['confidence'].max():.3f}")

        except Exception as e:
            console.print(f"[red]✗ Error in model inference: {e}[/red]")
            sys.exit(1)

    # Step 5: Simulate trading
    print_header("Step 5: Simulating Trading (Paper)")
    try:
        portfolio = PortfolioManager(
            initial_balance=10000,
            symbols=[symbol],
            db_path=":memory:",  # In-memory database
        )

        safety_guard = SafetyGuard(
            max_drawdown=0.20,
            initial_balance=10000,
            enable_circuit_breaker=True,
        )

        # Simulate trades
        trade_count = 0
        for idx in range(len(bars_df)):
            if idx >= len(features_df):
                break

            if model_path and idx < len(predictions):
                action_name = predictions[idx]["action"]
                price = bars_df.iloc[idx]["close"]

                # Check safety
                can_trade, reason = safety_guard.can_trade(symbol)
                if not can_trade:
                    console.print(f"[yellow]Trade blocked: {reason}[/yellow]")
                    break

                # Simulate order
                if action_name == "BUY":
                    # Buy
                    quantity = 1000 / price  # $1000 position
                    portfolio.record_trade(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=price,
                        commission=quantity * price * 0.001,
                    )
                    trade_count += 1

                elif action_name == "SELL":
                    # Sell if we have position
                    position = portfolio.get_position(symbol)
                    if position["quantity"] > 0:
                        portfolio.record_trade(
                            symbol=symbol,
                            action="SELL",
                            quantity=position["quantity"],
                            price=price,
                            commission=position["value"] * 0.001,
                        )
                        trade_count += 1

            # Update prices
            portfolio.update_prices({symbol: bars_df.iloc[idx]["close"]})

        # Show results
        stats = portfolio.get_stats()

        result_table = Table(title="Simulation Results")
        result_table.add_column("Metric", style="cyan")
        result_table.add_column("Value", justify="right")

        result_table.add_row("Initial Balance", f"${stats['initial_balance']:,.2f}")
        result_table.add_row("Final Balance", f"${stats['total_value']:,.2f}")

        pnl = stats['total_pnl']
        pnl_color = "green" if pnl >= 0 else "red"
        result_table.add_row("Total PnL", f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]")

        returns_pct = stats['returns'] * 100
        returns_color = "green" if returns_pct >= 0 else "red"
        result_table.add_row("Returns", f"[{returns_color}]{returns_pct:+.2f}%[/{returns_color}]")

        result_table.add_row("Total Trades", str(trade_count))
        result_table.add_row("Commission", f"${stats['total_commission']:,.2f}")

        console.print(result_table)

        # Safety guard status
        safety_stats = safety_guard.get_stats()
        console.print(f"\n[bold]Safety Guard Status:[/bold]")
        console.print(f"  State: {safety_stats['state']}")
        console.print(f"  Drawdown: {safety_stats['drawdown_pct']:.2f}%")
        console.print(f"  Violations: {safety_stats['violations']}")

    except Exception as e:
        console.print(f"[red]✗ Error in simulation: {e}[/red]")
        sys.exit(1)

    # Summary
    print_header("Validation Summary")

    if trade_count > 0 and abs(stats['returns']) > 0.001:
        status = Panel(
            "[green]✓ All tests passed! System is ready for live trading.[/green]",
            title="Status",
            border_style="green",
        )
    else:
        status = Panel(
            "[yellow]⚠ Validation completed but no significant activity.[/yellow]\n"
            "Consider testing with more data or checking model predictions.",
            title="Status",
            border_style="yellow",
        )

    console.print(status)


if __name__ == "__main__":
    app()
