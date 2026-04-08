#!/usr/bin/env python3
"""
Complete Live Trading Example with RL Models

This comprehensive example demonstrates how to:
1. Load trained RL models (PPO/A2C/DQN)
2. Set up live trading components
3. Implement custom safety guards
4. Handle errors gracefully
5. Analyze trading performance

IMPORTANT: This example uses BTCUSDT on Binance testnet.
- Testnet uses fake money (no risk)
- Get testnet keys from: https://testnet.binance.vision/
- Always validate on testnet before using real money

Example usage:
    # Validate with historical data (no trading)
    uv run python examples/live_trading_example.py validate \\
        --model checkpoints/PPO_returns_20251028_143659/best_model.zip \\
        --days 1

    # Run on testnet (paper trading)
    uv run python examples/live_trading_example.py trade \\
        --model checkpoints/PPO_returns_20251028_143659/best_model.zip \\
        --symbol BTCUSDT

    # Analyze past trading session
    uv run python examples/live_trading_example.py analyze \\
        --db portfolio.db
"""

import asyncio
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from rl_trading_lab.data import BinanceDataAdapter, BarProcessor, FeaturePipeline
from rl_trading_lab.live import (
    FeatureComputer,
    ModelInferenceEngine,
    OrderExecutor,
    PortfolioManager,
    SafetyGuard,
    StreamConsumer,
    TradingDashboard,
    CircuitBreakerState,
)
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()

# Initialize Typer app
app = typer.Typer(help="Live Trading Example with RL Models")

warnings.simplefilter("default", DeprecationWarning)
warnings.warn(
    "This entrypoint is an optional demo/integration surface and not part of the canonical runtime path. "
    "Use experiments/train.py for canonical core workflows.",
    DeprecationWarning,
)


# ============================================================================
# SECTION 1: CUSTOM SAFETY GUARD
# ============================================================================


class CustomSafetyGuard(SafetyGuard):
    """
    Extended safety guard with custom rules.

    This demonstrates how to add custom safety checks beyond the default
    circuit breaker logic. Examples include:
    - Time-based trading windows (only trade during market hours)
    - Symbol-specific risk limits
    - Volatility-based position sizing
    - Custom violation handlers
    """

    def __init__(self, *args, trading_hours: Optional[Tuple[int, int]] = None, **kwargs):
        """
        Initialize custom safety guard.

        Args:
            trading_hours: Tuple of (start_hour, end_hour) in UTC (e.g., (9, 21) for 9am-9pm)
            *args, **kwargs: Passed to parent SafetyGuard
        """
        super().__init__(*args, **kwargs)
        self.trading_hours = trading_hours

        # Custom violation tracking
        self.custom_violations: Dict[str, int] = {
            "outside_trading_hours": 0,
            "high_volatility": 0,
        }

        logger.info(
            f"Initialized CustomSafetyGuard with trading hours: "
            f"{trading_hours if trading_hours else 'always'}"
        )

    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if trading is allowed with custom rules.

        First checks parent class rules (drawdown, rate limits, etc.),
        then applies custom rules.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (allowed, reason)
        """
        # Check parent safety rules first
        allowed, reason = super().can_trade(symbol)
        if not allowed:
            return False, reason

        # Custom rule: Trading hours check
        if self.trading_hours is not None:
            current_hour = datetime.utcnow().hour
            start_hour, end_hour = self.trading_hours

            if not (start_hour <= current_hour < end_hour):
                self.custom_violations["outside_trading_hours"] += 1
                return False, f"Outside trading hours ({start_hour}:00-{end_hour}:00 UTC)"

        # All checks passed
        return True, "OK"

    def get_custom_stats(self) -> Dict:
        """Get statistics including custom violations."""
        base_stats = self.get_stats()
        base_stats["custom_violations"] = self.custom_violations.copy()
        return base_stats


# ============================================================================
# SECTION 2: MODEL LOADING AND INSPECTION
# ============================================================================


def load_and_inspect_model(
    model_path: str,
    vecnormalize_path: Optional[str] = None,
) -> ModelInferenceEngine:
    """
    Load a trained RL model and inspect its properties.

    This function demonstrates:
    - Loading models with/without VecNormalize
    - Inspecting model metadata
    - Testing model predictions
    - Handling loading errors

    Args:
        model_path: Path to trained model (.zip file)
        vecnormalize_path: Optional path to VecNormalize wrapper (.pkl file)

    Returns:
        ModelInferenceEngine instance

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model is incompatible
    """
    model_path = Path(model_path)

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Available models in checkpoints/:\n"
            f"{list(Path('checkpoints').glob('*/best_model*'))}"
        )

    console.print(Panel.fit(
        f"[bold cyan]Loading Model[/bold cyan]\n\n"
        f"Path: {model_path}\n"
        f"VecNormalize: {vecnormalize_path if vecnormalize_path else 'None'}",
        border_style="cyan"
    ))

    # Auto-detect VecNormalize if in same directory
    if vecnormalize_path is None:
        potential_vecnorm = model_path.parent / "vecnormalize.pkl"
        if potential_vecnorm.exists():
            vecnormalize_path = str(potential_vecnorm)
            logger.info(f"Auto-detected VecNormalize: {vecnormalize_path}")

    try:
        # Load model
        engine = ModelInferenceEngine(
            model_path=str(model_path),
            vecnormalize_path=vecnormalize_path,
        )

        # Display model info
        table = Table(title="Model Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model Type", engine.model_type)
        table.add_row("Model Path", str(model_path.name))
        table.add_row("Has VecNormalize", "Yes" if engine.vecnormalize else "No")

        # Try to read metadata if available
        metadata_path = model_path.parent / "best_model.metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
            table.add_row("Training Date", metadata.get("timestamp", "Unknown"))
            if "custom" in metadata and "best_mean_reward" in metadata["custom"]:
                table.add_row("Best Mean Reward", f"{metadata['custom']['best_mean_reward']:.4f}")

        console.print(table)

        # Display training configuration if available
        from rl_trading_lab.utils.checkpoint_manager import CheckpointManager
        try:
            config = CheckpointManager.get_training_config(model_path)
            if config:
                config_table = Table(title="Training Configuration")
                config_table.add_column("Config Type", style="cyan")
                config_table.add_column("Details", style="yellow")

                # Show observation config
                if "observation" in config:
                    obs_config = config["observation"]
                    features = obs_config.get("input_features", [])
                    if features:
                        config_table.add_row("Observation Features", f"{len(features)} features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")

                # Show feature engineering config
                if "feature_engineering" in config:
                    fe_config = config["feature_engineering"]
                    if "technical_indicators" in fe_config:
                        indicators = fe_config["technical_indicators"]
                        indicator_names = []
                        if indicators.get("sma_ratios"):
                            indicator_names.append(f"SMA ratios: {indicators['sma_ratios']}")
                        if indicators.get("range_ratios"):
                            indicator_names.append(f"Range ratios: {indicators['range_ratios']}")
                        if indicator_names:
                            config_table.add_row("Technical Indicators", ", ".join(indicator_names))

                # Show environment config
                if "env" in config:
                    env_config = config["env"]
                    if "environment_params" in env_config:
                        params = env_config["environment_params"]
                        config_table.add_row("Reward Type", params.get("reward_type", "Unknown"))
                        config_table.add_row("Initial Balance", f"${params.get('initial_balance', 10000):,.2f}")
                        config_table.add_row("Commission Rate", f"{params.get('commission_rate', 0.001):.4f}")

                config_table.add_row("Config Source", config.get("source", "unknown"))
                console.print("\n")
                console.print(config_table)
        except Exception as e:
            logger.debug(f"Could not load training config: {e}")

        return engine

    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise


# ============================================================================
# SECTION 3: VALIDATION WITH HISTORICAL DATA
# ============================================================================


def validate_with_historical_data(
    symbol: str,
    model_path: str,
    vecnormalize_path: Optional[str] = None,
    days: int = 1,
) -> Dict:
    """
    Validate the live trading pipeline with historical data.

    This is a CRITICAL step before going live. It tests:
    - Data loading from MinIO
    - Dollar volume bar creation
    - Feature computation
    - Model predictions
    - Simulated trade execution

    NO REAL TRADING occurs - this is purely validation.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        model_path: Path to trained model
        vecnormalize_path: Optional path to VecNormalize
        days: Number of days of historical data to test

    Returns:
        Dict with validation results and statistics
    """
    console.print(Panel.fit(
        f"[bold yellow]Validation Mode[/bold yellow]\n\n"
        f"Symbol: {symbol}\n"
        f"Days: {days}\n"
        f"Model: {Path(model_path).name}\n\n"
        f"This will test the entire pipeline without real trading.",
        border_style="yellow"
    ))

    # Step 1: Load historical data
    console.print("\n[cyan]Step 1:[/cyan] Loading historical data from MinIO...")

    try:
        adapter = BinanceDataAdapter()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = adapter.load_symbol_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        console.print(f"  ✓ Loaded {len(df):,} trades")

    except Exception as e:
        console.print(f"  [red]✗ Failed to load data: {e}[/red]")
        return {"success": False, "error": str(e)}

    # Step 2: Create dollar volume bars
    console.print("\n[cyan]Step 2:[/cyan] Creating dollar volume bars...")

    try:
        bar_processor = BarProcessor(
            dollar_volume_threshold=1_000_000,  # $1M per bar
        )

        bars_df = bar_processor.process_trades(df)
        console.print(f"  ✓ Created {len(bars_df)} bars")

    except Exception as e:
        console.print(f"  [red]✗ Failed to create bars: {e}[/red]")
        return {"success": False, "error": str(e)}

    # Step 3: Engineer features
    console.print("\n[cyan]Step 3:[/cyan] Computing features...")

    try:
        pipeline = FeaturePipeline()
        features_df = pipeline.transform(bars_df)

        console.print(f"  ✓ Computed {len(features_df.columns)} features")
        console.print(f"    Features: {', '.join(features_df.columns[:5])}...")

    except Exception as e:
        console.print(f"  [red]✗ Failed to compute features: {e}[/red]")
        return {"success": False, "error": str(e)}

    # Step 4: Load model and test predictions
    console.print("\n[cyan]Step 4:[/cyan] Testing model predictions...")

    try:
        engine = load_and_inspect_model(model_path, vecnormalize_path)

        # Test predictions on recent data
        predictions = []
        for i in range(min(10, len(features_df))):
            features = features_df.iloc[i].values.reshape(1, -1)
            action, confidence = engine.predict(features, deterministic=True)
            action_name = engine.get_action_name(action)
            predictions.append((action_name, confidence))

        console.print(f"  ✓ Model predictions working")
        console.print(f"    Sample: {predictions[:3]}")

    except Exception as e:
        console.print(f"  [red]✗ Failed to load model: {e}[/red]")
        return {"success": False, "error": str(e)}

    # Step 5: Simulate trades
    console.print("\n[cyan]Step 5:[/cyan] Simulating trades...")

    try:
        portfolio = PortfolioManager(
            initial_balance=10000,
            symbols=[symbol],
            db_path=":memory:",  # In-memory for validation
        )

        num_trades = 0
        for i in range(len(features_df)):
            features = features_df.iloc[i].values.reshape(1, -1)
            action, confidence = engine.predict(features, deterministic=True)

            if action != 0:  # Not HOLD
                price = bars_df.iloc[i]["close"]
                quantity = 100 / price  # $100 position

                action_name = "BUY" if action == 1 else "SELL"
                portfolio.record_trade(
                    symbol=symbol,
                    action=action_name,
                    quantity=quantity,
                    price=price,
                    commission=quantity * price * 0.001,
                )
                num_trades += 1

        stats = portfolio.get_stats()

        console.print(f"  ✓ Simulated {num_trades} trades")
        console.print(f"    Final balance: ${stats['total_value']:,.2f}")
        console.print(f"    Total PnL: ${stats['total_pnl']:+,.2f}")
        console.print(f"    Returns: {stats['returns']*100:+.2f}%")

    except Exception as e:
        console.print(f"  [red]✗ Failed to simulate trades: {e}[/red]")
        return {"success": False, "error": str(e)}

    console.print("\n[green]✓ Validation complete! All systems operational.[/green]")

    return {
        "success": True,
        "trades": len(df),
        "bars": len(bars_df),
        "simulated_trades": num_trades,
        "final_balance": stats['total_value'],
        "pnl": stats['total_pnl'],
        "returns": stats['returns'],
    }


# ============================================================================
# SECTION 4: LIVE TRADING SYSTEM
# ============================================================================


class LiveTradingSystem:
    """
    Complete live trading system with error handling.

    This class orchestrates all components for live trading:
    - WebSocket streaming
    - Feature computation
    - Model inference
    - Order execution
    - Portfolio management
    - Safety guards

    Key features:
    - Graceful error handling
    - Auto-reconnection
    - Safety circuit breakers
    - Real-time monitoring
    """

    def __init__(
        self,
        symbol: str,
        model_path: str,
        vecnormalize_path: Optional[str] = None,
        initial_balance: float = 10000,
        max_drawdown: float = 0.20,
        max_trades_per_hour: int = 20,
        trading_hours: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize live trading system.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            model_path: Path to trained model
            vecnormalize_path: Optional VecNormalize path
            initial_balance: Starting balance in USD
            max_drawdown: Maximum allowed drawdown (0-1)
            max_trades_per_hour: Rate limit
            trading_hours: Optional (start_hour, end_hour) in UTC
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.running = False

        # Load model
        console.print("[cyan]Initializing trading system...[/cyan]")
        self.inference_engine = load_and_inspect_model(model_path, vecnormalize_path)

        # Initialize portfolio
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            symbols=[symbol],
            db_path="portfolio.db",
        )

        # Initialize custom safety guard
        self.safety_guard = CustomSafetyGuard(
            max_drawdown=max_drawdown,
            max_trades_per_hour=max_trades_per_hour,
            initial_balance=initial_balance,
            trading_hours=trading_hours,
        )

        # Initialize feature computer
        self.feature_computer = FeatureComputer(
            symbol=symbol,
            lookback_window=100,
            min_bars_required=21,
        )

        # Initialize executor
        api_key = os.getenv("BINANCE_TESTNET_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET")

        if not api_key or not api_secret:
            raise ValueError(
                "Missing Binance API credentials!\n"
                "Set BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET environment variables\n"
                "Get keys from: https://testnet.binance.vision/"
            )

        self.executor = OrderExecutor(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )

        # Initialize dashboard
        self.dashboard = TradingDashboard(
            refresh_rate=1.0,
            show_predictions=True,
        )

        console.print("[green]✓ Trading system initialized[/green]")

    async def on_bar(self, symbol: str, bar: pd.DataFrame):
        """
        Main trading logic - called when a new bar is created.

        This is the heart of the trading system. Steps:
        1. Add bar to feature computer
        2. Check if enough bars for features
        3. Compute features
        4. Get model prediction
        5. Check safety guards
        6. Execute trade if allowed
        7. Update portfolio and dashboard

        Args:
            symbol: Trading symbol
            bar: New bar DataFrame
        """
        try:
            # Step 1: Add bar and check if ready
            is_ready = self.feature_computer.add_bar(bar)
            if not is_ready:
                logger.debug(f"{symbol}: Accumulating bars (need 21 minimum)")
                return

            # Step 2: Compute features
            features = self.feature_computer.get_latest_features()
            if features is None:
                logger.warning(f"{symbol}: Failed to compute features")
                return

            # Step 3: Get model prediction
            action, confidence = self.inference_engine.predict(
                features, deterministic=True
            )
            action_name = self.inference_engine.get_action_name(action)

            logger.info(
                f"{symbol}: Prediction = {action_name} (confidence={confidence:.2f})"
            )

            # Update dashboard with prediction
            self.dashboard.update(
                predictions={symbol: (action_name, confidence)}
            )

            # Skip if HOLD
            if action == 0:
                return

            # Step 4: Check safety guards
            can_trade, reason = self.safety_guard.can_trade(symbol)
            if not can_trade:
                logger.warning(f"{symbol}: Trading not allowed: {reason}")
                self.dashboard.update(
                    safety_stats=self.safety_guard.get_custom_stats()
                )
                return

            # Step 5: Execute trade
            current_price = float(bar.iloc[0]["close"])

            # Update portfolio with current price
            self.portfolio.update_prices({symbol: current_price})

            result = self.executor.execute_action(
                symbol=symbol,
                action=action,
                price=current_price,
                max_position_size=1000,
            )

            # Step 6: Update portfolio if executed
            if result["executed"]:
                self.portfolio.record_trade(
                    symbol=symbol,
                    action=result["action"],
                    quantity=result["quantity"],
                    price=result["price"],
                    commission=result["commission"],
                )

                # Update safety guard
                pnl = result.get("pnl", 0)
                self.safety_guard.record_trade(
                    symbol=symbol,
                    pnl=pnl,
                    new_balance=self.portfolio.get_total_value(),
                    trade_details=result,
                )

                logger.info(f"{symbol}: {result['message']}")

            # Step 7: Update dashboard
            self._update_dashboard()

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)
            # Don't crash - log and continue

    def _update_dashboard(self):
        """Update dashboard with latest data."""
        self.dashboard.update(
            portfolio_stats=self.portfolio.get_stats(),
            recent_trades=self.portfolio.get_trade_history(limit=10).to_dict("records"),
            safety_stats=self.safety_guard.get_custom_stats(),
        )
        self.dashboard.refresh()

    async def run(self):
        """
        Main run loop with error handling.

        Handles:
        - WebSocket connection errors
        - Auto-reconnection
        - Graceful shutdown
        - Final summary
        """
        logger.info(f"Starting live trading for {self.symbol}...")

        # Create stream consumer
        api_key = os.getenv("BINANCE_TESTNET_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET")

        consumer = StreamConsumer(
            symbols=[self.symbol],
            on_bar_callback=self.on_bar,
            dollar_volume_thresholds={self.symbol: 1_000_000},
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )

        # Start dashboard
        with self.dashboard.live():
            try:
                self.running = True
                logger.info(f"Trading live on {self.symbol}")

                # Start streaming
                await consumer.start()

            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")
            except Exception as e:
                logger.error(f"Fatal error: {e}", exc_info=True)
            finally:
                self.running = False
                await consumer.stop()

        # Print final summary
        self._print_summary()

    def _print_summary(self):
        """Print final trading session summary."""
        console.print("\n" + "="*60)
        console.print("[bold cyan]Trading Session Summary[/bold cyan]")
        console.print("="*60)

        stats = self.portfolio.get_stats()

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Symbol", self.symbol)
        table.add_row("Initial Balance", f"${self.initial_balance:,.2f}")
        table.add_row("Final Balance", f"${stats['total_value']:,.2f}")
        table.add_row("Total PnL", f"${stats['total_pnl']:+,.2f}")
        table.add_row("Returns", f"{stats['returns']*100:+.2f}%")
        table.add_row("Total Trades", str(stats['total_trades']))
        table.add_row("Commission Paid", f"${stats['total_commission']:,.2f}")

        safety_stats = self.safety_guard.get_custom_stats()
        table.add_row("Circuit Breaker", safety_stats['state'])
        table.add_row("Violations", str(safety_stats['violations']))

        console.print(table)


# ============================================================================
# SECTION 5: TRADE ANALYSIS
# ============================================================================


def analyze_trading_session(db_path: str = "portfolio.db"):
    """
    Analyze past trading session from database.

    This function demonstrates:
    - Loading trade history from SQLite
    - Calculating performance metrics
    - Generating statistics
    - Exporting results

    Args:
        db_path: Path to portfolio database
    """
    console.print(Panel.fit(
        "[bold cyan]Trade Analysis[/bold cyan]\n\n"
        f"Database: {db_path}",
        border_style="cyan"
    ))

    if not Path(db_path).exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        return

    # Load trade history
    import sqlite3

    conn = sqlite3.connect(db_path)
    trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
    conn.close()

    if len(trades_df) == 0:
        console.print("[yellow]No trades found in database[/yellow]")
        return

    console.print(f"\n[cyan]Total trades:[/cyan] {len(trades_df)}")

    # Calculate statistics
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['pnl'] = trades_df['pnl'].fillna(0)

    # Basic stats
    total_pnl = trades_df['pnl'].sum()
    total_commission = trades_df['commission'].sum()
    net_pnl = total_pnl - total_commission

    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] < 0]

    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

    # Create statistics table
    table = Table(title="Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Trades", str(len(trades_df)))
    table.add_row("Winning Trades", str(len(winning_trades)))
    table.add_row("Losing Trades", str(len(losing_trades)))
    table.add_row("Win Rate", f"{win_rate*100:.1f}%")
    table.add_row("Total PnL", f"${total_pnl:+,.2f}")
    table.add_row("Total Commission", f"${total_commission:,.2f}")
    table.add_row("Net PnL", f"${net_pnl:+,.2f}")
    table.add_row("Average Win", f"${avg_win:+,.2f}")
    table.add_row("Average Loss", f"${avg_loss:+,.2f}")

    if avg_loss != 0:
        profit_factor = abs(avg_win * len(winning_trades)) / abs(avg_loss * len(losing_trades))
        table.add_row("Profit Factor", f"{profit_factor:.2f}")

    console.print(table)

    # Show recent trades
    console.print("\n[cyan]Recent Trades:[/cyan]")
    recent_table = Table()
    recent_table.add_column("Time", style="cyan")
    recent_table.add_column("Symbol", style="yellow")
    recent_table.add_column("Action", style="magenta")
    recent_table.add_column("Price", style="green")
    recent_table.add_column("PnL", style="green")

    for _, trade in trades_df.tail(10).iterrows():
        pnl_color = "green" if trade['pnl'] > 0 else "red"
        recent_table.add_row(
            trade['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            trade['symbol'],
            trade['action'],
            f"${trade['price']:,.2f}",
            f"[{pnl_color}]${trade['pnl']:+,.2f}[/{pnl_color}]",
        )

    console.print(recent_table)

    # Export to CSV
    export_path = Path("trade_analysis.csv")
    trades_df.to_csv(export_path, index=False)
    console.print(f"\n[green]✓ Exported to {export_path}[/green]")


# ============================================================================
# SECTION 6: CLI COMMANDS
# ============================================================================


@app.command()
def validate(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model (.zip file)",
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading symbol",
    ),
    days: int = typer.Option(
        1,
        "--days",
        "-d",
        help="Number of days of historical data to validate",
    ),
    vecnormalize: Optional[str] = typer.Option(
        None,
        "--vecnormalize",
        help="Path to VecNormalize file (.pkl)",
    ),
):
    """
    Validate trading pipeline with historical data (no real trading).

    This is the FIRST step before live trading. Tests the entire pipeline
    with historical data to ensure everything works correctly.

    Example:
        uv run python examples/live_trading_example.py validate \\
            --model checkpoints/PPO_returns_20251028_143659/best_model.zip \\
            --days 1
    """
    try:
        results = validate_with_historical_data(
            symbol=symbol,
            model_path=model,
            vecnormalize_path=vecnormalize,
            days=days,
        )

        if results["success"]:
            console.print("\n[green]✓ System ready for live trading![/green]")
            sys.exit(0)
        else:
            console.print(f"\n[red]✗ Validation failed: {results['error']}[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Validation error: {e}[/red]")
        sys.exit(1)


@app.command()
def trade(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to trained model (.zip file)",
    ),
    symbol: str = typer.Option(
        "BTCUSDT",
        "--symbol",
        "-s",
        help="Trading symbol",
    ),
    balance: float = typer.Option(
        10000,
        "--balance",
        "-b",
        help="Initial balance in USD",
    ),
    max_drawdown: float = typer.Option(
        0.20,
        "--max-drawdown",
        help="Maximum drawdown before circuit breaker (0-1)",
    ),
    vecnormalize: Optional[str] = typer.Option(
        None,
        "--vecnormalize",
        help="Path to VecNormalize file (.pkl)",
    ),
):
    """
    Run live trading on Binance testnet.

    IMPORTANT: This uses TESTNET with fake money. Get API keys from:
    https://testnet.binance.vision/

    Set environment variables:
        export BINANCE_TESTNET_KEY="your_key"
        export BINANCE_TESTNET_SECRET="your_secret"

    Example:
        uv run python examples/live_trading_example.py trade \\
            --model checkpoints/PPO_returns_20251028_143659/best_model.zip \\
            --symbol BTCUSDT \\
            --balance 10000
    """
    try:
        # Create trading system
        system = LiveTradingSystem(
            symbol=symbol,
            model_path=model,
            vecnormalize_path=vecnormalize,
            initial_balance=balance,
            max_drawdown=max_drawdown,
            trading_hours=(9, 21),  # Only trade 9am-9pm UTC
        )

        # Run
        asyncio.run(system.run())

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def analyze(
    db: str = typer.Option(
        "portfolio.db",
        "--db",
        help="Path to portfolio database",
    ),
):
    """
    Analyze past trading session from database.

    Example:
        uv run python examples/live_trading_example.py analyze --db portfolio.db
    """
    try:
        analyze_trading_session(db)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()
