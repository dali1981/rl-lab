#!/usr/bin/env python3
"""
Live Trading Runner - Orchestrates all components for live trading.

This script connects all the pieces:
- WebSocket streaming
- Feature computation
- Model inference
- Order execution
- Portfolio management
- Safety guards
- Real-time dashboard

Example usage:
    # Test with testnet
    uv run python experiments/live_trading.py \\
        --config configs/trading/testnet.yaml \\
        --symbols BTCUSDT

    # With specific model
    uv run python experiments/live_trading.py \\
        --config configs/trading/testnet.yaml \\
        --symbols BTCUSDT \\
        --model checkpoints/PPO_returns_20251028_143659/best_model.zip
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List
import yaml
import typer
from rich.logging import RichHandler

from rl_trading_lab.live import (
    StreamConsumer,
    FeatureComputer,
    ModelInferenceEngine,
    OrderExecutor,
    PortfolioManager,
)
from rl_trading_lab.live.safety import SafetyGuard
from rl_trading_lab.live.dashboard import TradingDashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

app = typer.Typer()


class LiveTradingSystem:
    """
    Main live trading system that orchestrates all components.

    Architecture:
    1. WebSocket → bars
    2. Bars → features
    3. Features → predictions
    4. Predictions → actions
    5. Actions → orders
    6. Orders → portfolio updates
    7. Safety guards monitor everything
    8. Dashboard displays real-time status
    """

    def __init__(self, config: Dict):
        """Initialize trading system from config."""
        self.config = config
        self.symbols = config["symbols"]

        # Initialize components
        self._init_portfolio()
        self._init_safety_guard()
        self._init_feature_computers()
        self._init_inference_engines()
        self._init_executor()
        self._init_dashboard()

        # State
        self.running = False

    def _init_portfolio(self):
        """Initialize portfolio manager."""
        risk_config = self.config["risk"]
        self.portfolio = PortfolioManager(
            initial_balance=risk_config["initial_balance"],
            symbols=self.symbols,
            db_path="portfolio.db",
        )
        logger.info(f"Portfolio initialized with ${risk_config['initial_balance']:,.2f}")

    def _init_safety_guard(self):
        """Initialize safety guard."""
        risk_config = self.config["risk"]
        safety_config = self.config.get("safety", {})

        self.safety_guard = SafetyGuard(
            max_drawdown=risk_config.get("max_drawdown", 0.20),
            max_trades_per_hour=risk_config.get("max_trades_per_hour", 20),
            max_trades_per_day=risk_config.get("max_trades_per_day", 100),
            initial_balance=risk_config["initial_balance"],
            max_position_pct=risk_config.get("max_position_pct", 0.95),
            enable_circuit_breaker=safety_config.get("enable_circuit_breaker", True),
        )
        logger.info("Safety guard initialized")

    def _init_feature_computers(self):
        """Initialize feature computers for each symbol."""
        feature_config = self.config.get("features", {})

        self.feature_computers: Dict[str, FeatureComputer] = {}
        for symbol in self.symbols:
            self.feature_computers[symbol] = FeatureComputer(
                symbol=symbol,
                lookback_window=feature_config.get("lookback_window", 100),
                min_bars_required=feature_config.get("min_bars_required", 21),
            )

        logger.info(f"Feature computers initialized for {len(self.symbols)} symbols")

    def _init_inference_engines(self):
        """Initialize model inference engines."""
        model_config = self.config.get("models", {})

        self.inference_engines: Dict[str, ModelInferenceEngine] = {}
        for symbol in self.symbols:
            if symbol in model_config:
                model_info = model_config[symbol]
                self.inference_engines[symbol] = ModelInferenceEngine(
                    model_path=model_info["path"],
                    vecnormalize_path=model_info.get("vecnormalize"),
                    model_type=model_info.get("type"),
                )
                logger.info(f"Loaded model for {symbol}: {model_info['path']}")
            else:
                logger.warning(f"No model configured for {symbol}, will skip trading")

    def _init_executor(self):
        """Initialize order executor."""
        testnet_config = self.config.get("testnet", {})
        risk_config = self.config["risk"]

        # Get API credentials from environment or config
        import os
        api_key = os.getenv("BINANCE_TESTNET_KEY") or testnet_config.get("api_key")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET") or testnet_config.get("api_secret")

        if not api_key or not api_secret:
            logger.error("Missing Binance API credentials!")
            logger.error("Set BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET environment variables")
            sys.exit(1)

        self.executor = OrderExecutor(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            min_order_size=risk_config.get("min_order_size", 10.0),
        )
        logger.info("Order executor initialized (testnet mode)")

    def _init_dashboard(self):
        """Initialize dashboard."""
        dashboard_config = self.config.get("dashboard", {})

        self.dashboard = TradingDashboard(
            refresh_rate=dashboard_config.get("refresh_rate", 1.0),
            show_predictions=dashboard_config.get("show_predictions", True),
            show_features=dashboard_config.get("show_features", False),
        )
        logger.info("Dashboard initialized")

    async def on_bar(self, symbol: str, bar):
        """
        Callback when a new bar is created.

        This is the main trading logic loop:
        1. Add bar to feature computer
        2. If features ready, compute them
        3. If model available, predict action
        4. If action is BUY/SELL, check safety and execute
        5. Update portfolio and dashboard
        """
        try:
            # Add bar to feature computer
            feature_computer = self.feature_computers[symbol]
            is_ready = feature_computer.add_bar(bar)

            if not is_ready:
                logger.debug(f"{symbol}: Not enough bars yet for features")
                return

            # Compute features
            features = feature_computer.get_latest_features()
            if features is None:
                logger.warning(f"{symbol}: Failed to compute features")
                return

            # Get model prediction
            if symbol not in self.inference_engines:
                return

            engine = self.inference_engines[symbol]
            action, confidence = engine.predict(features, deterministic=True)
            action_name = engine.get_action_name(action)

            logger.info(
                f"{symbol}: Prediction = {action_name} (confidence={confidence:.2f})"
            )

            # Update dashboard predictions
            self.dashboard.update(
                predictions={symbol: (action_name, confidence)}
            )

            # Skip if HOLD
            if action == 0:  # HOLD
                return

            # Check safety guard
            can_trade, reason = self.safety_guard.can_trade(symbol)
            if not can_trade:
                logger.warning(f"{symbol}: Trading not allowed: {reason}")
                return

            # Get current price (from bar)
            current_price = float(bar.iloc[0]["close"])

            # Update portfolio with current prices
            self.portfolio.update_prices({symbol: current_price})

            # Execute trade
            risk_config = self.config["risk"]
            max_position_size = risk_config.get("max_position_size", 1000)

            result = self.executor.execute_action(
                symbol=symbol,
                action=action,
                price=current_price,
                max_position_size=max_position_size,
            )

            # If trade was executed, update portfolio
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

            # Update dashboard
            self._update_dashboard()

        except Exception as e:
            logger.error(f"Error in on_bar for {symbol}: {e}", exc_info=True)

    def _update_dashboard(self):
        """Update dashboard with latest data."""
        self.dashboard.update(
            portfolio_stats=self.portfolio.get_stats(),
            recent_trades=self.portfolio.get_trade_history(limit=10).to_dict("records"),
            safety_stats=self.safety_guard.get_stats(),
        )
        self.dashboard.refresh()

    async def run(self):
        """
        Main run loop.

        Starts:
        1. Dashboard
        2. WebSocket streams
        3. Trading logic
        """
        logger.info("Starting live trading system...")

        # Get dollar volume thresholds
        thresholds = self.config.get("dollar_volume_thresholds", {})

        # Create stream consumer
        testnet_config = self.config.get("testnet", {})
        import os
        api_key = os.getenv("BINANCE_TESTNET_KEY") or testnet_config.get("api_key")
        api_secret = os.getenv("BINANCE_TESTNET_SECRET") or testnet_config.get("api_secret")

        consumer = StreamConsumer(
            symbols=self.symbols,
            on_bar_callback=self.on_bar,
            dollar_volume_thresholds=thresholds,
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
        )

        # Start dashboard
        with self.dashboard.live():
            try:
                self.running = True
                logger.info(f"Trading live on {', '.join(self.symbols)}")

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
        logger.info("\n" + "="*60)
        logger.info("Trading Session Summary")
        logger.info("="*60)

        stats = self.portfolio.get_stats()
        logger.info(f"Final Balance: ${stats['total_value']:,.2f}")
        logger.info(f"Total PnL: ${stats['total_pnl']:+,.2f}")
        logger.info(f"Returns: {stats['returns']*100:+.2f}%")
        logger.info(f"Total Trades: {stats['total_trades']}")
        logger.info(f"Commission Paid: ${stats['total_commission']:,.2f}")

        safety_stats = self.safety_guard.get_stats()
        logger.info(f"Circuit Breaker: {safety_stats['state']}")
        logger.info(f"Violations: {safety_stats['violations']}")


@app.command()
def main(
    config: str = typer.Option(
        "configs/trading/testnet.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    symbols: List[str] = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Symbols to trade (overrides config)",
    ),
):
    """
    Run live trading system.

    Connects to Binance WebSocket, receives real-time trade data,
    creates dollar volume bars, computes features, makes predictions,
    and executes trades on testnet/live.
    """
    # Load configuration
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Override symbols if provided
    if symbols:
        config_dict["symbols"] = symbols

    logger.info(f"Loaded config: {config_path}")
    logger.info(f"Trading symbols: {', '.join(config_dict['symbols'])}")

    # Create and run system
    system = LiveTradingSystem(config_dict)

    try:
        asyncio.run(system.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
