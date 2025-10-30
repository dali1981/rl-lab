"""
Portfolio manager for multi-symbol trading.

Tracks positions, PnL, and trade history across multiple symbols.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import sqlite3
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages a multi-symbol trading portfolio.

    Tracks:
    - Current positions per symbol
    - Realized and unrealized PnL
    - Trade history
    - Portfolio statistics

    Example:
        >>> portfolio = PortfolioManager(
        ...     initial_balance=10000,
        ...     symbols=["BTCUSDT", "ETHUSDT"]
        ... )
        >>>
        >>> # Update position after trade
        >>> portfolio.record_trade("BTCUSDT", "BUY", 0.1, 67000, commission=6.7)
        >>>
        >>> # Update current prices
        >>> portfolio.update_prices({"BTCUSDT": 68000, "ETHUSDT": 3500})
        >>>
        >>> # Get portfolio stats
        >>> stats = portfolio.get_stats()
    """

    def __init__(
        self,
        initial_balance: float,
        symbols: List[str],
        db_path: str = "portfolio.db",
    ):
        """
        Initialize the portfolio manager.

        Args:
            initial_balance: Starting cash balance (USD)
            symbols: List of trading symbols
            db_path: Path to SQLite database for trade history
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.symbols = symbols

        # Current positions per symbol
        self.positions: Dict[str, Dict] = {
            symbol: {
                "quantity": 0.0,
                "entry_price": 0.0,
                "current_price": 0.0,
                "value": 0.0,
                "unrealized_pnl": 0.0,
            }
            for symbol in symbols
        }

        # Portfolio metrics
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_trades = 0

        # Current prices
        self.current_prices: Dict[str, float] = {symbol: 0.0 for symbol in symbols}

        # Initialize database
        self.db_path = Path(db_path)
        self._init_database()

        logger.info(
            f"Initialized PortfolioManager "
            f"(balance=${initial_balance:,.2f}, {len(symbols)} symbols)"
        )

    def _init_database(self):
        """Initialize SQLite database for trade history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    commission REAL NOT NULL,
                    pnl REAL,
                    balance REAL NOT NULL
                )
            """)
            conn.commit()

        logger.info(f"Initialized trade history database: {self.db_path}")

    def record_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
    ) -> Dict:
        """
        Record a trade and update portfolio state.

        Args:
            symbol: Trading symbol
            action: "BUY" or "SELL"
            quantity: Trade quantity
            price: Execution price
            commission: Trading commission

        Returns:
            Dict with trade details and updated portfolio state
        """
        if symbol not in self.positions:
            logger.error(f"Unknown symbol: {symbol}")
            return {}

        value = quantity * price
        position = self.positions[symbol]
        pnl = 0.0

        if action == "BUY":
            # Deduct cash
            cost = value + commission
            if cost > self.cash_balance:
                logger.error(f"Insufficient balance: ${self.cash_balance:.2f} < ${cost:.2f}")
                return {}

            self.cash_balance -= cost

            # Update position
            position["quantity"] = quantity
            position["entry_price"] = price
            position["current_price"] = price
            position["value"] = value
            position["unrealized_pnl"] = 0.0

        elif action == "SELL":
            # Add cash
            self.cash_balance += value - commission

            # Calculate PnL
            if position["quantity"] > 0:
                entry_value = position["quantity"] * position["entry_price"]
                pnl = value - entry_value - commission * 2  # Buy + sell commission
                self.realized_pnl += pnl

            # Clear position
            position["quantity"] = 0.0
            position["entry_price"] = 0.0
            position["current_price"] = price
            position["value"] = 0.0
            position["unrealized_pnl"] = 0.0

        else:
            logger.error(f"Unknown action: {action}")
            return {}

        # Update commission
        self.total_commission += commission
        self.total_trades += 1

        # Record to database
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": value,
            "commission": commission,
            "pnl": pnl if action == "SELL" else None,
            "balance": self.get_total_value(),
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades
                (timestamp, symbol, action, quantity, price, value, commission, pnl, balance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_record["timestamp"],
                trade_record["symbol"],
                trade_record["action"],
                trade_record["quantity"],
                trade_record["price"],
                trade_record["value"],
                trade_record["commission"],
                trade_record["pnl"],
                trade_record["balance"],
            ))
            conn.commit()

        logger.info(
            f"Recorded {action}: {quantity:.8f} {symbol} @ ${price:.2f} "
            f"(balance=${self.get_total_value():.2f})"
        )

        return trade_record

    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices and recalculate unrealized PnL.

        Args:
            prices: Dict mapping symbol to current price
        """
        for symbol, price in prices.items():
            if symbol not in self.positions:
                continue

            self.current_prices[symbol] = price
            position = self.positions[symbol]

            if position["quantity"] > 0:
                position["current_price"] = price
                position["value"] = position["quantity"] * price
                position["unrealized_pnl"] = (
                    position["value"] -
                    position["quantity"] * position["entry_price"]
                )

    def get_position(self, symbol: str) -> Dict:
        """Get current position for a symbol."""
        return self.positions.get(symbol, {})

    def get_total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        position_value = sum(
            pos["value"] for pos in self.positions.values()
        )
        return self.cash_balance + position_value

    def get_total_pnl(self) -> float:
        """Get total PnL (realized + unrealized)."""
        unrealized_pnl = sum(
            pos["unrealized_pnl"] for pos in self.positions.values()
        )
        return self.realized_pnl + unrealized_pnl

    def get_drawdown(self) -> float:
        """Get current drawdown from initial balance."""
        current_value = self.get_total_value()
        drawdown = (self.initial_balance - current_value) / self.initial_balance
        return max(0, drawdown)

    def get_returns(self) -> float:
        """Get returns since inception."""
        return (self.get_total_value() - self.initial_balance) / self.initial_balance

    def get_stats(self) -> Dict:
        """
        Get portfolio statistics.

        Returns:
            Dict with portfolio metrics
        """
        total_value = self.get_total_value()
        total_pnl = self.get_total_pnl()

        stats = {
            "initial_balance": self.initial_balance,
            "cash_balance": self.cash_balance,
            "position_value": sum(pos["value"] for pos in self.positions.values()),
            "total_value": total_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(
                pos["unrealized_pnl"] for pos in self.positions.values()
            ),
            "total_pnl": total_pnl,
            "returns": self.get_returns(),
            "drawdown": self.get_drawdown(),
            "total_commission": self.total_commission,
            "total_trades": self.total_trades,
            "active_positions": sum(
                1 for pos in self.positions.values() if pos["quantity"] > 0
            ),
        }

        # Per-symbol stats
        stats["positions"] = {}
        for symbol, pos in self.positions.items():
            if pos["quantity"] > 0:
                stats["positions"][symbol] = {
                    "quantity": pos["quantity"],
                    "entry_price": pos["entry_price"],
                    "current_price": pos["current_price"],
                    "value": pos["value"],
                    "unrealized_pnl": pos["unrealized_pnl"],
                    "unrealized_pnl_pct": (
                        pos["unrealized_pnl"] / (pos["quantity"] * pos["entry_price"])
                        if pos["quantity"] > 0 else 0
                    ),
                }

        return stats

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Get recent trade history.

        Args:
            limit: Maximum number of trades to return

        Returns:
            DataFrame with trade history
        """
        with sqlite3.connect(self.db_path) as conn:
            query = f"""
                SELECT * FROM trades
                ORDER BY timestamp DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def reset(self):
        """Reset portfolio to initial state (for testing)."""
        self.cash_balance = self.initial_balance
        self.realized_pnl = 0.0
        self.total_commission = 0.0
        self.total_trades = 0

        for symbol in self.positions:
            self.positions[symbol] = {
                "quantity": 0.0,
                "entry_price": 0.0,
                "current_price": 0.0,
                "value": 0.0,
                "unrealized_pnl": 0.0,
            }

        logger.info("Reset portfolio to initial state")
