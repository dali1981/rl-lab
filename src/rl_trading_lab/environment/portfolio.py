"""
Portfolio management for trading environments.
Handles cash, positions, and trade execution accounting.

ACCOUNTING MODEL: SPOT TRADING
==============================
This portfolio implements a spot trading model where you actually pay for
positions with cash (not margin/leverage). The accounting works as follows:

CASH FLOW:
- Opening a position: cash -= (position_value + commission)
  * LONG: You pay cash to buy the asset
  * SHORT: You receive cash from selling (borrowed) asset

- Closing a position: cash += (position_value - commission)
  * LONG: You receive cash from selling the asset
  * SHORT: You pay cash to buy back the (borrowed) asset

PORTFOLIO VALUE:
- Total value = cash + position_value
  where position_value = position.size * current_price
  * For LONG positions: position.size > 0, adds to value
  * For SHORT positions: position.size < 0, subtracts from value

POSITION SIZING:
- max_position_pct limits how much of your current cash can be deployed
- Example: With 10,000 cash and max_position_pct=0.95, you can deploy up to 9,500
- After opening, you'll have ~500 cash left (plus commission costs)

This model ensures:
1. You can't trade with money you don't have
2. Portfolio value correctly reflects mark-to-market value
3. Cash balance is meaningful (it's your buying power)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track current position state"""
    size: float = 0.0  # Position size (positive=long, negative=short, 0=flat)
    entry_price: float = 0.0
    entry_bar: int = 0
    current_bar: int = 0


class Cash:
    """
    Manages cash balance with explicit debit/credit operations.

    This class encapsulates cash management using standard accounting terminology:
    - debit(): Remove cash (paying for something)
    - credit(): Add cash (receiving payment)

    This is much clearer than manual sign handling and prevents errors.
    """

    def __init__(self, initial_balance: float):
        """
        Initialize cash account.

        Args:
            initial_balance: Starting cash amount
        """
        self._balance = initial_balance
        self.initial_balance = initial_balance

    @property
    def balance(self) -> float:
        """Get current cash balance"""
        return self._balance

    def debit(self, amount: float) -> float:
        """
        Remove cash from account (payment/cost).

        Args:
            amount: Amount to deduct (sign is ignored, always debits)

        Returns:
            Absolute amount debited
        """
        amount_abs = abs(amount)
        self._balance -= amount_abs
        return amount_abs

    def credit(self, amount: float) -> float:
        """
        Add cash to account (receipt/income).

        Args:
            amount: Amount to add (sign is ignored, always credits)

        Returns:
            Absolute amount credited
        """
        amount_abs = abs(amount)
        self._balance += amount_abs
        return amount_abs

    def has_funds(self, amount: float) -> bool:
        """
        Check if sufficient funds available.

        Args:
            amount: Required amount (absolute value)

        Returns:
            True if balance >= amount
        """
        return self._balance >= abs(amount)

    def reset(self) -> None:
        """Reset cash to initial balance"""
        self._balance = self.initial_balance


class Portfolio:
    """
    Manages portfolio accounting including cash, positions, and trade execution.

    Implements SPOT TRADING model (see module docstring for details).

    Attributes:
        cash: Cash instance managing available balance
        position: Current position state (size, entry_price, entry_bar)
        num_trades: Count of executed trades
        trade_history: List of completed round-trip trades

    Key Methods:
        - get_portfolio_value(): cash + position_value (mark-to-market)
        - execute_trade(): Open/close positions with proper cash accounting
        - _open_position(): Debit/credit cash for position opening
        - close_position(): Debit/credit cash for position closing
    """

    def __init__(
        self,
        initial_cash: float = 10000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_pct: float = 0.95,
        min_holding_period: int = 1,
    ):
        """
        Initialize portfolio with spot trading model.

        Args:
            initial_cash: Starting cash amount
            commission_rate: Commission as fraction of trade value (e.g., 0.001 = 0.1%)
            slippage_rate: Slippage as fraction of price (e.g., 0.0005 = 0.05%)
            max_position_pct: Maximum position size as fraction of current cash.
                In spot trading, this limits capital deployment (e.g., 0.95 = use up to 95% of cash)
            min_holding_period: Minimum bars to hold position before allowing close
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_pct = max_position_pct
        self.min_holding_period = min_holding_period

        # Account state
        self.cash = Cash(initial_cash)
        self.position = Position()

        # Trade tracking
        self.num_trades = 0
        self.trade_history: List[Dict[str, Any]] = []
        self._open_trade: Optional[Dict[str, Any]] = None

    def reset(self):
        """Reset portfolio to initial state"""
        self.cash.reset()
        self.position = Position()
        self.num_trades = 0
        self.trade_history.clear()
        self._open_trade = None

    def get_portfolio_value(self, current_price: float) -> float:
        """
        Get total portfolio value using spot trading model.

        Formula: portfolio_value = cash + position_value
        where position_value = position.size * current_price

        This works correctly for both longs and shorts:
        - LONG: position.size > 0, so position_value adds to total
        - SHORT: position.size < 0, so position_value subtracts from total

        Args:
            current_price: Current market price

        Returns:
            Total portfolio value (mark-to-market)
        """
        if self.position.size == 0:
            return self.cash.balance

        # Spot trading: value = cash + position value (not unrealized P&L)
        position_value = self.position.size * current_price
        return self.cash.balance + position_value

    def get_position_pnl(self, current_price: float) -> float:
        """
        Get unrealized P&L for current position.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L (0 if no position)
        """
        if self.position.size == 0:
            return 0.0

        pnl = self.position.size * (current_price - self.position.entry_price)
        return pnl

    def calculate_position_size(self, signal: float, current_price: float) -> float:
        """
        Calculate position size based on available cash and signal.

        Args:
            signal: Trading signal (-1 for short, +1 for long)
            current_price: Current market price

        Returns:
            Position size (signed)
        """
        if signal == 0:
            return 0.0

        max_position_value = self.cash.balance * self.max_position_pct
        position_size = (max_position_value / current_price) * np.sign(signal)
        return position_size

    def should_close_position(self, signal: float, current_step: int) -> bool:
        """
        Determine if current position should be closed.

        Args:
            signal: Trading signal (-1, 0, +1)
            current_step: Current bar/step number

        Returns:
            True if position should be closed
        """
        # No position to close
        if self.position.size == 0:
            return False

        # Don't close if minimum holding period not met
        bars_held = current_step - self.position.entry_bar
        if bars_held < self.min_holding_period:
            logger.debug(f"Position held for {bars_held} bars, need {self.min_holding_period} bars minimum")
            return False

        # Close if taking opposite direction
        if np.sign(signal) != np.sign(self.position.size):
            return True

        return False

    def execute_trade(
        self,
        signal: float,
        current_price: float,
        current_step: int,
        df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """
        Execute a trade based on signal.

        Handles closing existing positions and opening new ones.

        Args:
            signal: Trading signal (-1 for short, 0 for flat, +1 for long)
            current_price: Current market price
            current_step: Current bar/step number
            df: Optional dataframe for timestamp tracking

        Returns:
            True if a trade was executed
        """
        # Check if we should close current position
        if self.should_close_position(signal, current_step):
            logger.debug(f"Closing position: current={self.position.size:.4f}, signal={signal:.1f}")
            self.close_position(current_price, current_step, df)
            return True

        # Open new position if flat and signal is non-zero
        if self.position.size == 0 and signal != 0:
            logger.debug(f"Opening position: signal={signal:.1f}, price={current_price:.2f}")
            self._open_position(signal, current_price, current_step, df)
            logger.debug(f"Position opened: size={self.position.size:.4f}")
            return True

        return False

    def _open_position(
        self,
        signal: float,
        current_price: float,
        current_step: int,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Open a new position using spot trading model with explicit cash operations.

        CASH FLOW (SPOT TRADING):
        - LONG (signal > 0):
          * Debit trade value (pay cash to buy asset)
          * Debit commission
        - SHORT (signal < 0):
          * Credit trade value (receive cash from selling borrowed asset)
          * Debit commission

        Args:
            signal: Trading signal (-1 for short, +1 for long)
            current_price: Current market price
            current_step: Current bar/step number
            df: Optional dataframe for timestamp tracking
        """
        if signal == 0:
            return

        # Calculate position size based on available cash
        position_size = self.calculate_position_size(signal, current_price)

        # Apply slippage
        execution_price = current_price * (1 + self.slippage_rate * np.sign(signal))

        # Calculate trade costs
        trade_value = abs(position_size * execution_price)
        commission = trade_value * self.commission_rate

        # Check if we have enough cash for the trade (worst case: LONG)
        if not self.cash.has_funds(trade_value + commission):
            return

        # Update position
        self.position.size = position_size
        self.position.entry_price = execution_price
        self.position.entry_bar = current_step

        # Execute cash flow based on position type
        if signal > 0:  # LONG: Pay cash to buy
            debited = self.cash.debit(trade_value + commission)
            cash_flow_description = f"-${debited:.2f}"
        else:  # SHORT: Receive cash from selling, pay commission
            credited = self.cash.credit(trade_value)
            debited = self.cash.debit(commission)
            cash_flow_description = f"+${credited:.2f}, -${debited:.2f}"

        # Increment trade counter
        self.num_trades += 1
        logger.debug(
            f"Trade #{self.num_trades}: {'LONG' if signal > 0 else 'SHORT'} "
            f"{abs(position_size):.4f} @ ${execution_price:.2f} "
            f"(cash flow: {cash_flow_description}, remaining: ${self.cash.balance:.2f})"
        )

        # Track open trade for history
        timestamp = None
        if df is not None and 'timestamp' in df.columns:
            timestamp = df.iloc[current_step].get('timestamp', None)

        self._open_trade = {
            'trade_id': self.num_trades,
            'open_step': current_step,
            'open_timestamp': timestamp,
            'side': 'LONG' if signal > 0 else 'SHORT',
            'entry_price': execution_price,
            'position_size': position_size,
            'entry_commission': commission,
        }

    def close_position(
        self,
        current_price: float,
        current_step: int,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Close current position using spot trading model with explicit cash operations.

        CASH FLOW (SPOT TRADING):
        - Closing LONG (size > 0):
          * Credit trade value (receive cash from selling asset)
          * Debit commission
        - Closing SHORT (size < 0):
          * Debit trade value (pay cash to buy back borrowed asset)
          * Debit commission

        P&L is implicitly captured by the difference between entry and exit cash flows.

        Args:
            current_price: Current market price
            current_step: Current bar/step number
            df: Optional dataframe for timestamp tracking
        """
        if self.position.size == 0:
            return

        # Apply slippage (opposite direction when closing)
        execution_price = current_price * (1 - self.slippage_rate * np.sign(self.position.size))

        # Calculate trade value and commission
        trade_value = abs(self.position.size * execution_price)
        commission = trade_value * self.commission_rate

        # Calculate P&L for logging (difference between entry and exit)
        pnl = self.position.size * (execution_price - self.position.entry_price)

        # Execute cash flow based on position type
        if self.position.size > 0:  # Closing LONG: Receive cash from selling
            credited = self.cash.credit(trade_value)
            debited = self.cash.debit(commission)
            cash_flow_description = f"+${credited:.2f}, -${debited:.2f}"
        else:  # Closing SHORT: Pay cash to buy back
            debited_total = self.cash.debit(trade_value + commission)
            cash_flow_description = f"-${debited_total:.2f}"

        # Log closing trade
        logger.debug(
            f"Position closed: P&L=${pnl:.2f}, Commission=${commission:.2f}, "
            f"Net=${pnl-commission:.2f} (cash flow: {cash_flow_description}, "
            f"balance: ${self.cash.balance:.2f})"
        )

        # Record completed trade in history
        if self._open_trade is not None:
            timestamp = None
            if df is not None and 'timestamp' in df.columns:
                timestamp = df.iloc[current_step].get('timestamp', None)

            trade_value_at_entry = abs(self._open_trade['position_size'] * self._open_trade['entry_price'])
            self.trade_history.append({
                **self._open_trade,
                'close_step': current_step,
                'close_timestamp': timestamp,
                'exit_price': execution_price,
                'pnl': pnl,
                'exit_commission': commission,
                'net_pnl': pnl - commission,
                'return_pct': (pnl - commission) / trade_value_at_entry if trade_value_at_entry > 0 else 0.0,
                'hold_bars': current_step - self._open_trade['open_step'],
            })
            self._open_trade = None

        # Reset position
        self.position = Position()

    def close_all_positions(
        self,
        current_price: float,
        current_step: int,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Close all open positions at current price.
        Should be called at episode end to realize all P&L.

        Args:
            current_price: Current market price
            current_step: Current bar/step number
            df: Optional dataframe for timestamp tracking
        """
        if self.position.size != 0:
            logger.debug(f"Closing position at episode end: size={self.position.size:.4f}, price={current_price:.2f}")
            self.close_position(current_price, current_step, df)

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """
        Get history of completed round-trip trades.

        Returns:
            List of trade dictionaries with entry/exit details.
            Each trade contains:
            - trade_id: Trade number
            - open_step, close_step: Step numbers when opened/closed
            - open_timestamp, close_timestamp: Timestamps (if available)
            - side: 'LONG' or 'SHORT'
            - entry_price, exit_price: Execution prices
            - position_size: Size of position
            - pnl: Raw profit/loss
            - entry_commission, exit_commission: Commission paid
            - net_pnl: P&L after commissions
            - return_pct: Return as percentage of trade value
            - hold_bars: Number of bars position was held
        """
        return self.trade_history
