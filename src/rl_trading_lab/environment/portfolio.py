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
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from rl_trading_lab.domain.exceptions import InsufficientFundsError
from rl_trading_lab.domain.value_objects.position import Position
from rl_trading_lab.domain.value_objects.trade import CompletedTrade, TradeSide

logger = logging.getLogger(__name__)


class Cash:
    """
    Manages cash balance with explicit debit/credit operations.

    This class encapsulates cash management using standard accounting terminology:
    - debit(): Remove cash (paying for something)
    - credit(): Add cash (receiving payment)

    Enforces the invariant: cash balance cannot go negative.
    """

    def __init__(self, initial_balance: float):
        """
        Initialize cash account.

        Args:
            initial_balance: Starting cash amount

        Raises:
            ValueError: If initial_balance is negative
        """
        if initial_balance < 0:
            raise ValueError(f"Initial balance cannot be negative: {initial_balance}")
        self._balance = initial_balance
        self.initial_balance = initial_balance

    @property
    def balance(self) -> float:
        """Get current cash balance."""
        return self._balance

    def debit(self, amount: float, strict: bool = True) -> float:
        """
        Remove cash from account (payment/cost).

        Args:
            amount: Amount to deduct (sign is ignored, always debits)
            strict: If True, raise InsufficientFundsError when balance too low.
                   If False, allow negative balance (legacy behavior, deprecated).

        Returns:
            Absolute amount debited

        Raises:
            InsufficientFundsError: If strict=True and insufficient funds
        """
        amount_abs = abs(amount)

        if strict and amount_abs > self._balance:
            raise InsufficientFundsError(
                requested=amount_abs,
                available=self._balance,
            )

        if not strict and amount_abs > self._balance:
            warnings.warn(
                f"Debiting ${amount_abs:.2f} from balance ${self._balance:.2f} "
                "will result in negative balance. This behavior is deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._balance -= amount_abs
        return amount_abs

    def try_debit(self, amount: float) -> tuple:
        """
        Attempt to debit, returning success status.

        Use this instead of has_funds() + debit() for atomicity.

        Args:
            amount: Amount to debit (absolute value used)

        Returns:
            Tuple of (success: bool, amount_debited: float)
        """
        amount_abs = abs(amount)
        if amount_abs > self._balance:
            return (False, 0.0)
        self._balance -= amount_abs
        return (True, amount_abs)

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
        """Reset cash to initial balance."""
        self._balance = self.initial_balance


class Portfolio:
    """
    Manages portfolio accounting including cash, positions, and trade execution.

    Implements SPOT TRADING model (see module docstring for details).

    Uses immutable Position value objects - state changes create new Position instances.

    Attributes:
        cash: Cash instance managing available balance
        position: Current position state (immutable Position value object)
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
        self._position = Position.flat()  # Immutable position

        # Trade tracking
        self.num_trades = 0
        self._trade_history: List[CompletedTrade] = []
        self._open_trade_data: Optional[Dict[str, Any]] = None

    @property
    def position(self) -> Position:
        """Current position (immutable)."""
        return self._position

    def reset(self):
        """Reset portfolio to initial state."""
        self.cash.reset()
        self._position = Position.flat()
        self.num_trades = 0
        self._trade_history.clear()
        self._open_trade_data = None

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
        if self._position.is_flat:
            return self.cash.balance

        # Spot trading: value = cash + position value
        position_value = self._position.market_value(current_price)
        return self.cash.balance + position_value

    def get_position_pnl(self, current_price: float) -> float:
        """
        Get unrealized P&L for current position.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L (0 if no position)
        """
        return self._position.unrealized_pnl(current_price)

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
        if self._position.is_flat:
            return False

        # Don't close if minimum holding period not met
        bars_held = self._position.holding_period(current_step)
        if bars_held < self.min_holding_period:
            logger.debug(
                f"Position held for {bars_held} bars, need {self.min_holding_period} bars minimum"
            )
            return False

        # Close if taking opposite direction
        if np.sign(signal) != self._position.direction:
            return True

        return False

    def execute_trade(
        self,
        signal: float,
        current_price: float,
        current_step: int,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Execute a trade based on signal.

        Handles closing existing positions and opening new ones.
        On reversal (opposite signal), closes current position AND opens new one.

        Args:
            signal: Trading signal (-1 for short, 0 for flat, +1 for long)
            current_price: Current market price
            current_step: Current bar/step number
            timestamp: Optional timestamp for trade records

        Returns:
            True if a trade was executed
        """
        trade_executed = False

        # Check if we should close current position
        if self.should_close_position(signal, current_step):
            logger.debug(
                f"Closing position: current={self._position.size:.4f}, signal={signal:.1f}"
            )
            self.close_position(current_price, current_step, timestamp)
            trade_executed = True

            # After closing, if signal is non-zero, open new position (reversal)
            if signal != 0:
                logger.debug(f"Reversal: opening new position with signal={signal:.1f}")
                self._open_position(signal, current_price, current_step, timestamp)
                logger.debug(f"Position opened: size={self._position.size:.4f}")

            return True

        # Open new position if flat and signal is non-zero
        if self._position.is_flat and signal != 0:
            logger.debug(f"Opening position: signal={signal:.1f}, price={current_price:.2f}")
            self._open_position(signal, current_price, current_step, timestamp)
            logger.debug(f"Position opened: size={self._position.size:.4f}")
            return True

        return False

    def _open_position(
        self,
        signal: float,
        current_price: float,
        current_step: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Open a new position using spot trading model with explicit cash operations.

        Creates a new immutable Position object.

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
            timestamp: Optional timestamp for trade record
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
            logger.debug(
                f"Insufficient funds: need ${trade_value + commission:.2f}, "
                f"have ${self.cash.balance:.2f}"
            )
            return

        # Create new immutable position
        if signal > 0:
            self._position = Position.open_long(
                size=abs(position_size),
                price=execution_price,
                bar=current_step,
            )
        else:
            self._position = Position.open_short(
                size=abs(position_size),
                price=execution_price,
                bar=current_step,
            )

        # Execute cash flow based on position type
        if signal > 0:  # LONG: Pay cash to buy
            debited = self.cash.debit(trade_value + commission, strict=False)
            cash_flow_description = f"-${debited:.2f}"
        else:  # SHORT: Receive cash from selling, pay commission
            credited = self.cash.credit(trade_value)
            debited = self.cash.debit(commission, strict=False)
            cash_flow_description = f"+${credited:.2f}, -${debited:.2f}"

        # Increment trade counter
        self.num_trades += 1
        logger.debug(
            f"Trade #{self.num_trades}: {'LONG' if signal > 0 else 'SHORT'} "
            f"{abs(position_size):.4f} @ ${execution_price:.2f} "
            f"(cash flow: {cash_flow_description}, remaining: ${self.cash.balance:.2f})"
        )

        # Track open trade for history
        self._open_trade_data = {
            "trade_id": self.num_trades,
            "open_step": current_step,
            "open_timestamp": timestamp,
            "side": TradeSide.LONG if signal > 0 else TradeSide.SHORT,
            "entry_price": execution_price,
            "position_size": abs(position_size),
            "entry_commission": commission,
        }

    def close_position(
        self,
        current_price: float,
        current_step: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Close current position using spot trading model with explicit cash operations.

        Creates a CompletedTrade value object and resets position to flat.

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
            timestamp: Optional timestamp for trade record
        """
        if self._position.is_flat:
            return

        # Apply slippage (opposite direction when closing)
        execution_price = current_price * (
            1 - self.slippage_rate * self._position.direction
        )

        # Calculate trade value and commission
        trade_value = abs(self._position.size * execution_price)
        commission = trade_value * self.commission_rate

        # Calculate P&L for logging
        pnl = self._position.unrealized_pnl(execution_price)

        # Execute cash flow based on position type
        if self._position.is_long:  # Closing LONG: Receive cash from selling
            credited = self.cash.credit(trade_value)
            debited = self.cash.debit(commission, strict=False)
            cash_flow_description = f"+${credited:.2f}, -${debited:.2f}"
        else:  # Closing SHORT: Pay cash to buy back
            debited_total = self.cash.debit(trade_value + commission, strict=False)
            cash_flow_description = f"-${debited_total:.2f}"

        # Log closing trade
        logger.debug(
            f"Position closed: P&L=${pnl:.2f}, Commission=${commission:.2f}, "
            f"Net=${pnl - commission:.2f} (cash flow: {cash_flow_description}, "
            f"balance: ${self.cash.balance:.2f})"
        )

        # Record completed trade in history
        if self._open_trade_data is not None:
            completed_trade = CompletedTrade(
                trade_id=self._open_trade_data["trade_id"],
                side=self._open_trade_data["side"],
                entry_price=self._open_trade_data["entry_price"],
                exit_price=execution_price,
                position_size=self._open_trade_data["position_size"],
                entry_bar=self._open_trade_data["open_step"],
                exit_bar=current_step,
                entry_commission=self._open_trade_data["entry_commission"],
                exit_commission=commission,
                entry_timestamp=self._open_trade_data["open_timestamp"],
                exit_timestamp=timestamp,
            )
            self._trade_history.append(completed_trade)
            self._open_trade_data = None

        # Reset position to flat (new immutable instance)
        self._position = Position.flat()

    def close_all_positions(
        self,
        current_price: float,
        current_step: int,
        timestamp: Optional[datetime] = None,
    ):
        """
        Close all open positions at current price.
        Should be called at episode end to realize all P&L.

        Args:
            current_price: Current market price
            current_step: Current bar/step number
            timestamp: Optional timestamp for trade record
        """
        if not self._position.is_flat:
            logger.debug(
                f"Closing position at episode end: size={self._position.size:.4f}, "
                f"price={current_price:.2f}"
            )
            self.close_position(current_price, current_step, timestamp)

    def get_trade_history(self) -> List[CompletedTrade]:
        """
        Get history of completed round-trip trades.

        Returns:
            List of CompletedTrade value objects
        """
        return self._trade_history.copy()

    def get_trade_history_dicts(self) -> List[Dict[str, Any]]:
        """
        Get history of completed trades as dictionaries.

        For backward compatibility with code expecting dict format.

        Returns:
            List of trade dictionaries with entry/exit details.
        """
        return [trade.to_dict() for trade in self._trade_history]

    @property
    def trade_history(self) -> List[Dict[str, Any]]:
        """
        Legacy property for backward compatibility.

        Returns trade history as list of dicts.
        Prefer get_trade_history() for new code.
        """
        return self.get_trade_history_dicts()
