"""
TradingDomain - Pure domain logic for trading simulation.

This class implements the core trading domain without any external
framework dependencies (no Gymnasium, no pandas, no numpy).
All data access goes through the MarketDataPort interface.

Per Evans (DDD), this is the heart of the domain model where
business logic resides, isolated from infrastructure concerns.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Any, Dict

from rl_trading_lab.domain.ports.market_data import MarketDataPort
from rl_trading_lab.domain.value_objects.position import Position
from rl_trading_lab.domain.value_objects.trade import CompletedTrade, TradeSide
from rl_trading_lab.domain.services.position_sizing import (
    PositionSizingService,
    FixedPercentagePositionSizing,
)
from rl_trading_lab.domain.services.reward_calculation import (
    RewardCalculationService,
    ReturnsRewardCalculation,
)
from rl_trading_lab.domain.services.risk_management import (
    RiskManagementService,
    RiskCheckResult,
    StandardRiskManagement,
)


class OrderIntent(IntEnum):
    """
    Trading order intents.

    These represent the agent's desired action, not the actual
    trade execution (which depends on current position state).
    """
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass(frozen=True)
class TradingState:
    """
    Immutable snapshot of the trading domain state.

    This value object represents the current state at any point in time.
    """
    current_bar: int
    position: Position
    cash_balance: float
    portfolio_value: float
    is_terminated: bool = False
    termination_reason: Optional[str] = None


@dataclass
class StepResult:
    """
    Result of processing an order intent.

    Contains the new state, reward, and any trade that was executed.
    """
    new_state: TradingState
    reward: float
    trade_executed: Optional[CompletedTrade] = None


@dataclass
class TradingDomainConfig:
    """Configuration for the TradingDomain."""

    initial_balance: float = 10000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_pct: float = 0.95
    lookback_window: int = 20
    min_episode_length: int = 100
    min_holding_period: int = 1


class TradingDomain:
    """
    Pure domain logic for trading simulation.

    No external framework dependencies (no gym, no pandas, no numpy).
    All data access goes through MarketDataPort.

    This class orchestrates:
    - Order processing and trade execution
    - Position management
    - Portfolio value calculation
    - Risk management and termination conditions
    - Observation construction

    Example:
        >>> from rl_trading_lab.domain.trading_domain import TradingDomain, OrderIntent
        >>> domain = TradingDomain(
        ...     market_data=my_market_adapter,
        ...     observation_features=['close', 'volume'],
        ... )
        >>> state = domain.reset()
        >>> obs = domain.get_observation()
        >>> result = domain.process_order(OrderIntent.BUY)
        >>> print(f"Reward: {result.reward}, Position: {result.new_state.position.size}")
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        observation_features: List[str],
        config: Optional[TradingDomainConfig] = None,
        position_sizing: Optional[PositionSizingService] = None,
        reward_calculator: Optional[RewardCalculationService] = None,
        risk_manager: Optional[RiskManagementService] = None,
    ):
        """
        Initialize the trading domain.

        Args:
            market_data: Port for accessing market data
            observation_features: Feature columns to include in observations
            config: Domain configuration (uses defaults if not provided)
            position_sizing: Service for calculating position sizes
            reward_calculator: Service for calculating step rewards
            risk_manager: Service for checking termination conditions
        """
        self._market_data = market_data
        self._observation_features = observation_features
        self._config = config or TradingDomainConfig()

        # Domain services (use defaults if not provided)
        self._position_sizing = position_sizing or FixedPercentagePositionSizing()
        self._reward_calculator = reward_calculator or ReturnsRewardCalculation()
        self._risk_manager = risk_manager or StandardRiskManagement()

        # Internal state
        self._cash_balance: float = 0.0
        self._position: Position = Position.flat()
        self._current_bar: int = 0
        self._trade_history: List[CompletedTrade] = []
        self._open_trade_data: Optional[Dict[str, Any]] = None
        self._num_trades: int = 0
        self._is_terminated: bool = False
        self._termination_reason: Optional[str] = None

        # Cache for observation construction
        self._history: List[float] = []  # Portfolio value history

    # --- Configuration Properties ---

    @property
    def lookback_window(self) -> int:
        """Number of historical bars used in observations."""
        return self._config.lookback_window

    @property
    def min_episode_length(self) -> int:
        """Minimum number of steps per episode."""
        return self._config.min_episode_length

    @property
    def observation_dimension(self) -> int:
        """
        Total dimension of observation vector.

        = (lookback_window + 1) * num_features + 3 position info values
        """
        n_features = len(self._observation_features)
        return n_features * (self._config.lookback_window + 1) + 3

    @property
    def current_bar(self) -> int:
        """Current bar index."""
        return self._current_bar

    @property
    def position(self) -> Position:
        """Current position."""
        return self._position

    @property
    def cash_balance(self) -> float:
        """Current cash balance."""
        return self._cash_balance

    @property
    def num_trades(self) -> int:
        """Number of trades executed."""
        return self._num_trades

    @property
    def trade_history(self) -> List[CompletedTrade]:
        """List of completed trades."""
        return self._trade_history.copy()

    def __len__(self) -> int:
        """Number of bars in market data."""
        return len(self._market_data)

    # --- Core Operations ---

    def reset(self, start_bar: Optional[int] = None) -> TradingState:
        """
        Reset domain to initial state.

        Args:
            start_bar: Optional starting bar index (defaults to lookback_window)

        Returns:
            Initial trading state
        """
        # Reset state
        self._cash_balance = self._config.initial_balance
        self._position = Position.flat()
        self._current_bar = start_bar if start_bar is not None else self._config.lookback_window
        self._trade_history = []
        self._open_trade_data = None
        self._num_trades = 0
        self._history = []
        self._is_terminated = False
        self._termination_reason = None

        # Reset risk manager state
        self._risk_manager.reset()

        # Update risk manager peak tracking
        initial_value = self._get_portfolio_value()
        self._risk_manager.update_peak(initial_value)

        return self._get_state()

    def process_order(self, intent: OrderIntent) -> StepResult:
        """
        Process an order intent and advance one bar.

        Args:
            intent: The desired trading action

        Returns:
            StepResult containing new state, reward, and any executed trade
        """
        # Get portfolio value before action
        prev_value = self._get_portfolio_value()

        # Execute order based on intent
        trade = self._execute_intent(intent)
        if trade:
            self._trade_history.append(trade)

        # Advance to next bar
        self._current_bar += 1

        # Calculate new portfolio value
        current_value = self._get_portfolio_value()

        # Track history for metrics
        self._history.append(current_value)

        # Calculate reward
        reward = self._reward_calculator.calculate(
            prev_value=prev_value,
            current_value=current_value,
        )

        # Check termination conditions
        risk_result = self._risk_manager.check_termination(
            portfolio_value=current_value,
            initial_balance=self._config.initial_balance,
            current_bar=self._current_bar,
            trade_history=self._trade_history,
        )

        if risk_result.should_terminate:
            self._is_terminated = True
            self._termination_reason = risk_result.reason

        return StepResult(
            new_state=self._get_state(),
            reward=reward,
            trade_executed=trade,
        )

    def get_observation(self) -> Tuple[float, ...]:
        """
        Get current observation as flat tuple.

        The observation consists of:
        1. Feature window: (lookback_window + 1) bars of feature values
        2. Position info: current size, unrealized PnL ratio, portfolio ratio

        Returns:
            Tuple of observation values (can be converted to numpy array)
        """
        # Get feature window
        window = self._market_data.get_feature_window(
            start_index=self._current_bar - self._config.lookback_window,
            end_index=self._current_bar + 1,
            features=self._observation_features,
        )

        # Position information
        current_price = self._get_current_price()

        # Calculate position P&L ratio
        position_pnl_ratio = 0.0
        if not self._position.is_flat:
            pnl = self._position.unrealized_pnl(current_price)
            entry_value = abs(self._position.size * self._position.entry_price)
            if entry_value > 0:
                position_pnl_ratio = pnl / entry_value

        # Portfolio ratio (current value vs initial)
        portfolio_ratio = self._get_portfolio_value() / self._config.initial_balance

        # Combine into flat tuple
        # window.flatten() returns tuple, then add position info
        return window.flatten() + (
            self._position.size,
            position_pnl_ratio,
            portfolio_ratio,
        )

    def get_valid_actions(self) -> Tuple[bool, bool, bool]:
        """
        Get mask of valid actions.

        Returns:
            Tuple of (HOLD_valid, BUY_valid, SELL_valid)
        """
        if self._position.is_flat:
            # No position: can HOLD, BUY, or SELL
            return (True, True, True)
        elif self._position.is_long:
            # Long position: can HOLD or SELL (close), cannot BUY more
            return (True, False, True)
        else:
            # Short position: can HOLD or BUY (close), cannot SELL more
            return (True, True, False)

    def close_all_positions(self) -> Optional[CompletedTrade]:
        """
        Close all open positions at current price.

        Should be called at episode end to realize all P&L.

        Returns:
            CompletedTrade if a position was closed, None otherwise
        """
        if self._position.is_flat:
            return None

        current_price = self._get_current_price()
        return self._close_position(current_price)

    # --- Private Methods ---

    def _get_state(self) -> TradingState:
        """Get current state as immutable TradingState."""
        return TradingState(
            current_bar=self._current_bar,
            position=self._position,
            cash_balance=self._cash_balance,
            portfolio_value=self._get_portfolio_value(),
            is_terminated=self._is_terminated,
            termination_reason=self._termination_reason,
        )

    def _get_current_price(self, column: str = "close") -> float:
        """Get current price from market data."""
        return self._market_data.get_price(self._current_bar, column)

    def _get_portfolio_value(self) -> float:
        """
        Calculate current portfolio value.

        Formula: cash + position_market_value
        """
        if self._position.is_flat:
            return self._cash_balance

        current_price = self._get_current_price()
        position_value = self._position.market_value(current_price)
        return self._cash_balance + position_value

    def _execute_intent(self, intent: OrderIntent) -> Optional[CompletedTrade]:
        """
        Execute trading intent.

        Args:
            intent: The desired action

        Returns:
            CompletedTrade if a trade was completed, None otherwise
        """
        current_price = self._get_current_price()

        if intent == OrderIntent.HOLD:
            return None

        signal = 1.0 if intent == OrderIntent.BUY else -1.0

        # Check if we should close current position
        if self._should_close_position(signal):
            # Close and potentially reverse
            trade = self._close_position(current_price)

            # If signal is opposite direction and we just closed, open new position
            if signal != 0:
                self._open_position(signal, current_price)

            return trade

        # Open new position if flat
        if self._position.is_flat:
            self._open_position(signal, current_price)

        return None

    def _should_close_position(self, signal: float) -> bool:
        """
        Determine if current position should be closed.

        Args:
            signal: Trading signal (-1, 0, +1)

        Returns:
            True if position should be closed
        """
        if self._position.is_flat:
            return False

        # Check minimum holding period
        bars_held = self._position.holding_period(self._current_bar)
        if bars_held < self._config.min_holding_period:
            return False

        # Close if taking opposite direction
        position_sign = 1 if self._position.is_long else -1
        if signal != 0 and (signal > 0) != (position_sign > 0):
            return True

        return False

    def _open_position(self, signal: float, current_price: float) -> None:
        """
        Open a new position.

        Args:
            signal: Trading signal (-1 for short, +1 for long)
            current_price: Current market price
        """
        if signal == 0:
            return

        # Calculate position size
        position_size = self._position_sizing.calculate_size(
            available_cash=self._cash_balance,
            current_price=current_price,
            signal_strength=signal,
            max_position_pct=self._config.max_position_pct,
        )

        if position_size == 0:
            return

        # Apply slippage
        slippage_factor = 1 + self._config.slippage_rate * (1 if signal > 0 else -1)
        execution_price = current_price * slippage_factor

        # Calculate trade value and commission
        trade_value = abs(position_size * execution_price)
        commission = trade_value * self._config.commission_rate

        # Check sufficient funds
        if signal > 0:  # LONG: need to pay for position
            required = trade_value + commission
            if required > self._cash_balance:
                return
            self._cash_balance -= required
        else:  # SHORT: receive cash but pay commission
            self._cash_balance += trade_value
            self._cash_balance -= commission

        # Create new position
        if signal > 0:
            self._position = Position.open_long(
                size=abs(position_size),
                price=execution_price,
                bar=self._current_bar,
            )
        else:
            self._position = Position.open_short(
                size=abs(position_size),
                price=execution_price,
                bar=self._current_bar,
            )

        self._num_trades += 1

        # Track open trade data
        self._open_trade_data = {
            "trade_id": self._num_trades,
            "open_step": self._current_bar,
            "side": TradeSide.LONG if signal > 0 else TradeSide.SHORT,
            "entry_price": execution_price,
            "position_size": abs(position_size),
            "entry_commission": commission,
        }

    def _close_position(self, current_price: float) -> Optional[CompletedTrade]:
        """
        Close current position.

        Args:
            current_price: Current market price

        Returns:
            CompletedTrade record of the closed position
        """
        if self._position.is_flat:
            return None

        # Apply slippage (opposite direction when closing)
        direction = self._position.direction
        slippage_factor = 1 - self._config.slippage_rate * direction
        execution_price = current_price * slippage_factor

        # Calculate trade value and commission
        trade_value = abs(self._position.size * execution_price)
        commission = trade_value * self._config.commission_rate

        # Execute cash flow
        if self._position.is_long:  # Closing LONG: receive cash
            self._cash_balance += trade_value
            self._cash_balance -= commission
        else:  # Closing SHORT: pay to buy back
            self._cash_balance -= (trade_value + commission)

        # Create completed trade record
        completed_trade = None
        if self._open_trade_data is not None:
            completed_trade = CompletedTrade(
                trade_id=self._open_trade_data["trade_id"],
                side=self._open_trade_data["side"],
                entry_price=self._open_trade_data["entry_price"],
                exit_price=execution_price,
                position_size=self._open_trade_data["position_size"],
                entry_bar=self._open_trade_data["open_step"],
                exit_bar=self._current_bar,
                entry_commission=self._open_trade_data["entry_commission"],
                exit_commission=commission,
            )
            self._open_trade_data = None

        # Reset position
        self._position = Position.flat()

        return completed_trade

    # --- Metrics ---

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Get metrics for the current episode.

        Returns:
            Dictionary of metric name to value
        """
        current_value = self._get_portfolio_value()
        initial = self._config.initial_balance

        metrics = {
            "total_return": (current_value - initial) / initial,
            "num_trades": self._num_trades,
            "final_value": current_value,
        }

        # Calculate metrics from history
        if len(self._history) > 1:
            # Calculate returns
            returns = []
            for i in range(1, len(self._history)):
                ret = (self._history[i] - self._history[i-1]) / self._history[i-1]
                returns.append(ret)

            # Mean and std
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_return = variance ** 0.5

            # Sharpe ratio (annualized assuming daily bars)
            if std_return > 1e-8:
                metrics["sharpe"] = mean_return / std_return * (252 ** 0.5)
            else:
                metrics["sharpe"] = 0.0

            # Max drawdown
            peak = self._history[0]
            max_dd = 0.0
            for value in self._history:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            metrics["max_drawdown"] = max_dd

        # Trade-level metrics
        if self._trade_history:
            winning_trades = [t for t in self._trade_history if t.net_pnl > 0]
            metrics["win_rate"] = len(winning_trades) / len(self._trade_history)
            metrics["total_pnl"] = sum(t.net_pnl for t in self._trade_history)

        return metrics
