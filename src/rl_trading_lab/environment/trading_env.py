"""
Trading Environment for RL agents.
Compatible with Stable-Baselines3 and Gymnasium.
"""

import logging
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
from enum import IntEnum

from .portfolio import Portfolio

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions for discrete action space"""
    HOLD = 0
    BUY = 1
    SELL = 2


class TradingEnv(gym.Env):
    """
    A trading environment for RL agents.

    Features:
    - Discrete or continuous action space
    - Configurable reward functions
    - Transaction costs and slippage
    - Position tracking
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        lookback_window: int = 20,
        initial_balance: float = 10000,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        reward_type: str = "sharpe",
        discrete_actions: bool = True,
        max_position_pct: float = 0.95,
        features_to_use: Optional[list] = None,
        randomize_start: bool = True,
        min_episode_length: int = 100,
        min_holding_period: int = 1,
        hold_closes_position: bool = False,
        price_column: str = "close",
        one_trade_mode: bool = False,
    ):
        super().__init__()

        # Store configuration
        self.df = df.copy()

        self.price_column = price_column
        self.lookback_window = lookback_window
        self.min_episode_length = min_episode_length
        self.min_holding_period = min_holding_period
        self.one_trade_mode = one_trade_mode
        self.position_closed_this_episode = False
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.reward_type = reward_type
        self.discrete_actions = discrete_actions
        self.max_position_pct = max_position_pct
        self.randomize_start = randomize_start
        self.hold_closes_position = hold_closes_position

        # Validate reward type
        valid_reward_types = ["returns", "pnl"]
        if self.reward_type not in valid_reward_types:
            if self.reward_type == "sharpe":
                raise ValueError(
                    f"reward_type='sharpe' is not supported. "
                    f"Sharpe ratio cannot be meaningfully calculated over short step windows. "
                    f"Use reward_type='returns' or 'pnl' instead. "
                    f"Episode-level Sharpe ratio is automatically calculated and available in the info dict."
                )
            else:
                raise ValueError(
                    f"Invalid reward_type='{self.reward_type}'. "
                    f"Valid options: {valid_reward_types}"
                )

        # Validate price column exists
        if self.price_column not in self.df.columns:
            raise ValueError(
                f"Price column '{self.price_column}' not found in data. "
                f"Available columns: {sorted(self.df.columns.tolist())}"
            )

        # Prepare features
        if features_to_use:
            # Use specified features
            self.features = features_to_use
        else:
            # Use all numeric columns except price columns
            self.features = [col for col in df.columns
                           if df[col].dtype in ['float64', 'int64']
                           and col not in ['timestamp', 'bar_id']]

        self.n_features = len(self.features)

        # Clean NaN values from data at initialization
        # TODO: This should be done BEFORE splitting data in train.py, not here.
        # Cleaning after split means train/val/test may have different amounts of usable data.
        # Only drop rows where features we're using OR price column have NaN
        initial_rows = len(self.df)
        required_columns = self.features + [self.price_column]
        self.df = self.df.dropna(subset=required_columns).reset_index(drop=True)
        rows_dropped = initial_rows - len(self.df)
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows with NaN in required columns at initialization")

        # Validate sufficient data after cleaning
        min_required_rows = self.lookback_window + self.min_episode_length + 1
        if len(self.df) < min_required_rows:
            raise ValueError(
                f"Insufficient data after cleaning NaN values. "
                f"Need at least {min_required_rows} rows "
                f"(lookback_window={self.lookback_window} + min_episode_length={self.min_episode_length} + 1), "
                f"but only have {len(self.df)} rows after dropping {rows_dropped} NaN rows. "
                f"Check your input data and feature engineering."
            )

        # Define action and observation spaces
        if discrete_actions:
            # 0: Hold, 1: Buy, 2: Sell
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous: -1 (full sell) to 1 (full buy)
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32
            )

        # Observation space: features + position info
        # Features + [position, position_pnl, cash_pct]
        obs_dim = self.n_features * (self.lookback_window + 1) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.start_step = self.lookback_window
        self.max_steps = len(self.df) - 1

        # Portfolio management
        self.portfolio = Portfolio(
            initial_cash=initial_balance,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_pct=max_position_pct,
            min_holding_period=min_holding_period,
        )

        # History tracking
        self.history = {
            'portfolio_value': [],
            'returns': [],
            'positions': [],
            'actions': [],
            'rewards': [],
        }

        # Log initialization parameters
        logger.info(f"TradingEnv initialized: randomize_start={self.randomize_start}, "
                   f"hold_closes_position={self.hold_closes_position}, "
                   f"one_trade_mode={self.one_trade_mode}, "
                   f"min_episode_length={self.min_episode_length}, "
                   f"reward_type={self.reward_type}, "
                   f"data_length={len(self.df)}")

    # --- Convenience properties for backward compatibility ---

    @property
    def num_trades(self) -> int:
        """Number of trades executed (convenience property)."""
        return self.portfolio.num_trades

    @property
    def balance(self) -> float:
        """Current cash balance (convenience property)."""
        return self.portfolio.cash.balance

    @property
    def position(self):
        """Current position (convenience property)."""
        return self.portfolio.position

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value (convenience method for tests)."""
        current_price = self._get_current_price()
        return self.portfolio.get_portfolio_value(current_price)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset portfolio
        self.portfolio.reset()

        # Reset step counter with optional randomization
        if self.randomize_start:
            # Randomize starting point to train on diverse market conditions
            max_start = self.max_steps - self.min_episode_length
            if max_start > self.start_step:
                self.current_step = self.np_random.integers(self.start_step, max_start)
                logger.debug(f"Episode reset: randomized start at step {self.current_step}/{self.max_steps}")
            else:
                self.current_step = self.start_step
                logger.debug(f"Episode reset: insufficient data for randomization, start at {self.current_step}")
        else:
            # Always start from beginning (original behavior)
            self.current_step = self.start_step
            logger.debug(f"Episode reset: fixed start at step {self.current_step}")

        # Clear history
        for key in self.history:
            self.history[key].clear()

        # Reset ONE_TRADE tracking
        self.position_closed_this_episode = False

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""

        # Store previous balance for reward calculation
        current_price = self._get_current_price()
        prev_balance = self.portfolio.get_portfolio_value(current_price)

        # Execute action
        # Note: MaskablePPO will prevent invalid actions from being sampled
        # If used without masking support, invalid actions will be silently ignored by portfolio
        self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Calculate reward
        current_price = self._get_current_price()
        current_balance = self.portfolio.get_portfolio_value(current_price)
        reward = self._calculate_reward(prev_balance, current_balance)

        # Update history
        self.history['portfolio_value'].append(current_balance)
        self.history['returns'].append((current_balance - prev_balance) / prev_balance)
        self.history['positions'].append(self.portfolio.position.size)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)

        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps - 1

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get lookback window of features
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step + 1

        feature_window = self.df.iloc[start_idx:end_idx][self.features].values
        feature_vector = feature_window.flatten()

        # Add position information
        current_price = self._get_current_price()
        position_pnl = 0.0
        if self.portfolio.position.size != 0:
            position_pnl = (current_price - self.portfolio.position.entry_price) / self.portfolio.position.entry_price
            position_pnl *= np.sign(self.portfolio.position.size)

        portfolio_value = self.portfolio.get_portfolio_value(current_price)

        # TODO: Investigate if keeping both position_pnl (current trade) and
        # portfolio_ratio (cumulative episode performance) provides redundant vs
        # complementary signals. Position_pnl is localized to current position,
        # portfolio_ratio reflects total account performance including all past trades.
        position_info = np.array([
            self.portfolio.position.size,  # Current position
            position_pnl,  # Unrealized PnL (percentage)
            portfolio_value / self.initial_balance,  # Portfolio percentage (total account performance)
        ])

        # Combine features and position info
        obs = np.concatenate([feature_vector, position_info]).astype(np.float32)

        return obs

    def _get_current_timestamp(self):
        """Get current timestamp from data if available."""
        if 'timestamp' in self.df.columns:
            return self.df.iloc[self.current_step].get('timestamp', None)
        return None

    def _execute_action(self, action: Action):
        """Execute trading action"""
        current_price = self._get_current_price()
        current_timestamp = self._get_current_timestamp()

        # Track position state before action for ONE_TRADE mode
        had_position = self.portfolio.position.size != 0

        if self.discrete_actions:
            # Use Action enum for clarity
            if action == Action.HOLD:
                # Configurable behavior: close position or do nothing
                if self.hold_closes_position and self.portfolio.position.size != 0:
                    logger.debug(f"Hold action closing position: size={self.portfolio.position.size:.4f}")
                    self.portfolio.close_position(current_price, self.current_step, current_timestamp)
                    # Check if position was closed
                    if had_position and self.portfolio.position.size == 0:
                        self.position_closed_this_episode = True
                return
            elif action == Action.BUY:
                self.portfolio.execute_trade(1.0, current_price, self.current_step, current_timestamp)
            elif action == Action.SELL:
                self.portfolio.execute_trade(-1.0, current_price, self.current_step, current_timestamp)
        else:
            # Continuous action: -1 to 1
            if abs(action) > 0.1:  # Threshold to avoid tiny trades
                self.portfolio.execute_trade(action, current_price, self.current_step, current_timestamp)

        # Check if position was closed during execute_trade (for BUY/SELL actions)
        if had_position and self.portfolio.position.size == 0:
            self.position_closed_this_episode = True

    def close_all_positions(self):
        """
        Close all open positions at current price.
        Should be called at episode end to realize all P&L.
        """
        if self.portfolio.position.size != 0:
            current_price = self._get_current_price()
            current_timestamp = self._get_current_timestamp()
            self.portfolio.close_all_positions(current_price, self.current_step, current_timestamp)

    def _get_current_price(self) -> float:
        """Get current price from data using configured price column"""
        return self.df.iloc[self.current_step][self.price_column]

    def _calculate_reward(self, prev_value: float, current_value: float) -> float:
        """
        Calculate reward based on configured reward type.

        NOTE: Sharpe ratio is NOT a valid reward type because:
        - Requires many samples for meaningful variance estimation
        - Short windows (e.g., 20 steps) create non-stationary signals
        - Agent would optimize short-term Sharpe, not episode Sharpe
        - Episode-level Sharpe is available in info dict instead

        Valid reward types:
        - "returns": Percentage return per step (recommended)
        - "pnl": Absolute dollar P&L per step
        """
        returns = (current_value - prev_value) / prev_value

        if self.reward_type == "returns":
            reward = returns
        elif self.reward_type == "pnl":
            reward = current_value - prev_value
        else:
            # Default to returns for unknown types
            reward = returns

        # Clip reward to prevent extreme values
        return np.clip(reward, -10.0, 10.0)

    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # ONE_TRADE mode: Terminate after first position close
        if self.one_trade_mode and self.position_closed_this_episode:
            return True

        current_price = self._get_current_price()
        portfolio_value = self.portfolio.get_portfolio_value(current_price)

        # Terminate if portfolio value too low
        if portfolio_value < self.initial_balance * 0.2:  # Lost 80%
            return True

        # Check max drawdown
        if len(self.history['portfolio_value']) > 0:
            peak = max(self.history['portfolio_value'])
            drawdown = (peak - portfolio_value) / peak
            if drawdown > 0.3:  # 30% drawdown
                return True

        return False

    def action_masks(self) -> np.ndarray:
        """
        Get boolean mask of valid actions for current state.

        Required by MaskablePPO from sb3-contrib. Returns array where:
        - True = action is valid
        - False = action is invalid

        Returns:
            Boolean array of shape (n_actions,) indicating valid actions
        """
        position_sign = np.sign(self.portfolio.position.size)

        if position_sign == 0:  # Flat (no position)
            # All actions valid: HOLD, BUY, SELL
            return np.array([True, True, True], dtype=bool)
        elif position_sign > 0:  # LONG position
            # Can HOLD or SELL, cannot BUY
            return np.array([True, False, True], dtype=bool)
        else:  # SHORT position (position_sign < 0)
            # Can HOLD or BUY, cannot SELL
            return np.array([True, True, False], dtype=bool)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        current_price = self._get_current_price()
        portfolio_value = self.portfolio.get_portfolio_value(current_price)

        # Get action mask for info dict (for compatibility/debugging)
        action_mask_bool = self.action_masks()
        action_mask = action_mask_bool.astype(np.int8)

        info = {
            'step': self.current_step,
            'cash': self.portfolio.cash.balance,  # Available cash (buying power)
            'portfolio_value': portfolio_value,  # Total account value (cash + unrealized P&L)
            'position': self.portfolio.position.size,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'num_trades': self.portfolio.num_trades,
            'action_mask': action_mask,  # For compatibility/debugging
        }

        # Add performance metrics if we have history
        if len(self.history['returns']) > 1:
            returns = np.array(self.history['returns'])
            info['sharpe'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
            info['max_drawdown'] = self._calculate_max_drawdown()

        return info

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio value history"""
        if len(self.history['portfolio_value']) == 0:
            return 0.0

        portfolio_array = np.array(self.history['portfolio_value'])
        cummax = np.maximum.accumulate(portfolio_array)
        drawdown = (cummax - portfolio_array) / cummax
        return drawdown.max()

    def get_trade_history(self) -> list:
        """
        Get history of completed round-trip trades.

        Returns:
            List of trade dictionaries with entry/exit details.
            Each trade contains:
            - trade_id: Trade number
            - entry_bar, exit_bar: Bar indices when opened/closed
            - entry_timestamp, exit_timestamp: Timestamps (if available)
            - side: 'LONG' or 'SHORT'
            - entry_price, exit_price: Execution prices
            - position_size: Size of position
            - gross_pnl: P&L before commissions
            - entry_commission, exit_commission: Commission paid
            - net_pnl: P&L after commissions
            - return_pct: Return as percentage of trade value
            - hold_bars: Number of bars position was held
        """
        return self.portfolio.get_trade_history_dicts()

    def render(self):
        """Render environment (for debugging)"""
        info = self._get_info()
        print(f"Step: {info['step']}, Portfolio: ${info['portfolio_value']:.2f}, "
              f"Position: {info['position']:.4f}, Return: {info['total_return']:.2%}")