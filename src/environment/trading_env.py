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
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Track current position state"""
    size: float = 0.0  # Position size (positive=long, negative=short, 0=flat)
    entry_price: float = 0.0
    entry_bar: int = 0
    current_bar: int = 0


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
        hold_closes_position: bool = False,
    ):
        super().__init__()

        # Store configuration
        self.df = df.copy()
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.reward_type = reward_type
        self.discrete_actions = discrete_actions
        self.max_position_pct = max_position_pct
        self.randomize_start = randomize_start
        self.min_episode_length = min_episode_length
        self.hold_closes_position = hold_closes_position

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
        # Features + [position, entry_price, position_pnl, cash_pct]
        obs_dim = self.n_features * self.lookback_window + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Episode tracking
        self.current_step = 0
        self.start_step = self.lookback_window
        self.max_steps = len(self.df) - 1

        # Account state
        self.balance = initial_balance
        self.position = Position()

        # History tracking
        self.history = {
            'balance': [],
            'returns': [],
            'positions': [],
            'actions': [],
            'rewards': [],
        }

        # Log initialization parameters
        logger.info(f"TradingEnv initialized: randomize_start={self.randomize_start}, "
                   f"hold_closes_position={self.hold_closes_position}, "
                   f"min_episode_length={self.min_episode_length}, "
                   f"reward_type={self.reward_type}, "
                   f"data_length={len(self.df)}")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset account state
        self.balance = self.initial_balance
        self.position = Position()

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

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""

        # Store previous balance for reward calculation
        prev_balance = self._get_portfolio_value()

        # Execute action
        self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Calculate reward
        current_balance = self._get_portfolio_value()
        reward = self._calculate_reward(prev_balance, current_balance)

        # Update history
        self.history['balance'].append(current_balance)
        self.history['returns'].append((current_balance - prev_balance) / prev_balance)
        self.history['positions'].append(self.position.size)
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
        end_idx = self.current_step

        feature_window = self.df.iloc[start_idx:end_idx][self.features].values
        feature_vector = feature_window.flatten()

        # Add position information
        current_price = self._get_current_price()
        position_pnl = 0.0
        if self.position.size != 0:
            position_pnl = (current_price - self.position.entry_price) / self.position.entry_price
            position_pnl *= np.sign(self.position.size)

        position_info = np.array([
            self.position.size,  # Current position
            self.position.entry_price / current_price if self.position.size != 0 else 0,
            position_pnl,  # Unrealized PnL
            self.balance / self.initial_balance,  # Cash percentage
        ])

        # Combine features and position info
        obs = np.concatenate([feature_vector, position_info]).astype(np.float32)

        # Handle NaN values (important for z-scores at the beginning)
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return obs

    def _execute_action(self, action: int):
        """Execute trading action"""
        current_price = self._get_current_price()

        if self.discrete_actions:
            # Discrete actions: 0=Hold, 1=Buy, 2=Sell
            if action == 0:  # Hold
                # Configurable behavior: close position or do nothing
                if self.hold_closes_position and self.position.size != 0:
                    logger.debug(f"Hold action closing position: size={self.position.size:.4f}")
                    self._close_position(current_price)
                return
            elif action == 1:  # Buy
                self._enter_position(1.0, current_price)
            elif action == 2:  # Sell
                self._enter_position(-1.0, current_price)
        else:
            # Continuous action: -1 to 1
            if abs(action) > 0.1:  # Threshold to avoid tiny trades
                self._enter_position(action, current_price)

    def _should_close_position(self, signal: float) -> bool:
        """
        Determine if current position should be closed.

        Args:
            signal: Trading signal (-1, 0, +1)

        Returns:
            True if position should be closed
        """
        # No position to close
        if self.position.size == 0:
            return False

        # Close if taking opposite direction
        if np.sign(signal) != np.sign(self.position.size):
            return True

        return False

    def _execute_open(self, signal: float, current_price: float):
        """
        Open a new position.

        Args:
            signal: Trading signal (-1 for short, +1 for long)
            current_price: Current market price
        """
        if signal == 0:
            return

        # Calculate position size
        max_position_value = self.balance * self.max_position_pct
        position_size = (max_position_value / current_price) * np.sign(signal)

        # Apply slippage
        execution_price = current_price * (1 + self.slippage_rate * np.sign(signal))

        # Calculate cost
        trade_value = abs(position_size * execution_price)
        commission = trade_value * self.commission_rate

        if self.balance >= trade_value + commission:
            # Update position
            self.position.size = position_size
            self.position.entry_price = execution_price
            self.position.entry_bar = self.current_step

            # Update balance
            self.balance -= commission

    def _enter_position(self, signal: float, current_price: float):
        """
        Enter or modify position based on signal.

        This is the main entry point for position management.
        Handles closing existing positions and opening new ones.

        Args:
            signal: Trading signal (-1 for short, 0 for flat, +1 for long)
            current_price: Current market price
        """
        current_pos = self.position.size

        # Check if we should close current position
        if self._should_close_position(signal):
            logger.debug(f"Closing position: current={current_pos:.4f}, signal={signal:.1f}")
            self._close_position(current_price)

        # Open new position if flat and signal is non-zero
        if self.position.size == 0 and signal != 0:
            logger.debug(f"Opening position: signal={signal:.1f}, price={current_price:.2f}")
            self._execute_open(signal, current_price)
            logger.debug(f"Position opened: size={self.position.size:.4f}")

    def _close_position(self, current_price: float):
        """Close current position"""
        if self.position.size == 0:
            return

        # Apply slippage (opposite direction)
        execution_price = current_price * (1 - self.slippage_rate * np.sign(self.position.size))

        # Calculate PnL
        pnl = self.position.size * (execution_price - self.position.entry_price)

        # Calculate commission
        trade_value = abs(self.position.size * execution_price)
        commission = trade_value * self.commission_rate

        # Update balance
        self.balance += pnl - commission

        # Reset position
        self.position = Position()

    def _get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + position value)"""
        if self.position.size == 0:
            return self.balance

        current_price = self._get_current_price()
        position_value = abs(self.position.size) * current_price
        unrealized_pnl = self.position.size * (current_price - self.position.entry_price)

        return self.balance + unrealized_pnl

    def _get_current_price(self) -> float:
        """Get current price from data"""
        return self.df.iloc[self.current_step]['close']

    def _calculate_reward(self, prev_value: float, current_value: float) -> float:
        """Calculate reward based on configured reward type"""
        returns = (current_value - prev_value) / prev_value

        if self.reward_type == "returns":
            reward = returns
        elif self.reward_type == "pnl":
            reward = current_value - prev_value
        elif self.reward_type == "sharpe":
            # Simple Sharpe approximation
            if len(self.history['returns']) > 1:
                returns_array = np.array(self.history['returns'][-20:])  # Last 20 returns
                reward = returns_array.mean() / (returns_array.std() + 1e-8)
            else:
                reward = returns
        else:
            reward = returns

        # Clip reward to prevent extreme values
        # Sharpe ratios typically range from -3 to 3, returns are small decimals
        return np.clip(reward, -10.0, 10.0)

    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if balance too low
        if self.balance < self.initial_balance * 0.2:  # Lost 80%
            return True

        # Check max drawdown
        if len(self.history['balance']) > 0:
            peak = max(self.history['balance'])
            drawdown = (peak - self._get_portfolio_value()) / peak
            if drawdown > 0.3:  # 30% drawdown
                return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        portfolio_value = self._get_portfolio_value()

        info = {
            'step': self.current_step,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'position': self.position.size,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
        }

        # Add performance metrics if we have history
        if len(self.history['returns']) > 1:
            returns = np.array(self.history['returns'])
            info['sharpe'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
            info['max_drawdown'] = self._calculate_max_drawdown()

        return info

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from balance history"""
        if len(self.history['balance']) == 0:
            return 0.0

        balance_array = np.array(self.history['balance'])
        cummax = np.maximum.accumulate(balance_array)
        drawdown = (cummax - balance_array) / cummax
        return drawdown.max()

    def render(self):
        """Render environment (for debugging)"""
        info = self._get_info()
        print(f"Step: {info['step']}, Balance: ${info['balance']:.2f}, "
              f"Position: {info['position']:.4f}, Return: {info['total_return']:.2%}")