"""
GymTradingEnvAdapter - Anti-Corruption Layer for Gymnasium interface.

This adapter translates between the pure TradingDomain and the Gymnasium
environment interface required by RL frameworks like Stable-Baselines3.

Per Martin (Clean Architecture) and Evans (DDD), Anti-Corruption Layers
isolate domain logic from external system interfaces, preventing foreign
concepts from leaking into the domain model.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_trading_lab.domain.trading_domain import (
    OrderIntent,
    TradingDomain,
    TradingDomainConfig,
    TradingState,
)
from rl_trading_lab.domain.ports.market_data import MarketDataPort

logger = logging.getLogger(__name__)


class GymTradingEnvAdapter(gym.Env):
    """
    Anti-Corruption Layer adapting TradingDomain to Gymnasium interface.

    This adapter:
    - Translates gym.Env interface to domain operations
    - Converts numpy arrays to/from domain types
    - Handles gym-specific metadata and rendering
    - Provides action masking for MaskablePPO

    The adapter keeps external framework concerns (numpy arrays, gym spaces)
    separate from the pure domain logic.

    Example:
        >>> from rl_trading_lab.infrastructure.adapters.gym_adapter import GymTradingEnvAdapter
        >>> from rl_trading_lab.infrastructure.adapters.market_data_adapter import ParquetMarketDataAdapter
        >>>
        >>> # Setup domain
        >>> market_data = ParquetMarketDataAdapter(df)
        >>> domain = TradingDomain(market_data, observation_features=['close', 'volume'])
        >>>
        >>> # Wrap in gym adapter
        >>> env = GymTradingEnvAdapter(domain)
        >>>
        >>> # Use with SB3
        >>> from stable_baselines3 import PPO
        >>> model = PPO("MlpPolicy", env)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        domain: TradingDomain,
        randomize_start: bool = True,
        min_episode_length: Optional[int] = None,
    ):
        """
        Initialize the Gymnasium adapter.

        Args:
            domain: The TradingDomain instance to adapt
            randomize_start: Whether to randomize episode starting point
            min_episode_length: Minimum episode length (defaults to domain config)
        """
        super().__init__()

        self._domain = domain
        self._randomize_start = randomize_start
        self._min_episode_length = min_episode_length or domain.min_episode_length

        # Define action space: discrete with 3 actions (HOLD, BUY, SELL)
        self.action_space = spaces.Discrete(3)

        # Define observation space
        obs_dim = domain.observation_dimension
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Track episode stats for info dict
        self._episode_step = 0
        self._start_bar = 0

        logger.debug(
            f"GymTradingEnvAdapter initialized: "
            f"obs_dim={obs_dim}, randomize_start={randomize_start}"
        )

    @property
    def domain(self) -> TradingDomain:
        """Access to the underlying domain."""
        return self._domain

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        # Determine starting bar
        start_bar = None
        if self._randomize_start:
            max_start = len(self._domain) - self._min_episode_length
            if max_start > self._domain.lookback_window:
                start_bar = self.np_random.integers(
                    self._domain.lookback_window,
                    max_start,
                )
                logger.debug(f"Episode reset: randomized start at bar {start_bar}")
            else:
                logger.debug("Episode reset: insufficient data for randomization")

        # Reset domain
        state = self._domain.reset(start_bar=start_bar)

        # Track episode state
        self._episode_step = 0
        self._start_bar = self._domain.current_bar

        # Get observation (convert tuple to numpy array)
        obs = self._get_observation()
        info = self._state_to_info(state)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action index (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert gym action to domain intent
        intent = OrderIntent(action)

        # Process in domain
        result = self._domain.process_order(intent)

        # Increment episode step
        self._episode_step += 1

        # Get new observation
        obs = self._get_observation()

        # Extract reward as Python float
        reward = float(result.reward)

        # Check termination conditions
        terminated = result.new_state.is_terminated

        # Check if truncated (reached end of data)
        truncated = self._domain.current_bar >= len(self._domain) - 1

        # Build info dict
        info = self._state_to_info(result.new_state)
        info["action"] = action
        info["action_name"] = intent.name

        # Add trade info if a trade was executed
        if result.trade_executed:
            info["trade"] = {
                "side": result.trade_executed.side.value,
                "entry_price": result.trade_executed.entry_price,
                "exit_price": result.trade_executed.exit_price,
                "net_pnl": result.trade_executed.net_pnl,
                "return_pct": result.trade_executed.return_pct,
            }

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Get boolean mask of valid actions for current state.

        Required by MaskablePPO from sb3-contrib. Returns array where:
        - True = action is valid
        - False = action is invalid

        Returns:
            Boolean array of shape (3,) for (HOLD, BUY, SELL)
        """
        valid_actions = self._domain.get_valid_actions()
        return np.array(valid_actions, dtype=bool)

    def close_all_positions(self) -> None:
        """
        Close all open positions at current price.

        Should be called at episode end to realize all P&L.
        """
        trade = self._domain.close_all_positions()
        if trade:
            logger.debug(
                f"Closed position at episode end: "
                f"side={trade.side.value}, pnl={trade.net_pnl:.2f}"
            )

    def render(self) -> None:
        """Render the environment (console output)."""
        state = self._domain._get_state()
        metrics = self._domain.get_episode_metrics()

        print(
            f"Step: {self._episode_step}, "
            f"Bar: {state.current_bar}, "
            f"Portfolio: ${state.portfolio_value:.2f}, "
            f"Position: {state.position.size:.4f}, "
            f"Return: {metrics.get('total_return', 0):.2%}"
        )

    def get_trade_history(self) -> list:
        """
        Get history of completed trades.

        Returns:
            List of trade dictionaries (for backward compatibility)
        """
        return [trade.to_dict() for trade in self._domain.trade_history]

    # --- Private Methods ---

    def _get_observation(self) -> np.ndarray:
        """Get observation as numpy array."""
        obs_tuple = self._domain.get_observation()
        return np.array(obs_tuple, dtype=np.float32)

    def _state_to_info(self, state: TradingState) -> Dict[str, Any]:
        """Convert TradingState to info dictionary."""
        # Get action mask for info (for debugging/compatibility)
        action_mask_bool = self.action_masks()
        action_mask = action_mask_bool.astype(np.int8)

        info = {
            "step": self._episode_step,
            "bar": state.current_bar,
            "cash": state.cash_balance,
            "portfolio_value": state.portfolio_value,
            "position": state.position.size,
            "total_return": (state.portfolio_value - self._domain._config.initial_balance)
            / self._domain._config.initial_balance,
            "num_trades": self._domain.num_trades,
            "action_mask": action_mask,
        }

        if state.is_terminated:
            info["termination_reason"] = state.termination_reason

        # Add episode metrics when available
        if self._episode_step > 1:
            metrics = self._domain.get_episode_metrics()
            if "sharpe" in metrics:
                info["sharpe"] = metrics["sharpe"]
            if "max_drawdown" in metrics:
                info["max_drawdown"] = metrics["max_drawdown"]

        return info


def create_gym_trading_env(
    market_data: MarketDataPort,
    observation_features: list,
    config: Optional[TradingDomainConfig] = None,
    randomize_start: bool = True,
    **kwargs,
) -> GymTradingEnvAdapter:
    """
    Factory function to create a Gymnasium trading environment.

    This is the recommended way to create environments as it properly
    sets up the domain and adapter together.

    Args:
        market_data: MarketDataPort for accessing price/feature data
        observation_features: List of feature column names to include
        config: Optional TradingDomainConfig for customization
        randomize_start: Whether to randomize episode starting points
        **kwargs: Additional arguments passed to TradingDomain

    Returns:
        GymTradingEnvAdapter ready for use with RL frameworks

    Example:
        >>> from rl_trading_lab.infrastructure.adapters import (
        ...     ParquetMarketDataAdapter,
        ...     create_gym_trading_env,
        ... )
        >>>
        >>> # Load data
        >>> market_data = ParquetMarketDataAdapter(df)
        >>> features = ['close', 'volume', 'rsi', 'macd']
        >>>
        >>> # Create environment
        >>> env = create_gym_trading_env(
        ...     market_data=market_data,
        ...     observation_features=features,
        ...     config=TradingDomainConfig(initial_balance=50000),
        ... )
    """
    # Create domain
    domain = TradingDomain(
        market_data=market_data,
        observation_features=observation_features,
        config=config,
        **kwargs,
    )

    # Wrap in adapter
    return GymTradingEnvAdapter(
        domain=domain,
        randomize_start=randomize_start,
        min_episode_length=config.min_episode_length if config else None,
    )
