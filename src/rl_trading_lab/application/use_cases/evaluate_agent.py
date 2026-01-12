"""
EvaluateAgentUseCase - Use case for evaluating trained agents.

This use case handles:
1. Loading a trained agent
2. Running evaluation episodes
3. Computing performance metrics
4. Optional detailed backtest analysis
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rl_trading_lab.application.services.agent_service import AgentService
from rl_trading_lab.application.services.environment_service import EnvironmentService

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """
    Result of agent evaluation.

    Contains comprehensive performance metrics from running
    the agent on a test environment.
    """

    # Episode-level metrics
    mean_reward: float
    std_reward: float
    mean_episode_length: float
    total_episodes: int

    # Trading metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float

    # Raw data
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    # Additional metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"mean_reward={self.mean_reward:.4f}, "
            f"sharpe={self.sharpe_ratio:.2f}, "
            f"max_dd={self.max_drawdown:.2%}, "
            f"win_rate={self.win_rate:.2%})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/saving."""
        return {
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "mean_episode_length": self.mean_episode_length,
            "total_episodes": self.total_episodes,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            **self.metrics,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    # Model
    model_path: str
    algorithm: Optional[str] = None  # Auto-detected if None

    # Data
    data_path: str = ""
    observation_features: List[str] = field(default_factory=list)

    # Environment (should match training)
    initial_balance: float = 10000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_pct: float = 0.95
    lookback_window: int = 20
    min_episode_length: int = 100
    reward_type: str = "returns"
    max_drawdown_pct: float = 0.30
    min_portfolio_pct: float = 0.20

    # Evaluation
    n_episodes: int = 10
    deterministic: bool = True
    device: str = "auto"


class EvaluateAgentUseCase:
    """
    Use case: Evaluate a trained RL agent.

    This use case:
    1. Loads the trained agent from checkpoint
    2. Creates test environment
    3. Runs multiple evaluation episodes
    4. Computes comprehensive performance metrics

    Example:
        >>> eval_use_case = EvaluateAgentUseCase(
        ...     environment_service=env_service,
        ...     agent_service=agent_service,
        ... )
        >>>
        >>> config = EvaluationConfig(
        ...     model_path="models/best_model",
        ...     data_path="data/btc_features.parquet",
        ...     observation_features=["close", "volume", "rsi"],
        ... )
        >>> result = eval_use_case.execute(config)
        >>> print(f"Sharpe: {result.sharpe_ratio:.2f}")
    """

    # Constants
    TRADING_DAYS_PER_YEAR = 252
    POSITION_EPSILON = 1e-3

    def __init__(
        self,
        environment_service: EnvironmentService,
        agent_service: AgentService,
    ):
        """
        Initialize the evaluation use case.

        Args:
            environment_service: Service for creating environments
            agent_service: Service for agent management
        """
        self._env_service = environment_service
        self._agent_service = agent_service

    def execute(self, config: EvaluationConfig) -> EvaluationResult:
        """
        Execute the evaluation use case.

        Args:
            config: Evaluation configuration

        Returns:
            EvaluationResult with metrics
        """
        # 1. Create test environment
        logger.info("Creating test environment...")
        test_env = self._env_service.create_test_env(
            data_path=config.data_path,
            observation_features=config.observation_features,
            initial_balance=config.initial_balance,
            commission_rate=config.commission_rate,
            slippage_rate=config.slippage_rate,
            max_position_pct=config.max_position_pct,
            lookback_window=config.lookback_window,
            min_episode_length=config.min_episode_length,
            reward_type=config.reward_type,
            max_drawdown_pct=config.max_drawdown_pct,
            min_portfolio_pct=config.min_portfolio_pct,
        )

        # 2. Load agent
        logger.info(f"Loading agent from {config.model_path}...")
        agent, vec_env = self._agent_service.load_agent(
            model_path=Path(config.model_path),
            env=test_env,
            algorithm=config.algorithm,
            device=config.device,
        )

        # 3. Run evaluation episodes
        logger.info(f"Running {config.n_episodes} evaluation episodes...")
        episode_results = self._run_episodes(
            agent=agent,
            vec_env=vec_env,
            n_episodes=config.n_episodes,
            deterministic=config.deterministic,
            initial_balance=config.initial_balance,
        )

        # 4. Compute metrics
        result = self._compute_metrics(episode_results, config)

        logger.info(f"Evaluation complete: {result}")

        return result

    def _run_episodes(
        self,
        agent,
        vec_env,
        n_episodes: int,
        deterministic: bool,
        initial_balance: float,
    ) -> Dict[str, List]:
        """
        Run evaluation episodes and collect data.

        Returns:
            Dictionary with episode data
        """
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        all_trades = []
        all_positions = []
        all_step_returns = []

        for ep in range(n_episodes):
            obs = vec_env.reset()
            done = False
            truncated = False

            episode_reward = 0.0
            episode_length = 0
            positions = []
            portfolio_values = [initial_balance]

            while not done and not truncated:
                # Get action
                action, _ = agent.predict(obs, deterministic=deterministic)

                # Step
                step_result = vec_env.step(action)

                # Handle both old and new gym API
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                    truncated = False
                else:
                    obs, reward, done, truncated, info = step_result

                # Extract from vectorized format
                if isinstance(info, list):
                    info = info[0]
                if isinstance(done, np.ndarray):
                    done = done[0]
                if isinstance(truncated, (np.ndarray, bool)):
                    truncated = truncated[0] if isinstance(truncated, np.ndarray) else truncated

                # Collect data
                episode_reward += float(reward)
                episode_length += 1
                positions.append(info.get("position", 0))
                portfolio_values.append(info.get("portfolio_value", initial_balance))

            # Calculate episode return
            final_value = portfolio_values[-1]
            episode_return = (final_value - initial_balance) / initial_balance

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(episode_return)
            all_positions.extend(positions)

            # Calculate step returns
            for i in range(1, len(portfolio_values)):
                step_ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                all_step_returns.append(step_ret)

            # Get trades from environment (unwrap to access)
            try:
                unwrapped = vec_env.envs[0] if hasattr(vec_env, 'envs') else vec_env
                if hasattr(unwrapped, 'env'):
                    unwrapped = unwrapped.env
                if hasattr(unwrapped, 'get_trade_history'):
                    all_trades.extend(unwrapped.get_trade_history())
            except (AttributeError, IndexError):
                pass

            logger.debug(f"Episode {ep+1}: reward={episode_reward:.4f}, return={episode_return:.2%}")

        return {
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "returns": episode_returns,
            "positions": all_positions,
            "step_returns": all_step_returns,
            "trades": all_trades,
        }

    def _compute_metrics(
        self,
        episode_data: Dict[str, List],
        config: EvaluationConfig,
    ) -> EvaluationResult:
        """
        Compute comprehensive metrics from episode data.
        """
        rewards = np.array(episode_data["rewards"])
        lengths = np.array(episode_data["lengths"])
        returns = np.array(episode_data["returns"])
        step_returns = np.array(episode_data["step_returns"])
        trades = episode_data["trades"]

        # Basic metrics
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_length = float(np.mean(lengths))

        # Return metrics
        total_return = float(np.mean(returns))

        # Sharpe ratio from step returns
        if len(step_returns) > 1 and np.std(step_returns) > 1e-8:
            sharpe = float(
                np.mean(step_returns) / np.std(step_returns) *
                np.sqrt(self.TRADING_DAYS_PER_YEAR)
            )
        else:
            sharpe = 0.0

        # Max drawdown
        max_dd = self._compute_max_drawdown(step_returns)

        # Trade metrics
        num_trades = self._count_trades(episode_data["positions"])
        win_rate = self._compute_win_rate(trades)

        return EvaluationResult(
            mean_reward=mean_reward,
            std_reward=std_reward,
            mean_episode_length=mean_length,
            total_episodes=config.n_episodes,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            num_trades=num_trades,
            win_rate=win_rate,
            episode_rewards=list(rewards),
            episode_lengths=list(lengths.astype(int)),
            metrics={
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "avg_trades_per_episode": num_trades / config.n_episodes if config.n_episodes > 0 else 0,
            },
        )

    def _compute_max_drawdown(self, step_returns: np.ndarray) -> float:
        """Compute maximum drawdown from step returns."""
        if len(step_returns) == 0:
            return 0.0

        cumulative = np.cumsum(step_returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / (np.abs(peak) + 1e-8)
        return float(np.max(drawdown))

    def _count_trades(self, positions: List[float]) -> int:
        """Count number of trades from position history."""
        if len(positions) == 0:
            return 0

        total_trades = 0

        # First position opening
        if abs(positions[0]) > self.POSITION_EPSILON:
            total_trades += 1

        # Subsequent position changes
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]

            prev_is_flat = abs(prev_pos) < self.POSITION_EPSILON
            curr_is_flat = abs(curr_pos) < self.POSITION_EPSILON

            # Opening from flat
            if prev_is_flat and not curr_is_flat:
                total_trades += 1
            # Reversing direction
            elif not prev_is_flat and not curr_is_flat:
                if np.sign(prev_pos) != np.sign(curr_pos):
                    total_trades += 1

        return total_trades

    def _compute_win_rate(self, trades: List) -> float:
        """Compute win rate from trade history."""
        if not trades:
            return 0.0

        winning_trades = sum(
            1 for t in trades
            if (isinstance(t, dict) and t.get("net_pnl", 0) > 0) or
               (hasattr(t, "net_pnl") and t.net_pnl > 0)
        )
        return winning_trades / len(trades)
