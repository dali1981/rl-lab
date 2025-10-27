"""
Custom callbacks for Stable-Baselines3 training.
Provides MLflow integration and trading-specific metrics logging.
"""

import numpy as np
import mlflow
from typing import Optional, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback


class MLflowCallback(BaseCallback):
    """
    Callback for logging metrics to MLflow during training.

    Logs metrics in real-time throughout training using SB3 callback hooks:
    - Training metrics every `log_freq` steps
    - Rollout statistics at end of each rollout
    - Episode metrics when episodes complete

    Usage:
        callback = MLflowCallback(log_freq=1000)
        model.learn(total_timesteps=100000, callback=callback)
    """

    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 0,
    ):
        """
        Initialize MLflow callback.

        Args:
            log_freq: Log metrics every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0

        # Accumulators for episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_trading_metrics = []

    def _init_callback(self) -> None:
        """Called once at the beginning of training."""
        # Check if MLflow run is active
        if not mlflow.active_run():
            self.logger.warning("No active MLflow run. Metrics will not be logged.")

    def _on_step(self) -> bool:
        """
        Called at every step of training.

        Logs metrics every `log_freq` steps and when episodes complete.
        """
        # Log metrics at specified frequency
        if self.n_calls % self.log_freq == 0 and mlflow.active_run():
            self._log_training_metrics()

        # Check for completed episodes
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._log_episode_metrics(i)

        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout (collection phase).

        Logs rollout statistics to MLflow.
        """
        if not mlflow.active_run():
            return

        # Log rollout buffer statistics if available
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            buffer = self.model.rollout_buffer

            if buffer.full or buffer.pos > 0:
                # Calculate statistics from rollout buffer
                rewards = buffer.rewards.flatten()
                values = buffer.values.flatten()
                advantages = buffer.advantages.flatten() if hasattr(buffer, 'advantages') else None

                mlflow.log_metrics({
                    "rollout/mean_reward": float(np.mean(rewards)),
                    "rollout/std_reward": float(np.std(rewards)),
                    "rollout/mean_value": float(np.mean(values)),
                    "rollout/std_value": float(np.std(values)),
                }, step=self.num_timesteps)

                if advantages is not None:
                    mlflow.log_metrics({
                        "rollout/mean_advantage": float(np.mean(advantages)),
                        "rollout/std_advantage": float(np.std(advantages)),
                    }, step=self.num_timesteps)

    def _log_training_metrics(self) -> None:
        """Log current training progress metrics."""
        metrics = {
            "time/total_timesteps": self.num_timesteps,
            "time/episodes": self.episode_count,
        }

        # Add FPS if available
        if hasattr(self.model, 'num_timesteps') and hasattr(self.model, '_total_timesteps'):
            if self.model._total_timesteps > 0:
                progress = self.num_timesteps / self.model._total_timesteps
                metrics["time/progress"] = progress

        mlflow.log_metrics(metrics, step=self.num_timesteps)

        if self.verbose >= 1:
            print(f"[MLflowCallback] Logged training metrics at step {self.num_timesteps}")

    def _log_episode_metrics(self, env_idx: int) -> None:
        """
        Log metrics for a completed episode.

        Args:
            env_idx: Index of the environment that completed an episode
        """
        if not mlflow.active_run():
            return

        # Get episode info from the environment
        info = self.locals["infos"][env_idx]

        # Log standard episode metrics
        if "episode" in info:
            episode_info = info["episode"]
            episode_reward = episode_info["r"]
            episode_length = episode_info["l"]

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_count += 1

            mlflow.log_metrics({
                "episode/reward": float(episode_reward),
                "episode/length": float(episode_length),
                "episode/count": self.episode_count,
            }, step=self.num_timesteps)

            # Log rolling statistics (last 100 episodes)
            if len(self.episode_rewards) >= 10:
                recent_rewards = self.episode_rewards[-100:]
                mlflow.log_metrics({
                    "episode/mean_reward_100": float(np.mean(recent_rewards)),
                    "episode/std_reward_100": float(np.std(recent_rewards)),
                }, step=self.num_timesteps)

        # Log trading-specific metrics if available
        trading_metrics = self._extract_trading_metrics(info)
        if trading_metrics:
            self.episode_trading_metrics.append(trading_metrics)

            # Log individual trading metrics
            mlflow.log_metrics({
                f"trading/{key}": float(value)
                for key, value in trading_metrics.items()
            }, step=self.num_timesteps)

            # Log rolling trading statistics
            if len(self.episode_trading_metrics) >= 10:
                self._log_rolling_trading_metrics()

        if self.verbose >= 1:
            print(f"[MLflowCallback] Episode {self.episode_count} completed: "
                  f"reward={episode_reward:.2f}, length={episode_length}")

    def _extract_trading_metrics(self, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract trading-specific metrics from environment info.

        Args:
            info: Info dict from environment

        Returns:
            Dictionary of trading metrics
        """
        metrics = {}

        # Common trading metrics
        trading_keys = [
            "total_return",
            "sharpe",
            "max_drawdown",
            "portfolio_value",
            "position",
            "balance",
        ]

        for key in trading_keys:
            if key in info:
                value = info[key]
                if isinstance(value, (int, float, np.number)):
                    metrics[key] = value

        return metrics

    def _log_rolling_trading_metrics(self) -> None:
        """Log rolling statistics for trading metrics."""
        if len(self.episode_trading_metrics) < 10:
            return

        recent_metrics = self.episode_trading_metrics[-100:]

        # Calculate statistics for each metric
        metric_keys = set()
        for m in recent_metrics:
            metric_keys.update(m.keys())

        rolling_stats = {}
        for key in metric_keys:
            values = [m[key] for m in recent_metrics if key in m]
            if values:
                rolling_stats[f"trading_rolling/{key}_mean"] = float(np.mean(values))
                rolling_stats[f"trading_rolling/{key}_std"] = float(np.std(values))

                # Add min/max for certain metrics
                if key in ["total_return", "sharpe"]:
                    rolling_stats[f"trading_rolling/{key}_min"] = float(np.min(values))
                    rolling_stats[f"trading_rolling/{key}_max"] = float(np.max(values))

        if rolling_stats:
            mlflow.log_metrics(rolling_stats, step=self.num_timesteps)


class TradingMetricsCallback(BaseCallback):
    """
    Enhanced callback to track trading-specific metrics during training.

    Logs comprehensive trading metrics including:
    - Win rate and trade statistics
    - Risk metrics (Sharpe, Sortino, max drawdown)
    - Position statistics
    - Profit/loss metrics
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []

        # Trading-specific accumulators
        self.trades_count = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def _on_step(self) -> bool:
        """Called at every step."""
        # Check if any episode finished
        if len(self.locals.get("dones", [])) > 0:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self._log_episode_trading_metrics(i)

        return True

    def _log_episode_trading_metrics(self, env_idx: int) -> None:
        """
        Log detailed trading metrics for completed episode.

        Args:
            env_idx: Index of the environment that completed
        """
        info = self.locals["infos"][env_idx]

        # Store basic metrics
        if "episode" in info:
            self.episode_rewards.append(info["episode"]["r"])
            self.episode_lengths.append(info["episode"]["l"])

        # Extract and store trading metrics
        if "total_return" in info:
            metrics = {
                "sharpe": info.get("sharpe", 0),
                "total_return": info.get("total_return", 0),
                "max_drawdown": info.get("max_drawdown", 0),
                "portfolio_value": info.get("portfolio_value", 0),
            }
            self.episode_metrics.append(metrics)

            # Update trade statistics
            if info.get("total_return", 0) > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.trades_count += 1

            # Log to SB3 logger (which includes MLflow via MLflowOutputFormat)
            if self.logger:
                self.logger.record("trading/sharpe", info["sharpe"])
                self.logger.record("trading/total_return", info["total_return"])
                self.logger.record("trading/max_drawdown", info["max_drawdown"])
                self.logger.record("trading/portfolio_value", info["portfolio_value"])

                # Log trade statistics
                if self.trades_count > 0:
                    win_rate = self.winning_trades / self.trades_count
                    self.logger.record("trading/win_rate", win_rate)
                    self.logger.record("trading/total_trades", self.trades_count)
                    self.logger.record("trading/winning_trades", self.winning_trades)
                    self.logger.record("trading/losing_trades", self.losing_trades)

            # Also log directly to MLflow for real-time tracking
            if mlflow.active_run():
                mlflow.log_metrics({
                    "trading/sharpe": float(info["sharpe"]),
                    "trading/total_return": float(info["total_return"]),
                    "trading/max_drawdown": float(info["max_drawdown"]),
                    "trading/portfolio_value": float(info["portfolio_value"]),
                }, step=self.num_timesteps)

                if self.trades_count > 0:
                    mlflow.log_metrics({
                        "trading/win_rate": float(self.winning_trades / self.trades_count),
                        "trading/total_trades": self.trades_count,
                    }, step=self.num_timesteps)

        if self.verbose >= 1:
            sharpe = info.get("sharpe", 0)
            ret = info.get("total_return", 0)
            print(f"[TradingMetrics] Episode complete: Return={ret:.2%}, Sharpe={sharpe:.2f}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for all episodes.

        Returns:
            Dictionary with training statistics
        """
        stats = {
            "total_episodes": len(self.episode_rewards),
            "total_trades": self.trades_count,
        }

        if self.episode_rewards:
            stats["mean_episode_reward"] = float(np.mean(self.episode_rewards))
            stats["std_episode_reward"] = float(np.std(self.episode_rewards))

        if self.trades_count > 0:
            stats["win_rate"] = self.winning_trades / self.trades_count
            stats["winning_trades"] = self.winning_trades
            stats["losing_trades"] = self.losing_trades

        if self.episode_metrics:
            for key in ["sharpe", "total_return", "max_drawdown"]:
                values = [m[key] for m in self.episode_metrics if key in m]
                if values:
                    stats[f"mean_{key}"] = float(np.mean(values))
                    stats[f"std_{key}"] = float(np.std(values))

        return stats
