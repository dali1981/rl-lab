"""
MLflow callback for Stable-Baselines3 training.
Logs metrics to MLflow in real-time during training.
"""

import logging
import numpy as np
import mlflow
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


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

        logger.debug(f"Logged training metrics at step {self.num_timesteps}")

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

        # Log episode completion at debug level
        logger.debug(f"Episode {self.episode_count} completed: "
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
