"""
Trading metrics callback for Stable-Baselines3 training.
Tracks comprehensive trading-specific metrics during multi-trade episodes.
"""

import logging
import numpy as np
import mlflow
from typing import Dict, Any
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class TradingMetricsCallback(BaseCallback):
    """
    Enhanced callback to track trading-specific metrics during training.

    Logs comprehensive trading metrics including:
    - Win rate and trade statistics
    - Risk metrics (Sharpe, Sortino, max drawdown)
    - Position statistics
    - Profit/loss metrics

    NOTE: This callback is intended for multi-trade episodes where each episode
    contains multiple trades. For one_trade_mode (where each episode is a single trade),
    this callback should not be used as win/loss tracking is redundant with episode rewards.
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

        # Log episode completion at debug level
        sharpe = info.get("sharpe", 0)
        ret = info.get("total_return", 0)
        logger.debug(f"Episode complete: Return={ret:.2%}, Sharpe={sharpe:.2f}")

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
