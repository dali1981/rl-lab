"""
Callback factory for SB3 training.

Creates training callbacks with lazy imports for optional dependencies
(MLflow, TensorBoard). If a dependency is not installed, the corresponding
callback is simply not created.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

logger = logging.getLogger(__name__)


class CallbackFactory:
    """
    Creates training callbacks based on configuration.

    MLflow and TensorBoard are optional - if not installed,
    the corresponding callbacks are silently skipped.

    Example:
        >>> factory = CallbackFactory()
        >>> callbacks = factory.create_all(
        ...     eval_env=eval_env,
        ...     save_path=Path("checkpoints"),
        ...     eval_freq=5000,
        ...     save_freq=10000,
        ... )
    """

    def create_eval_callback(
        self,
        eval_env,
        save_path: Path,
        eval_freq: int,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaseCallback:
        """Create evaluation callback that saves best model with metadata."""
        from rl_trading_lab.utils.custom_callbacks import BestModelCallback

        return BestModelCallback(
            eval_env,
            best_model_save_path=str(save_path / "best_model"),
            log_path=str(save_path / "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
            metadata=metadata or {},
        )

    def create_checkpoint_callback(
        self,
        save_path: Path,
        save_freq: int,
        verbose: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BaseCallback:
        """Create checkpoint callback that saves models at regular intervals."""
        from rl_trading_lab.utils.custom_callbacks import CheckpointManagerCallback

        return CheckpointManagerCallback(
            save_freq=save_freq,
            save_path=str(save_path / "checkpoints"),
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=verbose,
            metadata=metadata or {},
        )

    def create_trading_metrics_callback(
        self,
        one_trade_mode: bool = False,
        verbose: int = 1,
    ) -> Optional[BaseCallback]:
        """
        Create trading metrics callback (disabled in one_trade_mode).

        Returns None if one_trade_mode is True, since each episode
        is a single trade and win/loss tracking is redundant.
        """
        if one_trade_mode:
            logger.info("TradingMetricsCallback disabled (one_trade_mode=True)")
            return None

        from rl_trading_lab.utils.callbacks import TradingMetricsCallback

        logger.info("TradingMetricsCallback enabled (multi-trade mode)")
        return TradingMetricsCallback(verbose=verbose)

    def create_logging_setup(
        self,
        tensorboard_log: Optional[str] = None,
    ) -> tuple:
        """
        Create logger output formats for MLflow and TensorBoard.

        Returns:
            Tuple of (format_strings, custom_output_formats) for SB3 logger setup.
            format_strings: List of standard SB3 format names (e.g. ["stdout", "tensorboard"])
            custom_output_formats: List of custom KVWriter instances (e.g. MLflowOutputFormat)
        """
        format_strings = ["stdout"]
        custom_output_formats = []

        if tensorboard_log:
            format_strings.append("tensorboard")

        # Lazy import MLflow - only add if installed and an active run exists
        try:
            import mlflow
            if mlflow.active_run():
                from rl_trading_lab.utils.mlflow_logger import MLflowOutputFormat
                custom_output_formats.append(MLflowOutputFormat())
                logger.info("MLflow logging enabled")
        except ImportError:
            pass

        return format_strings, custom_output_formats

    def create_all(
        self,
        eval_env=None,
        save_path: Optional[Path] = None,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        save_freq: Optional[int] = None,
        one_trade_mode: bool = False,
        verbose: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        extra_callbacks: Optional[List[BaseCallback]] = None,
    ) -> Optional[CallbackList]:
        """
        Create all training callbacks in one call.

        Args:
            eval_env: Evaluation environment (optional)
            save_path: Base path for saving models/checkpoints
            eval_freq: Evaluation frequency in timesteps
            n_eval_episodes: Episodes per evaluation
            save_freq: Checkpoint frequency in timesteps
            one_trade_mode: Whether training uses one-trade-per-episode mode
            verbose: Verbosity level
            metadata: Metadata to attach to checkpoints
            extra_callbacks: Additional user-provided callbacks

        Returns:
            Combined CallbackList, or None if no callbacks
        """
        callbacks = []

        if eval_env is not None and eval_freq and save_path:
            callbacks.append(self.create_eval_callback(
                eval_env=eval_env,
                save_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                verbose=verbose,
                metadata=metadata,
            ))

        if save_freq and save_path:
            callbacks.append(self.create_checkpoint_callback(
                save_path=save_path,
                save_freq=save_freq,
                verbose=verbose,
                metadata=metadata,
            ))

        trading_cb = self.create_trading_metrics_callback(
            one_trade_mode=one_trade_mode,
            verbose=verbose,
        )
        if trading_cb:
            callbacks.append(trading_cb)

        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        return CallbackList(callbacks) if callbacks else None
