#!/usr/bin/env python
"""
Main training script for RL trading agents.
Uses Hydra for configuration and MLflow for tracking.

Usage:
    python train.py
    python train.py agent=ppo env.reward_type=sharpe
    python train.py +experiment=hyperopt
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import subprocess
from typing import Callable, Dict, Any, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_trading_lab.config import RootConfig, load_config
from rl_trading_lab.environment import TradingEnv, create_make_env
from rl_trading_lab.agents.sb3_agents import Trainer

# Constants
POSITION_EPSILON = 1e-3  # Threshold for considering a position as non-zero
TRADING_DAYS_PER_YEAR = 252  # Standard trading days for Sharpe calculation
GIT_HASH_SHORT_LENGTH = 8

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


# ============================================================================
# Helper Functions
# ============================================================================

def unwrap_vectorized_env(vec_env):
    """
    Unwrap the base environment from a vectorized environment.

    Handles both VecNormalize and DummyVecEnv wrappers.

    Args:
        vec_env: Vectorized environment (DummyVecEnv or VecNormalize)

    Returns:
        Unwrapped TradingEnv instance
    """
    if hasattr(vec_env, 'venv'):
        # VecNormalize wrapper
        return vec_env.venv.envs[0]
    else:
        # DummyVecEnv
        return vec_env.envs[0]


def extract_vec_env_step_result(step_result) -> Tuple[Any, float, bool, bool, Dict]:
    """
    Extract step results from vectorized environment, handling both gym and gymnasium APIs.

    Args:
        step_result: Result tuple from vec_env.step()

    Returns:
        Tuple of (obs, reward, done, truncated, info) normalized to gymnasium format
    """
    if len(step_result) == 4:
        # Old gym API (VecNormalize)
        obs, reward, done, info = step_result
        truncated = False
    else:
        # New gymnasium API
        obs, reward, done, truncated, info = step_result

    # Extract from vectorized format
    if isinstance(info, list):
        info = info[0]
    if isinstance(done, np.ndarray):
        done = done[0]
    if isinstance(truncated, (np.ndarray, bool)):
        truncated = truncated[0] if isinstance(truncated, np.ndarray) else truncated

    return obs, reward, done, truncated, info


def count_trades(positions: List[float]) -> int:
    """
    Count the number of actual trades from position history.

    A trade occurs when:
    1. Opening a position from flat (0 -> long/short)
    2. Reversing direction (long -> short or short -> long)

    Args:
        positions: List of position sizes over time

    Returns:
        Total number of trades
    """
    if len(positions) == 0:
        return 0

    total_trades = 0

    # First step: trade if we open a position
    if abs(positions[0]) > POSITION_EPSILON:
        total_trades += 1

    # Subsequent steps: detect position changes
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]

        prev_is_flat = abs(prev_pos) < POSITION_EPSILON
        curr_is_flat = abs(curr_pos) < POSITION_EPSILON

        # Opening position from flat
        if prev_is_flat and not curr_is_flat:
            total_trades += 1
        # Reversing direction (both non-flat but opposite signs)
        elif not prev_is_flat and not curr_is_flat:
            if np.sign(prev_pos) != np.sign(curr_pos):
                total_trades += 1

    return total_trades


def calculate_sharpe_ratio(returns: np.ndarray, annualization_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Calculate annualized Sharpe ratio from returns.

    Args:
        returns: Array of period returns
        annualization_factor: Number of periods per year (252 for daily trading)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    mean_return = returns.mean()
    std_return = returns.std()

    if std_return < 1e-8:
        return 0.0

    return mean_return / std_return * np.sqrt(annualization_factor)


# ============================================================================
# Main Functions
# ============================================================================

def setup_mlflow(
    mlflow_config,
    run_name: str,
    full_config_dict: Dict[str, Any],
    base_params: Dict[str, Any]
):
    """
    Setup MLflow tracking.

    Args:
        mlflow_config: MLflow-specific configuration section
        run_name: Name for this experiment run
        full_config_dict: Complete config dictionary for artifact logging
        base_params: Key parameters to log (agent name, env params, etc.)
    """
    if not mlflow_config.enabled:
        return

    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)

    # Create or set experiment
    mlflow.set_experiment(mlflow_config.experiment_name)

    # Start run
    mlflow.start_run(run_name=run_name)

    # Log base parameters
    mlflow.log_params(base_params)

    # Log full config as artifact for reproducibility
    mlflow.log_dict(full_config_dict, "config.yaml")

    # Log all individual config files for complete reproducibility
    configs_dir = Path(__file__).parent.parent / "configs"
    if configs_dir.exists():
        mlflow.log_artifacts(str(configs_dir), "configs")
        logger.info(f"All config files logged ({len(list(configs_dir.rglob('*.yaml')))} YAML files)")

    # Log Hydra CLI overrides
    try:
        hydra_cfg = HydraConfig.get()
        cli_overrides = hydra_cfg.overrides.task
        if cli_overrides:
            mlflow.log_dict({"cli_overrides": cli_overrides}, "hydra/cli_overrides.yaml")
            logger.info(f"Hydra overrides logged: {cli_overrides}")
    except Exception as e:
        logger.debug(f"Could not log Hydra overrides: {e}")

    # Log git commit hash for code versioning
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        mlflow.log_param("git_commit", git_hash)
        logger.info(f"Git commit logged: {git_hash[:8]}")
    except Exception as e:
        logger.debug(f"Could not log git commit: {e}")

    logger.info(f"MLflow tracking enabled: {mlflow_config.tracking_uri}")
    logger.info("Full config logged to MLflow")


def _prepare_mlflow_params(config: RootConfig) -> Dict[str, Any]:
    """
    Prepare MLflow parameters from config.

    Args:
        config: Full training configuration

    Returns:
        Dictionary of parameters to log to MLflow
    """
    mlflow_params = {
        "agent": config.agent.name,
        "environment.reward_type": config.env.environment_params.reward_type,
        "environment.initial_balance": config.env.environment_params.initial_balance,
        "environment.commission_rate": config.env.environment_params.commission_rate,
        "environment.lookback_window": config.env.environment_params.lookback_window,
        "training.total_timesteps": config.training.total_timesteps,
    }

    # Add agent hyperparameters
    for key, value in config.agent.hyperparameters.model_dump().items():
        if not isinstance(value, (dict, list)):
            mlflow_params[f"agent.{key}"] = value

    return mlflow_params


def wrap_test_env_for_evaluation(make_env, trainer: Trainer):
    """
    Create and wrap test environment to match training setup.

    CRITICAL: If trainer was trained with VecNormalize, we must wrap the test environment
    the same way and copy the normalization statistics from training to ensure
    consistent observation scaling.

    Args:
        make_env: Factory function to create environments
        trainer: Trained Trainer instance

    Returns:
        Properly wrapped test environment
    """
    # Create test environment
    test_env = make_env('test')

    # Use trainer's wrapping method to ensure consistency
    test_vec_env = trainer._wrap_environment(test_env, is_eval=True)

    # Check if VecNormalize was used during training
    if trainer.vec_normalize_enabled:
        logger.info("Wrapping test environment with VecNormalize...")

        # CRITICAL: Copy normalization statistics from training environment
        # Without this, the test env would use different normalization!
        if hasattr(trainer.env, 'obs_rms'):
            test_vec_env.obs_rms = trainer.env.obs_rms
            logger.info("Copied observation normalization stats from training")
        else:
            logger.warning("Could not copy normalization stats!")

        if hasattr(trainer.env, 'ret_rms'):
            test_vec_env.ret_rms = trainer.env.ret_rms
    else:
        logger.info("VecNormalize not used during training - using DummyVecEnv only")

    return test_vec_env


def train_agent(
    agent_name: str,
    trainer: Trainer,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    save_freq: int,
    progress_bar: bool
) -> Tuple[Trainer, Dict[str, Any]]:
    """
    Train the RL agent.

    Args:
        agent_name: Name of the agent (for logging)
        trainer: Trainer instance
        total_timesteps: Total training timesteps
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of episodes for evaluation
        save_freq: Checkpoint save frequency
        progress_bar: Whether to show progress bar

    Returns:
        Tuple of (trained trainer, training metrics)
    """
    logger.info(f"Training {agent_name} agent...")

    # Train
    metrics = trainer.train(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        save_freq=save_freq,
        progress_bar=progress_bar,
    )

    logger.info("Training completed")

    return trainer, metrics


def evaluate_final_performance(trainer: Trainer, make_env, n_episodes=10):
    """Evaluate final model performance"""
    logger.info("Evaluating final performance...")

    # Create and wrap test environment to match training setup (critical for correct evaluation)
    test_env_wrapped = wrap_test_env_for_evaluation(make_env, trainer)

    # Evaluate using wrapped environment
    metrics = trainer.evaluate(
        env=test_env_wrapped,
        n_episodes=n_episodes,
        deterministic=True,
    )

    # Log episode rewards stats at debug level
    logger.debug(f"Episode rewards - Mean: {metrics.get('mean_reward', 'N/A')}, "
                f"Std: {metrics.get('std_reward', 'N/A')}")

    # Create results table
    table = Table(title="Test Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in metrics.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)

    return metrics


def _collect_backtest_trajectory(trainer: Trainer, vec_env) -> Dict[str, List]:
    """
    Run episode and collect trajectory data.

    Args:
        trainer: Trained agent
        vec_env: Vectorized environment

    Returns:
        Dictionary containing actions, rewards, positions, balances, and step_returns
    """
    obs = vec_env.reset()
    done = False
    truncated = False

    actions = []
    rewards = []
    positions = []
    balances = []
    step_returns = []

    while not done and not truncated:
        # Get action from trainer
        action, _ = trainer.predict(obs, deterministic=True)

        # Step environment and extract results using helper
        step_result = vec_env.step(action)
        obs, reward, done, truncated, info = extract_vec_env_step_result(step_result)

        # Collect data
        actions.append(action)
        rewards.append(reward)
        positions.append(info.get('position', 0))

        # Get portfolio value from info dict (already computed by environment)
        portfolio_value = info.get('portfolio_value', 0)
        balances.append(portfolio_value)

        # Calculate step return for Sharpe calculation
        if len(balances) > 1:
            step_return = (balances[-1] - balances[-2]) / balances[-2]
            step_returns.append(step_return)

    return {
        'actions': actions,
        'rewards': rewards,
        'positions': positions,
        'balances': balances,
        'step_returns': step_returns,
    }


def _calculate_final_metrics(vec_env, trajectory: Dict[str, List]) -> Dict[str, float]:
    """
    Calculate final backtest metrics including P&L and Sharpe ratio.

    Args:
        vec_env: Vectorized environment (to access unwrapped env)
        trajectory: Collected trajectory data

    Returns:
        Dictionary of final metrics
    """
    # Unwrap environment using helper
    unwrapped_env = unwrap_vectorized_env(vec_env)

    # Close any remaining open positions to realize all P&L
    unwrapped_env.close_all_positions()

    # Get final portfolio value after closing positions
    current_price = unwrapped_env._get_current_price()
    final_portfolio_value = unwrapped_env.portfolio.get_portfolio_value(current_price)

    # Calculate final return
    initial_balance = unwrapped_env.initial_balance
    final_return = (final_portfolio_value - initial_balance) / initial_balance

    # Calculate Sharpe from actual returns (not from Sharpe rewards)
    step_returns = np.array(trajectory['step_returns'])
    sharpe = calculate_sharpe_ratio(step_returns)

    # Count actual trades using helper function
    total_trades = count_trades(trajectory['positions'])
    trade_frequency = total_trades / len(trajectory['actions']) if len(trajectory['actions']) > 0 else 0

    return {
        'final_return': final_return,
        'sharpe_ratio': sharpe,
        'num_trades': total_trades,
        'trade_frequency': trade_frequency,
    }


def _log_backtest_results(metrics: Dict[str, float], trajectory: Dict[str, List]):
    """
    Log backtest results to console and MLflow.

    Args:
        metrics: Calculated performance metrics
        trajectory: Collected trajectory data
    """
    # Log backtest summary
    logger.info(f"Backtest completed - Steps: {len(trajectory['actions'])}, "
               f"Final Return: {metrics['final_return']:.2%}, "
               f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
               f"Trades: {metrics['num_trades']}, "
               f"Trade Frequency: {metrics['trade_frequency']:.1%}")

    # Log reward statistics at debug level
    rewards = np.array(trajectory['rewards'])
    logger.debug(f"Reward statistics - Mean: {rewards.mean():.4f}, Std: {rewards.std():.4f}, "
                f"Min/Max: {rewards.min():.4f}/{rewards.max():.4f}")

    # Log backtest metrics to MLflow
    # Note: Training metrics are logged via callbacks during training.
    # These backtest metrics are computed AFTER training completes on the test set.
    if mlflow.active_run():
        mlflow.log_metrics({
            "backtest/final_return": metrics['final_return'],
            "backtest/sharpe_ratio": metrics['sharpe_ratio'],
            "backtest/num_trades": metrics['num_trades'],
            "backtest/trade_frequency": metrics['trade_frequency'],
        })


def run_backtest(trainer: Trainer, make_env: Callable) -> Dict[str, Any]:
    """
    Run full backtest and collect detailed metrics.

    Args:
        trainer: Trained Trainer instance
        make_env: Factory function to create environments

    Returns:
        Dictionary containing backtest results including trajectory and metrics
    """
    logger.info("Running backtest...")

    # Create and wrap test environment to match training setup
    vec_env = wrap_test_env_for_evaluation(make_env, trainer)

    # Collect trajectory data
    trajectory = _collect_backtest_trajectory(trainer, vec_env)

    # Calculate final metrics
    metrics = _calculate_final_metrics(vec_env, trajectory)

    # Log results
    _log_backtest_results(metrics, trajectory)

    # Return combined results
    return {
        **metrics,
        'actions': trajectory['actions'],
        'positions': trajectory['positions'],
        'balances': trajectory['balances'],
        'step_returns': trajectory['step_returns'],
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training pipeline"""
    # Load and validate config using Pydantic
    config = load_config(cfg)

    console.print("[bold blue]RL Trading Lab[/bold blue]")
    console.print(f"Agent: [cyan]{config.agent.name}[/cyan]")
    console.print(f"Environment: [cyan]{config.env.environment_params.reward_type}[/cyan] reward")
    console.print(f"Device: [cyan]{config.experiment.device}[/cyan]\n")

    # Setup MLflow
    mlflow_params = _prepare_mlflow_params(config)
    setup_mlflow(
        mlflow_config=config.logging.mlflow,
        run_name=config.experiment.run_name,
        full_config_dict=config.model_dump(),
        base_params=mlflow_params
    )

    try:
        # Create make_env factory (encapsulates data loading)
        make_env = create_make_env(
            data_path=config.data.train_data_path,
            observation_config=config.observation,
            feature_engineering_config=config.feature_engineering,
            env_config=config.env,
            val_split=config.data.val_split,
            test_split=config.data.test_split,
        )

        # Create trainer (creates and wraps environments internally)
        trainer = Trainer(
            agent_config=config.agent,
            env_config=config.env,
            make_env=make_env,
            save_path=config.training.save_path,
            device=config.experiment.device,
        )

        # Train agent with only needed parameters
        trainer, train_metrics = train_agent(
            agent_name=config.agent.name,
            trainer=trainer,
            total_timesteps=config.training.total_timesteps,
            eval_freq=config.training.eval_freq,
            n_eval_episodes=config.training.n_eval_episodes,
            save_freq=config.training.save_freq,
            progress_bar=config.logging.console.progress_bar,
        )

        # Check if one_trade_mode is enabled
        one_trade_mode = config.env.environment_params.one_trade_mode

        if one_trade_mode:
            logger.warning("one_trade_mode is enabled - skipping final metrics evaluation")
            logger.warning("In one_trade_mode, each episode is a single trade, so training metrics are sufficient")
        else:
            # Evaluate on test set (only for multi-trade mode)
            test_metrics = evaluate_final_performance(trainer, make_env)

            # Run detailed backtest (only for multi-trade mode)
            backtest_results = run_backtest(trainer, make_env)

            # Log final test metrics to MLflow
            # Note: Training metrics are logged via callbacks (MLflowCallback, TradingMetricsCallback)
            # throughout training. These are final evaluation metrics computed AFTER training completes.
            if mlflow.active_run():
                # Add test/ prefix to distinguish from training metrics
                test_metrics_prefixed = {f"test/{k}": v for k, v in test_metrics.items()}
                mlflow.log_metrics(test_metrics_prefixed)
                mlflow.log_dict(backtest_results, "backtest_results.json")

        console.print("\n[bold green]Training Pipeline Complete![/bold green]")

    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()