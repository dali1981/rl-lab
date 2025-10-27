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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.trading_env import TradingEnv
from src.agents.sb3_agents import create_agent_from_config
from src.utils.data_loader import TradingDataLoader

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def setup_mlflow(cfg: DictConfig):
    """Setup MLflow tracking"""
    if not cfg.logging.mlflow.enabled:
        return

    # Set tracking URI
    mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)

    # Create or set experiment
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    # Start run
    mlflow.start_run(run_name=cfg.experiment.run_name)

    # Log parameters
    mlflow.log_params({
        "agent": cfg.agent.name,
        "environment.reward_type": cfg.env.environment_params.reward_type,
        "environment.initial_balance": cfg.env.environment_params.initial_balance,
        "environment.commission_rate": cfg.env.environment_params.commission_rate,
        "environment.lookback_window": cfg.env.environment_params.lookback_window,
        "training.total_timesteps": cfg.training.total_timesteps,
    })

    # Log hyperparameters
    if "hyperparameters" in cfg.agent:
        for key, value in cfg.agent.hyperparameters.items():
            if not isinstance(value, (dict, list)):
                mlflow.log_param(f"agent.{key}", value)

    # Log full config as artifact for reproducibility
    # This saves the complete resolved Hydra config (all defaults + overrides)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_dict(config_dict, "config.yaml")

    console.print(f"[green]✓[/green] MLflow tracking enabled: {cfg.logging.mlflow.tracking_uri}")
    console.print(f"[green]✓[/green] Full config logged to MLflow")


def create_environments(cfg: DictConfig):
    """Create training and evaluation environments"""
    console.print("\n[bold]Loading Data...[/bold]")

    # Initialize data loader
    data_loader = TradingDataLoader(
        data_path=cfg.data.train_data_path,
        features_config=OmegaConf.to_container(cfg.features),
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
    )

    # Load and prepare data
    train_df, val_df, test_df, feature_names = data_loader.load_and_prepare()

    console.print(f"[green]✓[/green] Loaded data: "
                 f"Train={len(train_df)} bars, Val={len(val_df)} bars, Test={len(test_df)} bars")
    console.print(f"[green]✓[/green] Features: {len(feature_names)} selected")

    # Extract environment parameters from config
    env_params = cfg.env.environment_params

    # Create training environment with randomization enabled
    train_env = TradingEnv(
        df=train_df,
        lookback_window=env_params.lookback_window,
        initial_balance=env_params.initial_balance,
        commission_rate=env_params.commission_rate,
        slippage_rate=env_params.slippage_rate,
        reward_type=env_params.reward_type,
        discrete_actions=env_params.discrete_actions,
        max_position_pct=env_params.max_position_pct,
        features_to_use=feature_names,
        randomize_start=env_params.randomize_start,
        min_episode_length=env_params.min_episode_length,
        hold_closes_position=env_params.hold_closes_position,
    )

    # Create evaluation environment (no randomization for consistency)
    eval_env = TradingEnv(
        df=val_df,
        lookback_window=env_params.lookback_window,
        initial_balance=env_params.initial_balance,
        commission_rate=env_params.commission_rate,
        slippage_rate=env_params.slippage_rate,
        reward_type=env_params.reward_type,
        discrete_actions=env_params.discrete_actions,
        max_position_pct=env_params.max_position_pct,
        features_to_use=feature_names,
        randomize_start=False,  # Disable for consistent evaluation
        min_episode_length=env_params.min_episode_length,
        hold_closes_position=env_params.hold_closes_position,
    )

    # Create test environment (no randomization for consistent evaluation)
    test_env = TradingEnv(
        df=test_df,
        lookback_window=env_params.lookback_window,
        initial_balance=env_params.initial_balance,
        commission_rate=env_params.commission_rate,
        slippage_rate=env_params.slippage_rate,
        reward_type=env_params.reward_type,
        discrete_actions=env_params.discrete_actions,
        max_position_pct=env_params.max_position_pct,
        features_to_use=feature_names,
        randomize_start=False,  # Disable for consistent evaluation
        min_episode_length=env_params.min_episode_length,
        hold_closes_position=env_params.hold_closes_position,
    )

    return train_env, eval_env, test_env


def wrap_test_env_for_evaluation(test_env, agent):
    """
    Wrap test environment to match training setup.

    CRITICAL: Agent was trained with VecNormalize, so it expects normalized observations.
    We must wrap the test environment the same way and copy the normalization statistics
    from training to ensure consistent observation scaling.

    Args:
        test_env: Raw TradingEnv instance
        agent: Trained agent wrapper (has agent.env which is VecNormalize)

    Returns:
        Properly wrapped test environment
    """
    console.print("[yellow]Wrapping test environment with VecNormalize...[/yellow]")

    # Wrap in DummyVecEnv to make it vectorized
    test_env_func = lambda e=test_env: e
    test_vec_env = DummyVecEnv([test_env_func])

    # Wrap with VecNormalize (same as training, but training=False)
    test_vec_env = VecNormalize(
        test_vec_env,
        norm_obs=True,       # Normalize observations (CRITICAL)
        norm_reward=False,   # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False,      # Don't update running statistics
    )

    # CRITICAL: Copy normalization statistics from training environment
    # Without this, the test env would use different normalization!
    if hasattr(agent.env, 'obs_rms'):
        test_vec_env.obs_rms = agent.env.obs_rms
        console.print("[green]✓[/green] Copied observation normalization stats from training")
    else:
        console.print("[red]Warning: Could not copy normalization stats![/red]")

    if hasattr(agent.env, 'ret_rms'):
        test_vec_env.ret_rms = agent.env.ret_rms

    return test_vec_env


def train_agent(cfg: DictConfig, train_env, eval_env):
    """Train the RL agent"""
    console.print(f"\n[bold]Training {cfg.agent.name} Agent...[/bold]")

    # Create agent
    agent = create_agent_from_config(
        config=OmegaConf.to_container(cfg),
        env=train_env,
        eval_env=eval_env,
    )

    # Train
    metrics = agent.train(
        total_timesteps=cfg.training.total_timesteps,
        eval_freq=cfg.training.eval_freq,
        n_eval_episodes=cfg.training.n_eval_episodes,
        save_freq=cfg.training.save_freq,
        progress_bar=cfg.logging.console.progress_bar,
    )

    console.print(f"[green]✓[/green] Training completed")

    return agent, metrics


def evaluate_final_performance(agent, test_env, n_episodes=10):
    """Evaluate final model performance"""
    console.print("\n[bold]Evaluating Final Performance...[/bold]")

    # Wrap test environment to match training setup (critical for correct evaluation)
    test_env_wrapped = wrap_test_env_for_evaluation(test_env, agent)

    # Evaluate using wrapped environment
    metrics = agent.evaluate(
        env=test_env_wrapped,
        n_episodes=n_episodes,
        deterministic=True,
    )

    # Add debugging: print individual episode rewards to investigate std_reward=0
    console.print(f"[yellow]Debug: Episode rewards stats[/yellow]")
    console.print(f"  Mean: {metrics.get('mean_reward', 'N/A')}")
    console.print(f"  Std: {metrics.get('std_reward', 'N/A')}")

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


def run_backtest(agent, test_env):
    """Run full backtest and collect detailed metrics"""
    console.print("\n[bold]Running Backtest...[/bold]")

    # Wrap test environment to match training setup
    test_env_wrapped = wrap_test_env_for_evaluation(test_env, agent)

    obs = test_env_wrapped.reset()
    done = False
    truncated = False

    actions = []
    rewards = []
    positions = []
    balances = []

    # Track actual returns (not Sharpe rewards)
    step_returns = []

    step_count = 0
    while not done and not truncated:
        # Get action from agent
        action, _ = agent.predict(obs, deterministic=True)

        # Step environment
        obs, reward, done, truncated, info = test_env_wrapped.step(action)

        # Extract info from vectorized environment
        if isinstance(info, list):
            info = info[0]
        if isinstance(done, np.ndarray):
            done = done[0]
        if isinstance(truncated, np.ndarray):
            truncated = truncated[0]

        # Collect data
        actions.append(action)
        rewards.append(reward)
        positions.append(info.get('position', 0))

        # Get portfolio value from unwrapped environment
        # Since we wrapped with VecNormalize, need to access the base env
        unwrapped_env = test_env_wrapped.venv.envs[0]
        portfolio_value = unwrapped_env._get_portfolio_value()
        balances.append(portfolio_value)

        # Calculate step return for Sharpe calculation
        if len(balances) > 1:
            step_return = (balances[-1] - balances[-2]) / balances[-2]
            step_returns.append(step_return)

        step_count += 1

    # Calculate final metrics
    initial_balance = test_env_wrapped.venv.envs[0].initial_balance
    final_return = (balances[-1] - initial_balance) / initial_balance

    # FIXED: Calculate Sharpe from actual returns, not from Sharpe rewards!
    # The rewards are Sharpe approximations, we need to use actual portfolio returns
    if len(step_returns) > 0:
        returns_array = np.array(step_returns)
        sharpe = returns_array.mean() / (returns_array.std() + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Trading frequency analysis
    total_trades = len([a for a in actions if a != 0])
    trade_frequency = total_trades / len(actions) if len(actions) > 0 else 0

    console.print(f"[green]✓[/green] Backtest completed")
    console.print(f"  Steps: {step_count}")
    console.print(f"  Final Return: {final_return:.2%}")
    console.print(f"  Sharpe Ratio (from returns): {sharpe:.2f}")
    console.print(f"  Total Trades: {total_trades}")
    console.print(f"  Trade Frequency: {trade_frequency:.1%}")

    # Debugging: Show reward statistics
    console.print(f"[yellow]Debug: Reward statistics[/yellow]")
    console.print(f"  Mean reward: {np.mean(rewards):.4f}")
    console.print(f"  Std reward: {np.std(rewards):.4f}")
    console.print(f"  Min/Max reward: {np.min(rewards):.4f} / {np.max(rewards):.4f}")

    # Log backtest metrics to MLflow
    # Note: Training metrics are logged via callbacks during training.
    # These backtest metrics are computed AFTER training completes on the test set,
    # so they're logged here as a final summary.
    if mlflow.active_run():
        mlflow.log_metrics({
            "backtest/final_return": final_return,
            "backtest/sharpe_ratio": sharpe,
            "backtest/num_trades": total_trades,
            "backtest/trade_frequency": trade_frequency,
        })

    return {
        "final_return": final_return,
        "sharpe_ratio": sharpe,
        "actions": actions,
        "positions": positions,
        "balances": balances,
        "step_returns": step_returns,
        "trade_frequency": trade_frequency,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training pipeline"""
    console.print("[bold blue]RL Trading Lab[/bold blue]")
    console.print(f"Agent: [cyan]{cfg.agent.name}[/cyan]")
    console.print(f"Environment: [cyan]{cfg.env.environment_params.reward_type}[/cyan] reward")
    console.print(f"Device: [cyan]{cfg.experiment.device}[/cyan]\n")

    # Setup MLflow
    setup_mlflow(cfg)

    try:
        # Create environments
        train_env, eval_env, test_env = create_environments(cfg)

        # Train agent
        agent, train_metrics = train_agent(cfg, train_env, eval_env)

        # Evaluate on test set
        test_metrics = evaluate_final_performance(agent, test_env)

        # Run detailed backtest
        backtest_results = run_backtest(agent, test_env)

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