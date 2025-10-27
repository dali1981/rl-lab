"""
Wrapper for Stable-Baselines3 agents.
Provides a unified interface for different RL algorithms.
"""

import os
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path
import logging

import gymnasium as gym
import numpy as np
import mlflow
from stable_baselines3 import PPO, A2C, DQN, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, configure

from src.utils.mlflow_logger import MLflowOutputFormat
from src.utils.callbacks import MLflowCallback, TradingMetricsCallback

logger = logging.getLogger(__name__)


# Algorithm mapping
ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
}


class TradingAgentWrapper:
    """
    Wrapper for SB3 agents with trading-specific features.

    Features:
    - Unified interface for different algorithms
    - Built-in evaluation and checkpointing
    - MLflow integration ready
    - Custom callbacks for trading metrics
    """

    def __init__(
        self,
        agent_config: Dict[str, Any],
        env: gym.Env,
        eval_env: Optional[gym.Env] = None,
        save_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize agent wrapper.

        Args:
            agent_config: Agent configuration dict from Hydra
            env: Training environment
            eval_env: Evaluation environment (optional)
            save_path: Path to save models
            device: Device to use (cpu, cuda, auto)
        """
        self.config = agent_config
        self.save_path = Path(save_path) if save_path else Path("checkpoints")
        self.device = device

        # Wrap environments with Monitor -> DummyVecEnv -> VecNormalize for better training stability
        # Monitor tracks episode statistics (required for proper callback logging)
        monitored_env = Monitor(env)
        train_env_func = lambda e=monitored_env: e  # Proper closure
        self.env = DummyVecEnv([train_env_func])

        # Wrap with VecNormalize to normalize observations and rewards
        self.env = VecNormalize(
            self.env,
            norm_obs=True,      # Normalize observations
            norm_reward=True,   # Normalize rewards (critical for stability)
            clip_obs=10.0,      # Clip observations
            clip_reward=10.0,   # Clip rewards
            gamma=agent_config.get("hyperparameters", {}).get("gamma", 0.99),
        )

        if eval_env is not None:
            # Wrap eval environment similarly, but don't normalize rewards during eval
            monitored_eval_env = Monitor(eval_env)
            eval_env_func = lambda e=monitored_eval_env: e  # Proper closure
            self.eval_env = DummyVecEnv([eval_env_func])
            self.eval_env = VecNormalize(
                self.eval_env,
                norm_obs=True,
                norm_reward=False,  # Don't normalize rewards during evaluation
                clip_obs=10.0,
                training=False,     # Disable updates to running stats during eval
            )
        else:
            self.eval_env = None

        # Get algorithm class
        algo_name = agent_config.get("algorithm", "PPO").split(".")[-1]
        if algo_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        self.algo_class = ALGORITHMS[algo_name]

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Initialize agent
        self.agent = self._create_agent()

        # Setup custom logger for MLflow and TensorBoard integration
        self._setup_logger()

        logger.info(f"Initialized {algo_name} agent with Monitor + VecNormalize wrappers")

    def _create_agent(self) -> BaseAlgorithm:
        """Create the SB3 agent"""
        # Extract hyperparameters
        hyperparams = self.config.get("hyperparameters", {})

        # Handle policy kwargs
        policy_kwargs = hyperparams.pop("policy_kwargs", None)
        if policy_kwargs and isinstance(policy_kwargs, dict):
            # Convert activation function string to actual function
            if "activation_fn" in policy_kwargs:
                import torch.nn as nn
                activation_name = policy_kwargs["activation_fn"]
                policy_kwargs["activation_fn"] = getattr(nn, activation_name)

        # Create agent
        agent = self.algo_class(
            policy=hyperparams.pop("policy", "MlpPolicy"),
            env=self.env,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.config.get("verbose", 1),
            tensorboard_log=self.config.get("tensorboard_log", None),
            **hyperparams
        )

        return agent

    def _setup_logger(self):
        """
        Configure SB3 logger with MLflow and TensorBoard outputs.

        Sets up automatic logging to both MLflow (if active run) and TensorBoard.
        All SB3 metrics (rollout/, train/, eval/) will be logged automatically.
        """
        # Get tensorboard log directory from agent config
        tensorboard_log = self.config.get("tensorboard_log", None)

        # Create custom output formats list
        custom_output_formats = []

        # Add MLflow output if there's an active run
        if mlflow.active_run():
            custom_output_formats.append(MLflowOutputFormat())
            logger.info("MLflow logging enabled")

        # Configure logger
        if tensorboard_log:
            # Configure with standard format strings (stdout, tensorboard)
            format_strings = ["stdout", "tensorboard"]
            new_logger = configure(tensorboard_log, format_strings)

            # Manually append custom output formats to the logger
            for output_format in custom_output_formats:
                new_logger.output_formats.append(output_format)

            self.agent.set_logger(new_logger)
            logger.info(f"TensorBoard logging enabled: {tensorboard_log}")
        elif custom_output_formats:
            # If only MLflow (no TensorBoard), create logger with custom formats
            new_logger = Logger(folder=None, output_formats=custom_output_formats)
            self.agent.set_logger(new_logger)

        # If no logging configured, agent will use default logger

    def train(
        self,
        total_timesteps: int,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        save_freq: Optional[int] = None,
        callbacks: Optional[list] = None,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_freq: Checkpoint save frequency
            callbacks: Additional callbacks
            progress_bar: Show progress bar

        Returns:
            Training history/metrics
        """
        callback_list = []

        # Add evaluation callback if eval environment provided
        if self.eval_env and eval_freq:
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=str(self.save_path / "best_model"),
                log_path=str(self.save_path / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
            callback_list.append(eval_callback)

        # Add checkpoint callback
        if save_freq:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.save_path / "checkpoints"),
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
            callback_list.append(checkpoint_callback)

        # Add custom callbacks
        if callbacks:
            callback_list.extend(callbacks)

        # Add MLflow logging callback
        mlflow_callback = MLflowCallback(
            log_freq=max(eval_freq // 10, 100) if eval_freq else 1000,
            verbose=self.config.get("verbose", 0),
        )
        callback_list.append(mlflow_callback)

        # Add trading metrics callback
        trading_callback = TradingMetricsCallback(verbose=self.config.get("verbose", 0))
        callback_list.append(trading_callback)

        # Combine callbacks
        combined_callback = CallbackList(callback_list) if callback_list else None

        # Train
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            progress_bar=progress_bar,
        )

        # Save final model
        self.save(self.save_path / "final_model")

        logger.info("Training completed")

        # Return metrics
        return {
            "total_timesteps": total_timesteps,
            "final_model_path": str(self.save_path / "final_model"),
        }

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the agent.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes...")

        # Evaluate
        episode_rewards, episode_lengths = evaluate_policy(
            self.agent,
            env,
            n_eval_episodes=n_episodes,
            deterministic=deterministic,
            return_episode_rewards=True,
        )

        # Debugging: Print individual episode rewards to investigate std_reward=0
        logger.info(f"DEBUG - Individual episode rewards: {episode_rewards}")
        logger.info(f"DEBUG - Individual episode lengths: {episode_lengths}")
        logger.info(f"DEBUG - Reward unique values: {np.unique(episode_rewards)}")

        # Calculate metrics
        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "total_episodes": n_episodes,
        }

        # Additional debugging metrics
        metrics["min_reward"] = float(np.min(episode_rewards))
        metrics["max_reward"] = float(np.max(episode_rewards))

        # Get trading-specific metrics from environment
        if hasattr(env, "get_trading_metrics"):
            trading_metrics = env.get_trading_metrics()
            metrics.update(trading_metrics)

        logger.info(f"Evaluation results: {metrics}")

        return metrics

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ):
        """
        Predict action for given observation.

        Args:
            observation: Environment observation
            deterministic: Use deterministic action

        Returns:
            action, state (if recurrent policy)
        """
        return self.agent.predict(observation, deterministic=deterministic)

    def save(self, path: Union[str, Path]):
        """Save the model"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load a saved model"""
        path = Path(path)
        self.agent = self.algo_class.load(str(path), env=self.env)
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        env: gym.Env,
        agent_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Load a pretrained model.

        Args:
            model_path: Path to saved model
            env: Environment
            agent_config: Optional config override

        Returns:
            TradingAgentWrapper instance
        """
        # Determine algorithm from file
        import pickle

        model_path = Path(model_path)

        # Create minimal config if not provided
        if agent_config is None:
            # Try to infer algorithm from saved model
            agent_config = {"algorithm": "PPO", "name": "PPO"}

        wrapper = cls(agent_config, env)
        wrapper.load(model_path)

        return wrapper


def create_agent_from_config(
    config: Dict[str, Any],
    env: gym.Env,
    eval_env: Optional[gym.Env] = None,
) -> TradingAgentWrapper:
    """
    Factory function to create agent from Hydra config.

    Args:
        config: Full Hydra configuration
        env: Training environment
        eval_env: Evaluation environment

    Returns:
        TradingAgentWrapper instance
    """
    agent_config = config.get("agent", {})
    experiment_config = config.get("experiment", {})
    training_config = config.get("training", {})

    # Create wrapper
    wrapper = TradingAgentWrapper(
        agent_config=agent_config,
        env=env,
        eval_env=eval_env,
        save_path=training_config.get("save_path", "checkpoints"),
        device=experiment_config.get("device", "auto"),
    )

    return wrapper