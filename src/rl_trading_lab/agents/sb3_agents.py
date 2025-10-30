"""
Wrapper for Stable-Baselines3 agents.
Provides a unified interface for different RL algorithms.
"""

import os
from typing import Dict, Any, Optional, Type, Union, Callable
from pathlib import Path
import logging
import importlib

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

from rl_trading_lab.config import RootConfig
from rl_trading_lab.config.agent import AgentConfig
from rl_trading_lab.config.env import EnvConfig
from rl_trading_lab.utils.mlflow_logger import MLflowOutputFormat
from rl_trading_lab.utils.callbacks import TradingMetricsCallback
from rl_trading_lab.utils.custom_callbacks import CheckpointManagerCallback, BestModelCallback
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

# Import custom policies
try:
    from rl_trading_lab.models import TransformerActorCriticPolicy
except ImportError:
    logger.warning("Could not import TransformerActorCriticPolicy. Transformer policy will not be available.")
    TransformerActorCriticPolicy = None

logger = logging.getLogger(__name__)


# Algorithm mapping
ALGORITHMS = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
}

# Custom policy mapping
CUSTOM_POLICIES = {
    "TransformerPolicy": TransformerActorCriticPolicy,
}


class Trainer:
    """
    RL agent trainer with environment management and training orchestration.

    Responsibilities:
    - Creates and wraps training/evaluation environments
    - Configures and trains RL agents
    - Handles checkpointing and evaluation
    - Integrates with MLflow for experiment tracking

    Features:
    - Unified interface for different RL algorithms (PPO, A2C, DQN, SAC)
    - Automatic environment wrapping (Monitor, DummyVecEnv, optional VecNormalize)
    - Built-in evaluation and checkpointing
    - MLflow and TensorBoard integration
    - Custom callbacks for trading metrics
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        env_config: EnvConfig,
        make_env: Callable[[str], gym.Env],
        save_path: Optional[str] = None,
        device: str = "auto",
    ):
        """
        Initialize the trainer.

        Args:
            agent_config: Agent configuration (algorithm, hyperparameters, etc.)
            env_config: Environment configuration (vec_normalize, trading params, etc.)
            make_env: Factory function that creates environments.
                      Takes mode ('train', 'eval', 'test') and returns gym.Env
            save_path: Path to save models and checkpoints
            device: Device to use (cpu, cuda, auto)

        Example:
            ```python
            def make_env(mode: str) -> gym.Env:
                df = load_data_for_mode(mode)
                return TradingEnv(df=df, ...)

            trainer = Trainer(
                agent_config=config.agent,
                env_config=config.env,
                make_env=make_env,
            )
            ```
        """
        self.config = agent_config
        self.env_config = env_config
        self.make_env = make_env
        self.save_path = Path(save_path) if save_path else Path("checkpoints")
        self.device = device

        # Create training environment
        logger.info("Creating training environment...")
        train_env = self.make_env('train')
        self.env = self._wrap_environment(train_env, is_eval=False)

        # Create evaluation environment
        logger.info("Creating evaluation environment...")
        logger.info(f"Env config one_trade_mode: {self.env_config.environment_params.one_trade_mode}")
        eval_env = self.make_env('eval')
        self.eval_env = self._wrap_environment(eval_env, is_eval=True)

        # Store VecNormalize state
        vec_normalize_config = self.env_config.vec_normalize
        self.vec_normalize_enabled = vec_normalize_config.enabled
        self.vec_normalize_config = vec_normalize_config

        # Get algorithm class
        algo_name = agent_config.algorithm.split(".")[-1]
        if algo_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        self.algo_class = ALGORITHMS[algo_name]

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Initialize agent
        self.agent = self._create_agent()

        # Setup custom logger for MLflow and TensorBoard integration
        self._setup_logger()

        if self.vec_normalize_enabled:
            logger.info(f"Initialized {algo_name} trainer with Monitor + VecNormalize wrappers")
        else:
            logger.info(f"Initialized {algo_name} trainer with Monitor wrapper (VecNormalize disabled)")

    def _wrap_environment(self, env: gym.Env, is_eval: bool = False) -> Union[DummyVecEnv, VecNormalize]:
        """
        Wrap environment with Monitor, DummyVecEnv, and optionally VecNormalize.

        Args:
            env: Raw environment to wrap
            is_eval: Whether this is an evaluation environment

        Returns:
            Wrapped environment ready for training/evaluation
        """
        # Wrap with Monitor for episode statistics
        monitored_env = Monitor(env)
        env_func = lambda e=monitored_env: e
        vec_env = DummyVecEnv([env_func])

        # Conditionally wrap with VecNormalize
        vec_normalize_config = self.env_config.vec_normalize
        if vec_normalize_config.enabled:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=vec_normalize_config.norm_obs,
                norm_reward=False if is_eval else vec_normalize_config.norm_reward,
                clip_obs=vec_normalize_config.clip_obs,
                clip_reward=vec_normalize_config.clip_reward if not is_eval else 10.0,
                gamma=self.config.hyperparameters.gamma,
                training=not is_eval,  # Disable updates during eval
            )

        return vec_env

    def _create_agent(self) -> BaseAlgorithm:
        """Create the SB3 agent"""
        # Get hyperparameters as dict
        # Exclude None values to avoid passing them to SB3 (which causes issues with DQN)
        hyperparams = self.config.hyperparameters.model_dump(exclude_none=True)

        # Handle policy kwargs
        policy_kwargs = hyperparams.pop("policy_kwargs", None)
        if policy_kwargs and isinstance(policy_kwargs, dict):
            # Convert activation function string to actual function
            if "activation_fn" in policy_kwargs:
                import torch.nn as nn
                activation_name = policy_kwargs["activation_fn"]
                policy_kwargs["activation_fn"] = getattr(nn, activation_name)

            # Handle features_extractor_class if it's a string reference
            if "features_extractor_class" in policy_kwargs:
                extractor_class = policy_kwargs["features_extractor_class"]
                if isinstance(extractor_class, str):
                    # Convert string reference to actual class
                    # e.g., "rl_trading_lab.models.TransformerFeatureExtractor"
                    try:
                        module_path, class_name = extractor_class.rsplit(".", 1)
                        module = importlib.import_module(module_path)
                        policy_kwargs["features_extractor_class"] = getattr(module, class_name)
                        logger.info(f"Loaded custom feature extractor: {class_name}")
                    except (ValueError, ImportError, AttributeError) as e:
                        logger.error(f"Failed to import features_extractor_class '{extractor_class}': {e}")
                        raise

        # Get policy (can be string like "MlpPolicy" or "TransformerPolicy")
        policy = hyperparams.pop("policy", "MlpPolicy")

        # Map custom policy strings to classes
        if isinstance(policy, str) and policy in CUSTOM_POLICIES:
            policy = CUSTOM_POLICIES[policy]
            logger.info(f"Using custom policy: {policy.__name__}")

        # Create agent
        agent = self.algo_class(
            policy=policy,
            env=self.env,
            policy_kwargs=policy_kwargs,
            device=self.device,
            verbose=self.config.verbose,
            tensorboard_log=self.config.tensorboard_log,
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
        tensorboard_log = self.config.tensorboard_log

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
        # Use BestModelCallback to save VecNormalize stats with best model
        if self.eval_env and eval_freq:
            eval_callback = BestModelCallback(
                self.eval_env,
                best_model_save_path=str(self.save_path / "best_model"),
                log_path=str(self.save_path / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=self.config.verbose,
                metadata={'agent_config': self.config.name},
            )
            callback_list.append(eval_callback)

        # Add checkpoint callback with metadata
        # Use CheckpointManagerCallback to save metadata with each checkpoint
        if save_freq:
            checkpoint_callback = CheckpointManagerCallback(
                save_freq=save_freq,
                save_path=str(self.save_path / "checkpoints"),
                name_prefix="rl_model",
                save_replay_buffer=True,
                save_vecnormalize=True,
                verbose=self.config.verbose,
                metadata={'agent_config': self.config.name},
            )
            callback_list.append(checkpoint_callback)

        # Add custom callbacks
        if callbacks:
            callback_list.extend(callbacks)

        # Add trading metrics callback (only in multi-trade mode)
        # In one_trade_mode, each episode is a single trade, so win/loss tracking
        # is redundant with episode rewards
        one_trade_mode = self.env_config.environment_params.one_trade_mode
        if not one_trade_mode:
            trading_callback = TradingMetricsCallback(verbose=self.config.verbose)
            callback_list.append(trading_callback)
            logger.info("TradingMetricsCallback enabled (multi-trade mode)")
        else:
            logger.info("TradingMetricsCallback disabled (one_trade_mode=True)")

        # Combine callbacks
        combined_callback = CallbackList(callback_list) if callback_list else None

        # Train
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            progress_bar=progress_bar,
            log_interval=2,  # Log every 2 rollouts/updates
            # Recommended values:
            # - log_interval=1: Log every rollout (PPO/A2C collect data in rollouts)
            # - log_interval=2: Log every 2 rollouts (good balance, ~4k steps with n_steps=2048)
            # - log_interval=4: Log every 4 rollouts
            # - log_interval=100: Log every 100 environment steps (for off-policy like DQN/SAC)
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
        """Load a saved model with CheckpointManager"""
        path = Path(path)

        # Use CheckpointManager for robust loading
        checkpoint_manager = CheckpointManager()

        # Load model and VecNormalize
        # Note: This replaces self.env with properly configured VecNormalize
        self.agent, self.env = checkpoint_manager.load_checkpoint(
            path,
            self.env.venv.envs[0] if hasattr(self.env, 'venv') else self.env,
            verbose=1
        )

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


