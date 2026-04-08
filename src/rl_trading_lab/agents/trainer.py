"""
Slim RL Trainer - orchestrates training, evaluation, saving, and loading.

Takes pre-built components (wrapped environments, callbacks, algorithm class).
No direct dependency on MLflow, config objects, or callback construction.
"""

import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy as evaluate_policy_maskable
    MASKABLE_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    evaluate_policy_maskable = None
    MASKABLE_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    RL agent trainer with environment management and training orchestration.

    This is a slim orchestrator. Environment wrapping, callback creation,
    and config parsing are handled externally (see TrainerFactory for
    the project-specific wiring).

    Example:
        >>> trainer = Trainer(
        ...     algo_class=PPO,
        ...     env=wrapped_train_env,
        ...     eval_env=wrapped_eval_env,
        ...     hyperparams={"learning_rate": 3e-4, "n_steps": 2048},
        ...     save_path=Path("checkpoints"),
        ... )
        >>> trainer.train(total_timesteps=100_000)
    """

    def __init__(
        self,
        algo_class: Type[BaseAlgorithm],
        env: Union[DummyVecEnv, VecNormalize],
        eval_env: Optional[Union[DummyVecEnv, VecNormalize]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        save_path: Optional[Path] = None,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        vec_normalize_enabled: bool = True,
    ):
        self.env = env
        self.eval_env = eval_env
        self.save_path = Path(save_path) if save_path else Path("checkpoints")
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.vec_normalize_enabled = vec_normalize_enabled

        hyperparams = dict(hyperparams or {})

        # Extract and process policy_kwargs
        policy_kwargs = hyperparams.pop("policy_kwargs", None)
        if policy_kwargs and isinstance(policy_kwargs, dict):
            policy_kwargs = _process_policy_kwargs(policy_kwargs)

        # Resolve custom policy strings
        resolved_policy = hyperparams.pop("policy", policy)
        resolved_policy = _resolve_policy(resolved_policy)

        # Filter None values
        hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

        # Create agent
        self.agent = algo_class(
            policy=resolved_policy,
            env=env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            **hyperparams,
        )

        logger.info(f"Initialized {algo_class.__name__} trainer")

    def setup_logger(
        self,
        format_strings: Optional[list] = None,
        custom_output_formats: Optional[list] = None,
        tensorboard_log: Optional[str] = None,
    ) -> None:
        """
        Configure SB3 logger with optional custom outputs.

        Args:
            format_strings: Standard SB3 format names (e.g. ["stdout", "tensorboard"])
            custom_output_formats: Custom KVWriter instances (e.g. MLflowOutputFormat)
            tensorboard_log: Directory for tensorboard logs
        """
        if not format_strings and not custom_output_formats:
            return

        if format_strings and tensorboard_log:
            new_logger = configure(tensorboard_log, format_strings)
            if custom_output_formats:
                for fmt in custom_output_formats:
                    new_logger.output_formats.append(fmt)
            self.agent.set_logger(new_logger)
        elif custom_output_formats:
            new_logger = Logger(folder=None, output_formats=custom_output_formats)
            self.agent.set_logger(new_logger)

    def train(
        self,
        total_timesteps: int,
        callbacks: Optional[CallbackList] = None,
        progress_bar: bool = True,
        log_interval: int = 2,
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            total_timesteps: Total training timesteps
            callbacks: Pre-built callback list (use CallbackFactory)
            progress_bar: Show progress bar
            log_interval: How often to log metrics (in rollouts for on-policy)

        Returns:
            Training metadata
        """
        logger.info(f"Starting training for {total_timesteps} timesteps...")

        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=progress_bar,
            log_interval=log_interval,
        )

        self.save(self.save_path / "final_model")
        logger.info("Training completed")

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
        Evaluate the agent on an environment.

        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes...")

        if MASKABLE_AVAILABLE and isinstance(self.agent, MaskablePPO):
            episode_rewards, episode_lengths = evaluate_policy_maskable(
                self.agent, env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                return_episode_rewards=True,
            )
        else:
            episode_rewards, episode_lengths = evaluate_policy(
                self.agent, env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                return_episode_rewards=True,
            )

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "total_episodes": n_episodes,
        }

        logger.info(f"Evaluation results: {metrics}")
        return metrics

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ):
        """Predict action for given observation."""
        if action_masks is not None and MASKABLE_AVAILABLE and isinstance(self.agent, MaskablePPO):
            return self.agent.predict(
                observation, deterministic=deterministic, action_masks=action_masks,
            )
        return self.agent.predict(observation, deterministic=deterministic)

    def save(self, path: Union[str, Path]) -> None:
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]) -> None:
        """Load a saved model with CheckpointManager."""
        from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

        path = Path(path)
        checkpoint_manager = CheckpointManager()
        self.agent, self.env = checkpoint_manager.load_checkpoint(
            path,
            self.env.venv.envs[0] if hasattr(self.env, "venv") else self.env,
            verbose=1,
        )
        logger.info(f"Model loaded from {path}")


# --- Module-level helpers ---

def _process_policy_kwargs(policy_kwargs: dict) -> dict:
    """Convert string references in policy_kwargs to actual objects."""
    import torch.nn as nn

    processed = dict(policy_kwargs)

    if "activation_fn" in processed:
        name = processed["activation_fn"]
        if isinstance(name, str):
            processed["activation_fn"] = getattr(nn, name)

    if "features_extractor_class" in processed:
        cls_ref = processed["features_extractor_class"]
        if isinstance(cls_ref, str):
            module_path, class_name = cls_ref.rsplit(".", 1)
            module = importlib.import_module(module_path)
            processed["features_extractor_class"] = getattr(module, class_name)
            logger.info(f"Loaded custom feature extractor: {class_name}")

    return processed


def _resolve_policy(policy):
    """Resolve custom policy name to class if needed."""
    if not isinstance(policy, str):
        return policy

    try:
        from rl_trading_lab.models import TransformerActorCriticPolicy
        if policy == "TransformerPolicy" and TransformerActorCriticPolicy is not None:
            logger.info(f"Using custom policy: TransformerActorCriticPolicy")
            return TransformerActorCriticPolicy
    except ImportError:
        pass

    return policy
