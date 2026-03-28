"""
Project-specific factory that wires Trainer from config objects.

This is the glue between the project's Pydantic config system and
the generic Trainer. Other projects would write their own factory
(or construct Trainer directly).
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from rl_trading_lab.agents.env_wrapper import EnvWrapperBuilder
from rl_trading_lab.agents.callback_factory import CallbackFactory
from rl_trading_lab.agents.sb3_agents import Trainer
from rl_trading_lab.config.agent import AgentConfig
from rl_trading_lab.config.env import EnvConfig

logger = logging.getLogger(__name__)

try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    MASKABLE_AVAILABLE = False

ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
}
if MASKABLE_AVAILABLE:
    ALGORITHMS["MaskablePPO"] = MaskablePPO


class TrainerFactory:
    """
    Creates a fully-wired Trainer from project config objects.

    This preserves backward compatibility with the old Trainer constructor
    signature while delegating to the new composable pieces.

    Example:
        >>> trainer = TrainerFactory.from_config(
        ...     agent_config=config.agent,
        ...     env_config=config.env,
        ...     make_env=make_env,
        ...     save_path=config.training.save_path,
        ... )
        >>> callbacks = TrainerFactory.create_callbacks(
        ...     trainer=trainer,
        ...     agent_config=config.agent,
        ...     env_config=config.env,
        ...     save_path=Path(config.training.save_path),
        ... )
        >>> trainer.train(total_timesteps=100_000, callbacks=callbacks)
    """

    @staticmethod
    def from_config(
        agent_config: AgentConfig,
        env_config: EnvConfig,
        make_env: Callable[[str], gym.Env],
        save_path: Optional[str] = None,
        device: str = "auto",
        observation_config=None,
        feature_engineering_config=None,
    ) -> Trainer:
        """
        Create a Trainer from project config objects.

        Args:
            agent_config: Agent configuration (algorithm, hyperparameters)
            env_config: Environment configuration (vec_normalize, trading params)
            make_env: Factory function: mode ('train'/'eval'/'test') -> gym.Env
            save_path: Path to save models and checkpoints
            device: Device to use (cpu, cuda, auto)
            observation_config: Observation config (stored as metadata)
            feature_engineering_config: Feature engineering config (stored as metadata)

        Returns:
            Fully configured Trainer instance
        """
        # Resolve algorithm
        algo_name = agent_config.algorithm.split(".")[-1]
        if algo_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        algo_class = ALGORITHMS[algo_name]

        # Build environment wrapper
        vec_cfg = env_config.vec_normalize
        wrapper_builder = EnvWrapperBuilder(
            vec_normalize_enabled=vec_cfg.enabled,
            norm_obs=vec_cfg.norm_obs,
            norm_reward=vec_cfg.norm_reward,
            clip_obs=vec_cfg.clip_obs,
            clip_reward=vec_cfg.clip_reward,
        )

        gamma = agent_config.hyperparameters.gamma

        # Create and wrap environments
        logger.info("Creating training environment...")
        train_env = wrapper_builder.build(make_env("train"), is_eval=False, gamma=gamma)

        logger.info("Creating evaluation environment...")
        eval_env = wrapper_builder.build(make_env("eval"), is_eval=True, gamma=gamma)

        # Build hyperparams dict
        hyperparams = agent_config.hyperparameters.model_dump(exclude_none=True)

        # Create trainer
        resolved_path = Path(save_path) if save_path else Path("checkpoints")
        trainer = Trainer(
            algo_class=algo_class,
            env=train_env,
            eval_env=eval_env,
            hyperparams=hyperparams,
            save_path=resolved_path,
            device=device,
            verbose=agent_config.verbose,
            tensorboard_log=agent_config.tensorboard_log,
            vec_normalize_enabled=vec_cfg.enabled,
        )

        # Setup logging (MLflow + TensorBoard)
        cb_factory = CallbackFactory()
        format_strings, custom_formats = cb_factory.create_logging_setup(
            tensorboard_log=agent_config.tensorboard_log,
        )
        trainer.setup_logger(
            format_strings=format_strings,
            custom_output_formats=custom_formats,
            tensorboard_log=agent_config.tensorboard_log,
        )

        # Store config references for metadata (used by callbacks)
        trainer._observation_config = observation_config
        trainer._feature_engineering_config = feature_engineering_config
        trainer._env_config = env_config
        trainer._agent_config = agent_config

        return trainer

    @staticmethod
    def create_callbacks(
        trainer: Trainer,
        agent_config: AgentConfig,
        env_config: EnvConfig,
        save_path: Path,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        save_freq: Optional[int] = None,
    ):
        """
        Create training callbacks from config.

        Args:
            trainer: The Trainer instance (for eval_env reference)
            agent_config: Agent config (for verbose, name)
            env_config: Env config (for one_trade_mode)
            save_path: Base save path for models/checkpoints
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            save_freq: Checkpoint save frequency

        Returns:
            CallbackList or None
        """
        # Build metadata for checkpoints
        metadata = {"agent_config": agent_config.name}
        if hasattr(trainer, "_observation_config") and trainer._observation_config:
            metadata["observation_config"] = trainer._observation_config.model_dump()
        if hasattr(trainer, "_feature_engineering_config") and trainer._feature_engineering_config:
            metadata["feature_engineering_config"] = trainer._feature_engineering_config.model_dump()
        if hasattr(trainer, "_env_config") and trainer._env_config:
            metadata["env_config"] = trainer._env_config.model_dump()

        factory = CallbackFactory()
        return factory.create_all(
            eval_env=trainer.eval_env,
            save_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            save_freq=save_freq,
            one_trade_mode=env_config.environment_params.one_trade_mode,
            verbose=agent_config.verbose,
            metadata=metadata,
        )
