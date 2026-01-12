"""
AgentService - Application service for RL agent management.

This service handles agent creation, configuration, and loading.
It abstracts the complexity of working with different RL algorithms
and their various configurations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)

# Try to import MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    MASKABLE_AVAILABLE = True
except ImportError:
    MaskablePPO = None
    MASKABLE_AVAILABLE = False
    logger.debug("sb3-contrib not installed, MaskablePPO unavailable")


# Algorithm registry
ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
    "DQN": DQN,
    "SAC": SAC,
}

if MASKABLE_AVAILABLE:
    ALGORITHMS["MaskablePPO"] = MaskablePPO


class AgentService:
    """
    Application service for RL agent management.

    Responsibilities:
    - Creating agents with appropriate configurations
    - Loading trained agents from checkpoints
    - Wrapping environments for SB3 compatibility
    - Managing VecNormalize statistics

    Example:
        >>> agent_service = AgentService()
        >>>
        >>> # Create a new agent
        >>> agent, vec_env = agent_service.create_agent(
        ...     algorithm="PPO",
        ...     env=train_env,
        ...     hyperparameters={"learning_rate": 3e-4},
        ... )
        >>>
        >>> # Load a trained agent
        >>> agent, vec_env = agent_service.load_agent(
        ...     model_path=Path("checkpoints/best_model"),
        ...     env=test_env,
        ... )
    """

    def __init__(
        self,
        vec_normalize_enabled: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        """
        Initialize the agent service.

        Args:
            vec_normalize_enabled: Whether to use VecNormalize wrapper
            norm_obs: Whether to normalize observations
            norm_reward: Whether to normalize rewards
            clip_obs: Clipping value for observations
            clip_reward: Clipping value for rewards
        """
        self._vec_normalize_enabled = vec_normalize_enabled
        self._norm_obs = norm_obs
        self._norm_reward = norm_reward
        self._clip_obs = clip_obs
        self._clip_reward = clip_reward

    def create_agent(
        self,
        algorithm: str,
        env: gym.Env,
        hyperparameters: Optional[Dict[str, Any]] = None,
        policy: str = "MlpPolicy",
        device: str = "auto",
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
    ) -> tuple:
        """
        Create a new RL agent.

        Args:
            algorithm: Algorithm name (PPO, A2C, DQN, SAC, MaskablePPO)
            env: Gymnasium environment
            hyperparameters: Algorithm hyperparameters
            policy: Policy type (MlpPolicy, CnnPolicy, etc.)
            device: Device to use (cpu, cuda, auto)
            tensorboard_log: Directory for TensorBoard logs
            verbose: Verbosity level

        Returns:
            Tuple of (agent, wrapped_environment)
        """
        # Get algorithm class
        algo_name = algorithm.split(".")[-1] if "." in algorithm else algorithm
        if algo_name not in ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {algo_name}. "
                f"Available: {list(ALGORITHMS.keys())}"
            )

        algo_class = ALGORITHMS[algo_name]

        # Wrap environment
        vec_env = self._wrap_environment(env, is_eval=False)

        # Prepare hyperparameters
        hyperparams = hyperparameters.copy() if hyperparameters else {}

        # Handle policy_kwargs
        policy_kwargs = hyperparams.pop("policy_kwargs", None)
        if policy_kwargs:
            policy_kwargs = self._process_policy_kwargs(policy_kwargs)

        # Remove None values (some algorithms don't support certain params)
        hyperparams = {k: v for k, v in hyperparams.items() if v is not None}

        # Create agent
        agent = algo_class(
            policy=policy,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            **hyperparams,
        )

        logger.info(f"Created {algo_name} agent with {policy} policy")

        return agent, vec_env

    def load_agent(
        self,
        model_path: Path,
        env: gym.Env,
        algorithm: Optional[str] = None,
        device: str = "auto",
    ) -> tuple:
        """
        Load a trained agent from checkpoint.

        Args:
            model_path: Path to the saved model
            env: Environment for the agent
            algorithm: Algorithm name (auto-detected if None)
            device: Device to use

        Returns:
            Tuple of (agent, wrapped_environment)
        """
        model_path = Path(model_path)

        # Try to auto-detect algorithm from path or metadata
        if algorithm is None:
            algorithm = self._detect_algorithm(model_path)

        algo_name = algorithm.split(".")[-1] if "." in algorithm else algorithm
        if algo_name not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        algo_class = ALGORITHMS[algo_name]

        # Wrap environment
        vec_env = self._wrap_environment(env, is_eval=True)

        # Load agent
        agent = algo_class.load(str(model_path), env=vec_env, device=device)

        # Try to load VecNormalize statistics
        vec_norm_path = model_path.parent / "vecnormalize.pkl"
        if vec_norm_path.exists() and isinstance(vec_env, VecNormalize):
            loaded_vec_env = VecNormalize.load(str(vec_norm_path), vec_env.venv)
            loaded_vec_env.training = False
            loaded_vec_env.norm_reward = False
            vec_env = loaded_vec_env
            agent.set_env(vec_env)
            logger.info(f"Loaded VecNormalize stats from {vec_norm_path}")

        logger.info(f"Loaded {algo_name} agent from {model_path}")

        return agent, vec_env

    def wrap_eval_environment(
        self,
        env: gym.Env,
        training_vec_env: Optional[VecNormalize] = None,
    ) -> Union[DummyVecEnv, VecNormalize]:
        """
        Wrap an evaluation environment, optionally copying normalization stats.

        Args:
            env: Raw environment to wrap
            training_vec_env: Training VecNormalize to copy stats from

        Returns:
            Wrapped environment
        """
        vec_env = self._wrap_environment(env, is_eval=True)

        # Copy normalization statistics from training env
        if training_vec_env and isinstance(vec_env, VecNormalize):
            if hasattr(training_vec_env, "obs_rms"):
                vec_env.obs_rms = training_vec_env.obs_rms
            if hasattr(training_vec_env, "ret_rms"):
                vec_env.ret_rms = training_vec_env.ret_rms
            logger.info("Copied normalization stats from training environment")

        return vec_env

    def predict(
        self,
        agent: BaseAlgorithm,
        observation: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Get agent's action for an observation.

        Args:
            agent: The RL agent
            observation: Environment observation
            deterministic: Use deterministic action
            action_masks: Optional action masks for MaskablePPO

        Returns:
            Tuple of (action, state)
        """
        if action_masks is not None and MASKABLE_AVAILABLE and isinstance(agent, MaskablePPO):
            return agent.predict(
                observation,
                deterministic=deterministic,
                action_masks=action_masks,
            )
        return agent.predict(observation, deterministic=deterministic)

    def _wrap_environment(
        self,
        env: gym.Env,
        is_eval: bool = False,
    ) -> Union[DummyVecEnv, VecNormalize]:
        """
        Wrap environment with Monitor, DummyVecEnv, and optionally VecNormalize.
        """
        # Wrap with Monitor for episode statistics
        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda e=monitored_env: e])

        # Optionally wrap with VecNormalize
        if self._vec_normalize_enabled:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=self._norm_obs,
                norm_reward=False if is_eval else self._norm_reward,
                clip_obs=self._clip_obs,
                clip_reward=self._clip_reward if not is_eval else 10.0,
                training=not is_eval,
            )

        return vec_env

    def _process_policy_kwargs(self, policy_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process policy_kwargs, converting string references to actual objects.
        """
        import importlib
        import torch.nn as nn

        processed = policy_kwargs.copy()

        # Convert activation function string to actual function
        if "activation_fn" in processed:
            activation_name = processed["activation_fn"]
            if isinstance(activation_name, str):
                processed["activation_fn"] = getattr(nn, activation_name)

        # Handle features_extractor_class if it's a string reference
        if "features_extractor_class" in processed:
            extractor_class = processed["features_extractor_class"]
            if isinstance(extractor_class, str):
                try:
                    module_path, class_name = extractor_class.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    processed["features_extractor_class"] = getattr(module, class_name)
                    logger.info(f"Loaded feature extractor: {class_name}")
                except (ValueError, ImportError, AttributeError) as e:
                    logger.error(f"Failed to import {extractor_class}: {e}")
                    raise

        return processed

    def _detect_algorithm(self, model_path: Path) -> str:
        """
        Try to detect algorithm from model path or metadata.
        """
        # Check for metadata file
        metadata_path = model_path.parent / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
                if "algorithm" in metadata:
                    return metadata["algorithm"]

        # Check path for algorithm hints
        path_str = str(model_path).lower()
        for algo_name in ALGORITHMS.keys():
            if algo_name.lower() in path_str:
                return algo_name

        # Default to PPO
        logger.warning(f"Could not detect algorithm for {model_path}, defaulting to PPO")
        return "PPO"
