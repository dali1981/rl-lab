"""
Environment wrapping utilities for SB3 training.

Builds the Monitor -> DummyVecEnv -> VecNormalize chain
required by Stable-Baselines3 algorithms.
"""

import logging
from typing import Optional, Union

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logger = logging.getLogger(__name__)


class EnvWrapperBuilder:
    """
    Builds wrapped environments for SB3 training and evaluation.

    Encapsulates the Monitor -> DummyVecEnv -> optional VecNormalize chain.

    Example:
        >>> builder = EnvWrapperBuilder(vec_normalize_enabled=True)
        >>> train_env = builder.build(raw_env, is_eval=False, gamma=0.99)
        >>> eval_env = builder.build(raw_eval_env, is_eval=True, gamma=0.99)
    """

    def __init__(
        self,
        vec_normalize_enabled: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
    ):
        self.vec_normalize_enabled = vec_normalize_enabled
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward

    def build(
        self,
        env: gym.Env,
        is_eval: bool = False,
        gamma: float = 0.99,
    ) -> Union[DummyVecEnv, VecNormalize]:
        """
        Wrap environment with Monitor, DummyVecEnv, and optionally VecNormalize.

        Args:
            env: Raw gymnasium environment
            is_eval: Whether this is an evaluation environment
            gamma: Discount factor for VecNormalize reward normalization

        Returns:
            Wrapped vectorized environment
        """
        monitored_env = Monitor(env)
        vec_env = DummyVecEnv([lambda e=monitored_env: e])

        if self.vec_normalize_enabled:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=self.norm_obs,
                norm_reward=False if is_eval else self.norm_reward,
                clip_obs=self.clip_obs,
                clip_reward=self.clip_reward if not is_eval else 10.0,
                gamma=gamma,
                training=not is_eval,
            )

        return vec_env

    def copy_normalization_stats(
        self,
        source: VecNormalize,
        target: VecNormalize,
    ) -> None:
        """
        Copy normalization statistics from a training env to an eval env.

        Args:
            source: Training environment with learned statistics
            target: Evaluation environment to receive the statistics
        """
        if hasattr(source, "obs_rms"):
            target.obs_rms = source.obs_rms
        if hasattr(source, "ret_rms"):
            target.ret_rms = source.ret_rms
        logger.info("Copied normalization stats from source to target environment")
