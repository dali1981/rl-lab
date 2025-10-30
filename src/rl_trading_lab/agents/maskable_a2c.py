"""
Custom A2C wrapper with action masking support.

Since sb3-contrib only provides MaskablePPO and not MaskableA2C,
this wrapper adds action masking capability to the standard A2C algorithm.
"""
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional, Tuple, Union


class MaskableA2C(A2C):
    """
    A2C with action masking support.

    This wrapper modifies the action selection process to respect action masks
    provided by the environment's info dict. Invalid actions (mask=0) have their
    logits set to -inf before sampling.

    Usage:
        model = MaskableA2C("MlpPolicy", env, ...)
        model.learn(total_timesteps=100000)

    Environment requirements:
        - Must provide 'action_mask' in the info dict returned by step() and reset()
        - action_mask should be a numpy array of shape (n_actions,) with 0/1 values
        - 1 = valid action, 0 = invalid action
    """

    def predict(
        self,
        observation: Union[np.ndarray, dict],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).

        Args:
            observation: The input observation
            state: The last hidden states (only for recurrent policies)
            episode_start: Whether this is the start of a new episode
            deterministic: Whether to use stochastic or deterministic actions
            action_masks: Optional action masks (shape: (n_envs, n_actions) or (n_actions,))
                         1 = valid action, 0 = invalid action

        Returns:
            The model's action and next hidden state (if applicable)
        """
        if action_masks is None:
            # No masking, use standard prediction
            return super().predict(observation, state, episode_start, deterministic)

        # Handle single environment case
        if action_masks.ndim == 1:
            action_masks = action_masks.reshape(1, -1)

        # Convert observation to tensor
        obs_tensor = self.policy.obs_to_tensor(observation)[0]

        with self.policy.set_training_mode(False):
            # Get action distribution
            distribution = self.policy.get_distribution(obs_tensor)

            # Get logits from the distribution
            if hasattr(distribution.distribution, 'logits'):
                logits = distribution.distribution.logits.clone()
            else:
                # For some distributions, we need to compute logits differently
                # This should work for Categorical distributions used in discrete action spaces
                logits = distribution.distribution.logits.clone()

            # Apply action masks by setting invalid actions to -inf
            # This ensures they have 0 probability after softmax
            logits[action_masks == 0] = -1e8

            # Sample action from masked distribution
            if deterministic:
                # For deterministic, take argmax of masked logits
                actions = logits.argmax(dim=1).cpu().numpy()
            else:
                # For stochastic, sample from masked distribution
                # Create new categorical distribution with masked logits
                from torch.distributions import Categorical
                masked_dist = Categorical(logits=logits)
                actions = masked_dist.sample().cpu().numpy()

        return actions, state


def make_maskable_a2c(
    env,
    learning_rate: float = 7e-4,
    n_steps: int = 5,
    gamma: float = 0.99,
    gae_lambda: float = 1.0,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    **kwargs
):
    """
    Convenience function to create MaskableA2C with common hyperparameters.

    Args:
        env: The environment
        learning_rate: Learning rate
        n_steps: Number of steps per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        **kwargs: Additional arguments passed to MaskableA2C

    Returns:
        MaskableA2C instance
    """
    return MaskableA2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        **kwargs
    )
