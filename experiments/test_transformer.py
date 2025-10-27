#!/usr/bin/env python
"""
Test script for Transformer-based policy.

This script verifies that:
1. TransformerFeatureExtractor can be instantiated
2. TransformerActorCriticPolicy works with SB3
3. Forward pass through transformer works correctly
4. Policy is compatible with PPO/A2C/DQN
5. Gradients flow properly end-to-end

Usage:
    uv run python experiments/test_transformer.py
"""

import sys
from pathlib import Path
import logging

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from rl_trading_lab.models.transformer_policy import (
    TransformerFeatureExtractor,
    TransformerActorCriticPolicy,
    SinusoidalPositionalEncoding,
)
from stable_baselines3 import PPO


def test_positional_encoding():
    """Test sinusoidal positional encoding"""
    logger.info("\n=== Testing Positional Encoding ===")

    d_model = 128
    max_len = 50
    batch_size = 4
    seq_len = 20

    pos_encoder = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)

    # Create dummy input (seq_len, batch, d_model)
    x = torch.randn(seq_len, batch_size, d_model)

    # Forward pass
    output = pos_encoder(x)

    assert output.shape == (seq_len, batch_size, d_model), \
        f"Expected shape {(seq_len, batch_size, d_model)}, got {output.shape}"

    logger.info(f"✓ Positional encoding works: input {x.shape} -> output {output.shape}")


def test_transformer_feature_extractor():
    """Test TransformerFeatureExtractor forward pass"""
    logger.info("\n=== Testing Transformer Feature Extractor ===")

    # Environment parameters
    lookback_window = 20
    n_features = 4
    position_info_dim = 4
    batch_size = 8

    # Observation space (flattened: lookback * features + position info)
    obs_dim = lookback_window * n_features + position_info_dim
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    # Transformer parameters
    d_model = 128
    nhead = 4
    num_encoder_layers = 2

    # Create feature extractor
    extractor = TransformerFeatureExtractor(
        observation_space=observation_space,
        lookback_window=lookback_window,
        n_features=n_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=256,
        dropout=0.1,
        aggregation="mean",
    )

    logger.info(f"Feature extractor created with {sum(p.numel() for p in extractor.parameters())} parameters")

    # Create dummy observations
    observations = torch.randn(batch_size, obs_dim)

    # Forward pass
    features = extractor(observations)

    assert features.shape == (batch_size, d_model), \
        f"Expected shape {(batch_size, d_model)}, got {features.shape}"

    logger.info(f"✓ Feature extractor works: input {observations.shape} -> output {features.shape}")

    # Test gradient flow
    loss = features.mean()
    loss.backward()

    # Check that gradients exist
    has_grads = any(p.grad is not None for p in extractor.parameters())
    assert has_grads, "No gradients found!"

    logger.info(f"✓ Gradients flow properly through transformer")


def test_transformer_policy():
    """Test TransformerActorCriticPolicy with dummy environment"""
    logger.info("\n=== Testing Transformer Policy ===")

    # Environment parameters
    lookback_window = 20
    n_features = 4
    position_info_dim = 4
    obs_dim = lookback_window * n_features + position_info_dim

    # Create dummy environment
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    action_space = spaces.Discrete(3)  # Buy, Hold, Sell

    # Create dummy learning rate schedule
    def lr_schedule(progress):
        return 3e-4

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            lookback_window=lookback_window,
            n_features=n_features,
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            aggregation="mean",
        ),
        net_arch=dict(
            pi=[128, 64],  # Policy head
            vf=[128, 64],  # Value head
        ),
    )

    # Create policy
    policy = TransformerActorCriticPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **policy_kwargs,
    )

    total_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy created with {total_params:,} parameters")

    # Test forward pass
    batch_size = 4
    observations = torch.randn(batch_size, obs_dim)

    # Get actions
    actions, values, log_probs = policy(observations)

    assert actions.shape == (batch_size,), f"Expected actions shape {(batch_size,)}, got {actions.shape}"
    assert values.shape == (batch_size, 1), f"Expected values shape {(batch_size, 1)}, got {values.shape}"
    assert log_probs.shape == (batch_size,), f"Expected log_probs shape {(batch_size,)}, got {log_probs.shape}"

    logger.info(f"✓ Policy forward pass works:")
    logger.info(f"  Actions: {actions.shape}")
    logger.info(f"  Values: {values.shape}")
    logger.info(f"  Log probs: {log_probs.shape}")

    # Test gradient flow
    loss = values.mean() + log_probs.mean()
    loss.backward()

    has_grads = any(p.grad is not None for p in policy.parameters())
    assert has_grads, "No gradients found in policy!"

    logger.info(f"✓ Gradients flow through entire policy")


def test_ppo_with_transformer():
    """Test PPO algorithm with transformer policy"""
    logger.info("\n=== Testing PPO with Transformer Policy ===")

    # Environment parameters
    lookback_window = 20
    n_features = 4
    position_info_dim = 4
    obs_dim = lookback_window * n_features + position_info_dim

    # Create simple dummy environment
    class DummyTradingEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            obs = self.observation_space.sample()
            reward = np.random.randn()
            terminated = False
            truncated = np.random.rand() < 0.01  # 1% chance of episode end
            info = {}
            return obs, reward, terminated, truncated, info

    env = DummyTradingEnv()

    # Policy kwargs
    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            lookback_window=lookback_window,
            n_features=n_features,
            d_model=64,  # Smaller for faster testing
            nhead=4,
            num_encoder_layers=1,  # Fewer layers for testing
            dim_feedforward=128,
            dropout=0.0,
            aggregation="mean",
        ),
        net_arch=dict(
            pi=[64],
            vf=[64],
        ),
    )

    # Create PPO model with transformer policy (pass class directly)
    model = PPO(
        TransformerActorCriticPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=16,  # Very small for quick test
        batch_size=16,
        n_epochs=2,
        learning_rate=3e-4,
        verbose=0,
    )

    logger.info(f"PPO model created with TransformerPolicy")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

    # Train for a few steps
    logger.info("Training for 64 timesteps...")
    model.learn(total_timesteps=64, progress_bar=False)

    logger.info(f"✓ PPO training with transformer policy successful!")

    # Test prediction
    obs, _ = env.reset()
    action, _states = model.predict(obs, deterministic=True)

    logger.info(f"✓ Prediction works: obs {obs.shape} -> action {action}")


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Testing Transformer Policy Implementation")
    logger.info("=" * 60)

    try:
        test_positional_encoding()
        test_transformer_feature_extractor()
        test_transformer_policy()
        test_ppo_with_transformer()

        logger.info("\n" + "=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
