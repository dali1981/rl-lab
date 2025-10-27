"""
Transformer-based Actor-Critic Policy for RL Trading

This module implements a custom policy network that uses a Transformer encoder
to process temporal sequences of market states. The architecture consists of:

1. Feature Projection: Projects input features to transformer hidden dimension
2. Positional Encoding: Sinusoidal encoding to provide temporal information
3. Transformer Encoder: Shared encoder with multi-head self-attention
4. Aggregation: Mean pooling over sequence to get fixed representation
5. Policy Head: Separate MLP for action distribution
6. Value Head: Separate MLP for state value estimation

All parameters are trainable end-to-end during RL training.
"""

import math
from typing import Dict, List, Tuple, Type, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import Distribution


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in "Attention is All You Need".

    Adds position information to the input sequence using sine and cosine functions
    of different frequencies. This is a fixed (non-learnable) encoding.

    Args:
        d_model: Dimension of the model (hidden size)
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor for temporal sequences.

    Processes sequences of market observations using a Transformer encoder
    to capture temporal dependencies and patterns. The output is a fixed-size
    representation obtained by mean pooling over the sequence.

    Architecture:
        Input (batch, obs_dim) → Reshape to (batch, seq_len, n_features)
        → Feature Projection (n_features → d_model)
        → Positional Encoding (sinusoidal)
        → Transformer Encoder (shared for policy and value)
        → Mean Pooling (seq_len → 1)
        → Output (batch, d_model)

    Args:
        observation_space: Gym observation space
        lookback_window: Number of timesteps in the sequence
        n_features: Number of features per timestep
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_encoder_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout rate
        aggregation: How to aggregate sequence ("mean", "last", or "cls")
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        lookback_window: int = 20,
        n_features: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ):
        # The features_dim is what this extractor outputs (will be input to policy/value heads)
        super().__init__(observation_space, features_dim=d_model)

        self.lookback_window = lookback_window
        self.n_features = n_features
        self.d_model = d_model
        self.aggregation = aggregation

        # Calculate expected observation dimension
        # TradingEnv observation = (n_features * lookback_window) + 4 position features
        self.sequence_dim = n_features * lookback_window
        self.position_info_dim = 4  # position, entry_price, pnl, cash_pct

        # Feature projection: maps input features to transformer dimension
        self.feature_projection = nn.Linear(n_features, d_model)

        # Position information projection (separate from temporal features)
        self.position_projection = nn.Linear(self.position_info_dim, d_model)

        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=lookback_window + 1,  # +1 for potential CLS token
            dropout=dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=False,  # (seq, batch, feature) format
            norm_first=True,    # Pre-LN for better training stability
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )

        # CLS token (only used if aggregation == "cls")
        if aggregation == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer feature extractor.

        Args:
            observations: Tensor of shape (batch_size, obs_dim)
                         where obs_dim = (n_features * lookback_window) + 4

        Returns:
            features: Tensor of shape (batch_size, d_model)
        """
        batch_size = observations.shape[0]

        # Split observation into temporal sequence and position info
        # Temporal features: first (n_features * lookback_window) dimensions
        # Position info: last 4 dimensions
        temporal_obs = observations[:, :self.sequence_dim]
        position_info = observations[:, self.sequence_dim:]

        # Reshape temporal observations to (batch, seq_len, n_features)
        sequence = temporal_obs.view(batch_size, self.lookback_window, self.n_features)

        # Project features to transformer dimension
        # (batch, seq_len, n_features) -> (batch, seq_len, d_model)
        x = self.feature_projection(sequence)

        # Project position information
        # (batch, 4) -> (batch, 1, d_model)
        position_embedding = self.position_projection(position_info).unsqueeze(1)

        # Add position information as an additional token
        # (batch, seq_len, d_model) -> (batch, seq_len+1, d_model)
        x = torch.cat([x, position_embedding], dim=1)

        # Handle CLS token if using cls aggregation
        if self.aggregation == "cls":
            # Add CLS token at the beginning
            cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # (1, batch, d_model)
            # Convert x to (seq, batch, d_model) format
            x = x.transpose(0, 1)  # (seq_len+1, batch, d_model)
            x = torch.cat([cls_tokens, x], dim=0)  # (seq_len+2, batch, d_model)
        else:
            # Convert to (seq, batch, feature) format for transformer
            x = x.transpose(0, 1)  # (seq_len+1, batch, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        # Output: (seq_len+1, batch, d_model) or (seq_len+2, batch, d_model) if CLS
        encoded = self.transformer_encoder(x)

        # Aggregate sequence to fixed-size representation
        if self.aggregation == "mean":
            # Mean pooling over sequence dimension
            # (seq, batch, d_model) -> (batch, d_model)
            features = encoded.mean(dim=0)
        elif self.aggregation == "last":
            # Use last token
            features = encoded[-1]
        elif self.aggregation == "cls":
            # Use CLS token (first token)
            features = encoded[0]
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return features


class TransformerActorCriticPolicy(ActorCriticPolicy):
    """
    Transformer-based Actor-Critic policy for RL agents.

    This policy uses a shared Transformer encoder to process temporal sequences
    of observations, followed by separate MLP heads for the policy (actor) and
    value function (critic).

    Architecture:
        Observations → TransformerFeatureExtractor (shared)
                    ├→ Policy Head (MLP) → Action Distribution
                    └→ Value Head (MLP) → State Value

    All components are trainable end-to-end during RL training.

    Usage with Stable-Baselines3:
        ```python
        policy_kwargs = dict(
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(
                lookback_window=20,
                n_features=4,
                d_model=128,
                nhead=4,
                num_encoder_layers=2,
                dim_feedforward=256,
                dropout=0.1,
                aggregation="mean",
            ),
            net_arch=dict(
                pi=[128, 64],  # Policy head layers
                vf=[128, 64],  # Value head layers
            ),
        )

        model = PPO("TransformerPolicy", env, policy_kwargs=policy_kwargs)
        ```

    Args:
        observation_space: Gym observation space
        action_space: Gym action space
        lr_schedule: Learning rate schedule
        net_arch: Policy and value network architecture (heads after transformer)
        activation_fn: Activation function for MLP heads
        **kwargs: Additional arguments passed to ActorCriticPolicy
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        # Set default network architecture for heads if not provided
        if net_arch is None:
            net_arch = dict(pi=[128, 64], vf=[128, 64])

        # Force the use of TransformerFeatureExtractor
        if 'features_extractor_class' not in kwargs:
            kwargs['features_extractor_class'] = TransformerFeatureExtractor

        # Initialize parent ActorCriticPolicy
        # This will automatically create the feature extractor and policy/value networks
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Build the policy and value networks (heads) after the transformer.

        This is called by the parent ActorCriticPolicy during initialization.
        The transformer feature extractor is already built, and this method
        creates the separate MLP heads for policy and value.
        """
        # Call parent implementation to build the MLP heads
        super()._build_mlp_extractor()


# Note: To use this policy with SB3, you can either:
# 1. Pass the class directly: model = PPO(TransformerActorCriticPolicy, env, ...)
# 2. Use policy string and import: from rl_trading_lab.models import TransformerActorCriticPolicy
#    Then create a custom mapping in your agent wrapper
