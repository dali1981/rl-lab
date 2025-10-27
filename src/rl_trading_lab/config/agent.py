"""Agent configuration models with discriminated unions for type safety."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class PolicyKwargs(BaseModel):
    """Network architecture configuration."""

    net_arch: Union[List[int], Dict[str, List[int]]]
    activation_fn: str
    normalize_images: bool = False


class BaseAgentConfig(BaseModel):
    """Base configuration shared by all agents."""

    name: str
    algorithm: str
    verbose: int
    tensorboard_log: str


class TrainingParams(BaseModel):
    """Training parameters specific to each agent."""

    total_timesteps: int
    eval_freq: int
    save_freq: int


class PPOHyperparameters(BaseModel):
    """PPO-specific hyperparameters."""

    # Learning
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int

    # Advantage estimation
    gamma: float
    gae_lambda: float

    # PPO clipping
    clip_range: float
    clip_range_vf: Optional[float]

    # Regularization
    ent_coef: float
    vf_coef: float
    max_grad_norm: float

    # Network architecture
    policy: str
    policy_kwargs: PolicyKwargs

    # Other settings
    normalize_advantage: bool
    use_sde: bool
    sde_sample_freq: int


class PPOConfig(BaseAgentConfig):
    """PPO agent configuration."""

    name: Literal["PPO"] = "PPO"
    hyperparameters: PPOHyperparameters
    training: TrainingParams


class A2CHyperparameters(BaseModel):
    """A2C-specific hyperparameters."""

    # Learning
    learning_rate: float
    n_steps: int

    # Discount and advantage
    gamma: float
    gae_lambda: float

    # Regularization
    ent_coef: float
    vf_coef: float
    max_grad_norm: float

    # Network architecture
    policy: str
    policy_kwargs: PolicyKwargs

    # RMS Prop optimizer
    rms_prop_eps: float
    use_rms_prop: bool
    normalize_advantage: bool


class A2CConfig(BaseAgentConfig):
    """A2C agent configuration."""

    name: Literal["A2C"] = "A2C"
    hyperparameters: A2CHyperparameters
    training: TrainingParams


class DQNPolicyKwargs(BaseModel):
    """DQN network architecture (simpler than actor-critic)."""

    net_arch: List[int]
    activation_fn: str


class DQNHyperparameters(BaseModel):
    """DQN-specific hyperparameters."""

    # Learning
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int

    # Discount
    gamma: float

    # Exploration
    exploration_fraction: float
    exploration_initial_eps: float
    exploration_final_eps: float

    # Target network
    target_update_interval: int
    tau: float

    # Training
    train_freq: int
    gradient_steps: int

    # Network architecture
    policy: str
    policy_kwargs: DQNPolicyKwargs

    # Double DQN
    optimize_memory_usage: bool
    max_grad_norm: float


class DQNConfig(BaseAgentConfig):
    """DQN agent configuration."""

    name: Literal["DQN"] = "DQN"
    hyperparameters: DQNHyperparameters
    training: TrainingParams


# Discriminated union of all agent configs
AgentConfig = Union[PPOConfig, A2CConfig, DQNConfig]
