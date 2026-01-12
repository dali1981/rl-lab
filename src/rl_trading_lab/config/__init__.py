"""Configuration package - Pydantic models for type-safe config management."""

from rl_trading_lab.config.agent import (
    A2CConfig,
    A2CHyperparameters,
    AgentConfig,
    BaseAgentConfig,
    DQNConfig,
    DQNHyperparameters,
    DQNPolicyKwargs,
    MaskablePPOConfig,
    PolicyKwargs,
    PPOConfig,
    PPOHyperparameters,
    TrainingParams,
)
from rl_trading_lab.config.data import DataConfig
from rl_trading_lab.config.env import EnvConfig, EnvironmentParams
from rl_trading_lab.config.experiment import ExperimentConfig
from rl_trading_lab.config.observation import ObservationConfig
from rl_trading_lab.config.feature_engineering import (
    FeatureEngineeringConfig,
    MissingValuesConfig,
)
from rl_trading_lab.config.loader import load_config
from rl_trading_lab.config.logging import (
    ConsoleConfig,
    LoggingConfig,
    MLflowConfig,
    TensorboardConfig,
)
from rl_trading_lab.config.main import RootConfig
from rl_trading_lab.config.training import TrainingConfig

__all__ = [
    # Main config
    "RootConfig",
    "load_config",
    # Agent configs
    "AgentConfig",
    "BaseAgentConfig",
    "PPOConfig",
    "A2CConfig",
    "DQNConfig",
    "MaskablePPOConfig",
    "PPOHyperparameters",
    "A2CHyperparameters",
    "DQNHyperparameters",
    "PolicyKwargs",
    "DQNPolicyKwargs",
    "TrainingParams",
    # Environment config
    "EnvConfig",
    "EnvironmentParams",
    # Observation config
    "ObservationConfig",
    # Feature engineering config
    "FeatureEngineeringConfig",
    "MissingValuesConfig",
    # Training config
    "TrainingConfig",
    # Experiment config
    "ExperimentConfig",
    # Data config
    "DataConfig",
    # Logging config
    "LoggingConfig",
    "MLflowConfig",
    "TensorboardConfig",
    "ConsoleConfig",
]
