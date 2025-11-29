"""Root configuration model combining all config sections."""

from typing import Union

from pydantic import BaseModel, Field

from rl_trading_lab.config.agent import A2CConfig, AgentConfig, DQNConfig, MaskablePPOConfig, PPOConfig
from rl_trading_lab.config.data import DataConfig
from rl_trading_lab.config.env import EnvConfig
from rl_trading_lab.config.experiment import ExperimentConfig
from rl_trading_lab.config.observation import ObservationConfig
from rl_trading_lab.config.feature_engineering import FeatureEngineeringConfig
from rl_trading_lab.config.logging import LoggingConfig
from rl_trading_lab.config.training import TrainingConfig


class RootConfig(BaseModel):
    """Root configuration combining all config sections.

    This is the main config object that gets passed around in the codebase.
    All configuration access should go through this typed object.
    """

    experiment: ExperimentConfig
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig
    env: EnvConfig
    observation: ObservationConfig
    feature_engineering: FeatureEngineeringConfig
    # Discriminated union - will be one of PPOConfig, A2CConfig, DQNConfig, or MaskablePPOConfig
    agent: Union[PPOConfig, A2CConfig, DQNConfig, MaskablePPOConfig] = Field(..., discriminator="name")
