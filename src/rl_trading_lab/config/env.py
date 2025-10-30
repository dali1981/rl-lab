"""Trading environment configuration."""

from typing import List

from pydantic import BaseModel, Field


class VecNormalizeConfig(BaseModel):
    """VecNormalize wrapper configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable VecNormalize wrapper for observation and reward normalization"
    )
    norm_obs: bool = Field(
        default=True,
        description="Normalize observations using running mean and std"
    )
    norm_reward: bool = Field(
        default=True,
        description="Normalize rewards using running mean and std (training only)"
    )
    clip_obs: float = Field(
        default=10.0,
        description="Clip normalized observations to [-clip_obs, clip_obs]"
    )
    clip_reward: float = Field(
        default=10.0,
        description="Clip normalized rewards to [-clip_reward, clip_reward]"
    )


class EnvironmentParams(BaseModel):
    """Trading environment parameters."""

    lookback_window: int
    initial_balance: float
    max_position_pct: float
    commission_rate: float
    slippage_rate: float
    reward_type: str
    discrete_actions: bool
    randomize_start: bool
    min_episode_length: int
    min_holding_period: int
    hold_closes_position: bool
    one_trade_mode: bool


class EnvConfig(BaseModel):
    """Environment configuration."""

    # Data requirements
    price_column: str = Field(
        default="close",
        description="Column to use for trade execution and P&L calculation"
    )

    required_columns: List[str] = Field(
        default=["close", "timestamp"],
        description="Columns that must exist in the data for environment to function"
    )

    # VecNormalize configuration
    vec_normalize: VecNormalizeConfig = Field(
        default_factory=VecNormalizeConfig,
        description="VecNormalize wrapper configuration for observation/reward normalization"
    )

    # Environment parameters
    environment_params: EnvironmentParams
