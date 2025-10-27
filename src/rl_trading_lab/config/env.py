"""Trading environment configuration."""

from typing import List

from pydantic import BaseModel, Field


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
    hold_closes_position: bool


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

    # Environment parameters
    environment_params: EnvironmentParams
