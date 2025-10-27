"""Observation space configuration - controls what the RL agent sees."""

from typing import List

from pydantic import BaseModel, Field


class ObservationConfig(BaseModel):
    """Observation space configuration.

    Controls which features are included in the agent's observation space.
    These must exist in the data - they are not created at runtime.
    """

    input_features: List[str] = Field(
        ...,
        description="List of features to include in agent's observation space",
        min_length=1
    )

    validate_features: bool = Field(
        default=True,
        description="Validate that specified features exist in the data"
    )

    log_all_features: bool = Field(
        default=True,
        description="Log complete feature list (if false, shows only first 5)"
    )
