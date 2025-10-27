"""Feature engineering configuration - creates new features at runtime."""

from typing import List, Optional

from pydantic import BaseModel, Field


class MissingValuesConfig(BaseModel):
    """Missing values handling for engineered features."""

    strategy: str = Field(
        default="forward_fill",
        description="Strategy for handling missing values: forward_fill, interpolate, drop"
    )

    initial_fill: float = Field(
        default=0.0,
        description="Fill value for NaN at start of data"
    )


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration.

    Controls runtime feature creation. Most users will create features in
    their data pipeline (e.g., Kedro) instead of using this.
    """

    enabled: bool = Field(
        default=False,
        description="Enable runtime feature engineering"
    )

    # Returns features
    add_returns: bool = Field(
        default=False,
        description="Add returns features"
    )

    return_periods: List[int] = Field(
        default=[1, 5, 20],
        description="Periods for return calculation"
    )

    add_log_returns: bool = Field(
        default=False,
        description="Add log returns"
    )

    # Rolling statistics
    add_rolling_stats: bool = Field(
        default=False,
        description="Add rolling statistics (mean, std, etc.)"
    )

    rolling_window: int = Field(
        default=20,
        description="Window size for rolling statistics"
    )

    rolling_stats: List[str] = Field(
        default=["mean", "std"],
        description="Which rolling statistics to compute"
    )

    # Missing values handling
    missing_values: Optional[MissingValuesConfig] = Field(
        default=None,
        description="How to handle missing values in engineered features"
    )

    def model_post_init(self, __context) -> None:
        """Validate configuration."""
        if self.enabled:
            # If enabled, ensure at least one feature engineering option is turned on
            if not any([
                self.add_returns,
                self.add_log_returns,
                self.add_rolling_stats
            ]):
                raise ValueError(
                    "Feature engineering is enabled but no features selected. "
                    "Set at least one of: add_returns, add_log_returns, add_rolling_stats"
                )
