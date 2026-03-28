"""Data configuration."""

from typing import Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data paths and preprocessing configuration."""

    train_data_path: str
    val_split: float
    test_split: float
    update_frequency: int

    source_type: str = Field(
        default="parquet",
        description="Data source type: 'parquet', 'csv', or 'binance_delta'"
    )

    feature_pipeline: str = Field(
        default="crypto",
        description="Feature engineering pipeline: 'crypto' or 'passthrough'"
    )
