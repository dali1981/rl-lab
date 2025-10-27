"""Data configuration."""

from pydantic import BaseModel


class DataConfig(BaseModel):
    """Data paths and preprocessing configuration."""

    train_data_path: str
    val_split: float
    test_split: float
    update_frequency: int
