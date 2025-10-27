"""Logging configuration."""

from pydantic import BaseModel


class MLflowConfig(BaseModel):
    """MLflow logging configuration."""

    enabled: bool
    tracking_uri: str
    experiment_name: str


class TensorboardConfig(BaseModel):
    """Tensorboard logging configuration."""

    enabled: bool
    log_dir: str


class ConsoleConfig(BaseModel):
    """Console logging configuration."""

    verbose: bool
    progress_bar: bool


class LoggingConfig(BaseModel):
    """Logging configuration."""

    mlflow: MLflowConfig
    tensorboard: TensorboardConfig
    console: ConsoleConfig
