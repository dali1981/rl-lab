"""Experiment configuration."""

from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Experiment metadata and settings."""

    name: str
    run_name: str
    seed: int
    device: str
