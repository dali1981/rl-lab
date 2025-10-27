"""Training configuration."""

from pydantic import BaseModel


class TrainingConfig(BaseModel):
    """Training parameters."""

    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    save_freq: int
    save_path: str
    early_stopping: bool
    patience: int
    min_improvement: float
