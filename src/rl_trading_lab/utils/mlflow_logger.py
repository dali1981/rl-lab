"""
MLflow integration for Stable-Baselines3 logging.
Implements custom output format to automatically log metrics to MLflow.
"""

from typing import Any, Dict, Tuple, Union
import numpy as np
import mlflow
from stable_baselines3.common.logger import KVWriter


class MLflowOutputFormat(KVWriter):
    """
    Custom logger for MLflow integration with SB3.

    Automatically logs all SB3 metrics (rollout/, train/, eval/) to MLflow.
    Filters out metrics that should not be logged based on exclusion rules.

    Usage:
        from stable_baselines3.common.logger import Logger, configure

        logger = Logger(folder=None, output_formats=[MLflowOutputFormat()])
        model.set_logger(logger)
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        Write key-value pairs to MLflow.

        Args:
            key_values: Dictionary of metrics to log
            key_excluded: Dictionary specifying which outputs to exclude per key
            step: Training step/timestep
        """
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # Skip if this key is excluded from MLflow
            if excluded is not None and "mlflow" in excluded:
                continue

            # Only log scalar values (not strings or complex types)
            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)

    def close(self) -> None:
        """Close the output format (no-op for MLflow)."""
        pass
