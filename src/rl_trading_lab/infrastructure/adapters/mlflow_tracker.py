"""
MLflowExperimentTracker - MLflow implementation of ExperimentTrackerPort.

This adapter implements experiment tracking using MLflow, providing:
- Automatic parameter and metric logging
- Artifact storage
- Run management
- Integration with MLflow UI
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow

from rl_trading_lab.application.ports.experiment_tracker import ExperimentTrackerPort

logger = logging.getLogger(__name__)


class MLflowExperimentTracker(ExperimentTrackerPort):
    """
    MLflow implementation of experiment tracking.

    Implements the ExperimentTrackerPort protocol using MLflow
    for experiment tracking, metric logging, and artifact storage.

    Example:
        >>> tracker = MLflowExperimentTracker(
        ...     tracking_uri="file:./mlruns",
        ...     experiment_name="trading_experiments",
        ... )
        >>> tracker.start_run(
        ...     run_name="ppo_returns_v1",
        ...     params={"algorithm": "PPO", "reward_type": "returns"},
        ... )
        >>> tracker.log_metrics({"loss": 0.5, "reward": 100.0}, step=1000)
        >>> tracker.end_run()
    """

    def __init__(
        self,
        tracking_uri: str = "file:./mlruns",
        experiment_name: str = "rl_trading_lab",
        create_experiment: bool = True,
    ):
        """
        Initialize the MLflow tracker.

        Args:
            tracking_uri: URI for MLflow tracking server
            experiment_name: Name of the MLflow experiment
            create_experiment: Whether to create experiment if it doesn't exist
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id: Optional[str] = None

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        if create_experiment:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

        logger.info(f"MLflow tracker initialized: {tracking_uri}/{experiment_name}")

    def start_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            params: Parameters to log
            tags: Optional tags
        """
        # End any existing run
        if mlflow.active_run():
            mlflow.end_run()

        # Start new run
        run = mlflow.start_run(run_name=run_name)
        self._run_id = run.info.run_id

        # Log parameters (filter non-serializable values)
        filtered_params = self._filter_params(params)
        if filtered_params:
            mlflow.log_params(filtered_params)

        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logger.info(f"Started MLflow run: {run_name} (ID: {self._run_id[:8]})")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to current run."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping log_params")
            return

        filtered = self._filter_params(params)
        if filtered:
            mlflow.log_params(filtered)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics to current run."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping log_metrics")
            return

        # Filter valid metric values
        valid_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v))  # Filter NaN
        }

        if valid_metrics:
            mlflow.log_metrics(valid_metrics, step=step)

    def log_artifact(
        self,
        path: Path,
        artifact_name: Optional[str] = None,
    ) -> None:
        """Log an artifact file to current run."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping log_artifact")
            return

        path = Path(path)
        if not path.exists():
            logger.warning(f"Artifact path does not exist: {path}")
            return

        mlflow.log_artifact(str(path), artifact_path=artifact_name)
        logger.debug(f"Logged artifact: {path.name}")

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_name: str,
    ) -> None:
        """Log a dictionary as a JSON artifact."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping log_dict")
            return

        # Convert to serializable format
        serializable = self._make_serializable(dictionary)
        mlflow.log_dict(serializable, artifact_name)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on current run."""
        if not mlflow.active_run():
            logger.warning("No active MLflow run, skipping set_tag")
            return

        mlflow.set_tag(key, value)

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if mlflow.active_run():
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
        self._run_id = None

    @property
    def is_active(self) -> bool:
        """Check if there's an active run."""
        return mlflow.active_run() is not None

    @property
    def run_id(self) -> Optional[str]:
        """Get current run ID."""
        return self._run_id

    def _filter_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter parameters to only include MLflow-compatible types.

        MLflow params must be strings, numbers, or booleans.
        """
        filtered = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif value is None:
                filtered[key] = "None"
            elif isinstance(value, (list, tuple)):
                # Convert short lists to string
                if len(value) <= 5:
                    filtered[key] = str(value)
            elif isinstance(value, dict):
                # Skip nested dicts, they can be logged as artifacts
                pass
            else:
                # Convert to string as fallback
                try:
                    filtered[key] = str(value)[:250]  # MLflow param limit
                except Exception:
                    pass
        return filtered

    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format.
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, "model_dump"):
            # Pydantic models
            return obj.model_dump()
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)


def create_mlflow_tracker(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    enabled: bool = True,
) -> ExperimentTrackerPort:
    """
    Factory function to create an experiment tracker.

    Args:
        tracking_uri: MLflow tracking URI (None uses default)
        experiment_name: Experiment name (None uses default)
        enabled: Whether to create real tracker or no-op

    Returns:
        ExperimentTrackerPort implementation
    """
    if not enabled:
        from rl_trading_lab.application.ports.experiment_tracker import NoOpExperimentTracker
        return NoOpExperimentTracker()

    return MLflowExperimentTracker(
        tracking_uri=tracking_uri or "file:./mlruns",
        experiment_name=experiment_name or "rl_trading_lab",
    )
