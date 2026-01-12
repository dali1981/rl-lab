"""
ExperimentTrackerPort - Interface for experiment tracking systems.

This port defines the contract for experiment tracking implementations
such as MLflow, Weights & Biases, or custom tracking solutions.

Per Evans (DDD) and Martin (Clean Architecture), ports define the
interface that the application needs, independent of specific implementations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Protocol


class ExperimentTrackerPort(Protocol):
    """
    Port for experiment tracking services.

    Implementations must provide methods for:
    - Starting/ending experiment runs
    - Logging parameters, metrics, and artifacts
    - Managing run metadata

    Example implementations:
    - MLflowExperimentTracker: Uses MLflow for tracking
    - WandbExperimentTracker: Uses Weights & Biases
    - NoOpExperimentTracker: For testing (does nothing)
    """

    def start_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Start a new experiment run.

        Args:
            run_name: Name for this run
            params: Parameters to log (hyperparameters, config, etc.)
            tags: Optional tags for categorization
        """
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names to values
        """
        ...

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names to values
            step: Optional step/epoch number for time-series metrics
        """
        ...

    def log_artifact(
        self,
        path: Path,
        artifact_name: Optional[str] = None,
    ) -> None:
        """
        Log an artifact (file) to the current run.

        Args:
            path: Path to the artifact file
            artifact_name: Optional name/subdirectory for the artifact
        """
        ...

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_name: str,
    ) -> None:
        """
        Log a dictionary as a JSON/YAML artifact.

        Args:
            dictionary: Dictionary to log
            artifact_name: Name for the artifact file
        """
        ...

    def set_tag(self, key: str, value: str) -> None:
        """
        Set a tag on the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        ...

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Final status of the run (FINISHED, FAILED, KILLED)
        """
        ...

    @property
    def is_active(self) -> bool:
        """Check if there's an active run."""
        ...


class NoOpExperimentTracker:
    """
    No-op implementation for testing or when tracking is disabled.

    All methods do nothing, which is useful for:
    - Unit testing without tracking overhead
    - Running experiments locally without tracking setup
    - Disabling tracking via configuration
    """

    def start_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        pass

    def log_artifact(
        self,
        path: Path,
        artifact_name: Optional[str] = None,
    ) -> None:
        pass

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_name: str,
    ) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        pass

    def end_run(self, status: str = "FINISHED") -> None:
        pass

    @property
    def is_active(self) -> bool:
        return False
