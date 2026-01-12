"""
CheckpointService - Application service for model persistence.

This service handles saving and loading model checkpoints,
including VecNormalize statistics and metadata.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize

logger = logging.getLogger(__name__)


class CheckpointService:
    """
    Application service for model checkpointing.

    Responsibilities:
    - Saving model checkpoints with metadata
    - Loading checkpoints with VecNormalize statistics
    - Creating training callbacks for periodic saving
    - Managing best model tracking

    Example:
        >>> checkpoint_service = CheckpointService(save_path=Path("models"))
        >>>
        >>> # Create callbacks for training
        >>> callbacks = checkpoint_service.create_training_callbacks(
        ...     eval_env=eval_vec_env,
        ...     eval_freq=10000,
        ...     save_freq=50000,
        ... )
        >>>
        >>> # Save final model
        >>> checkpoint_service.save_model(agent, vec_env, "final_model")
    """

    def __init__(
        self,
        save_path: Path,
        name_prefix: str = "rl_model",
    ):
        """
        Initialize the checkpoint service.

        Args:
            save_path: Base path for saving checkpoints
            name_prefix: Prefix for checkpoint files
        """
        self._save_path = Path(save_path)
        self._name_prefix = name_prefix
        self._best_model_path: Optional[Path] = None

        # Ensure save directory exists
        self._save_path.mkdir(parents=True, exist_ok=True)

    @property
    def save_path(self) -> Path:
        """Base path for checkpoints."""
        return self._save_path

    @property
    def best_model_path(self) -> Optional[Path]:
        """Path to the best model (if tracked)."""
        return self._best_model_path

    def save_model(
        self,
        agent: BaseAlgorithm,
        vec_env: Optional[VecNormalize] = None,
        name: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a model checkpoint with optional metadata.

        Args:
            agent: The RL agent to save
            vec_env: Optional VecNormalize environment to save stats
            name: Name for the checkpoint
            metadata: Optional metadata to save alongside

        Returns:
            Path to the saved model
        """
        checkpoint_dir = self._save_path / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = checkpoint_dir / "model.zip"
        agent.save(str(model_path))
        logger.info(f"Saved model to {model_path}")

        # Save VecNormalize stats
        if vec_env is not None and isinstance(vec_env, VecNormalize):
            vec_norm_path = checkpoint_dir / "vecnormalize.pkl"
            vec_env.save(str(vec_norm_path))
            logger.info(f"Saved VecNormalize stats to {vec_norm_path}")

        # Save metadata
        full_metadata = metadata.copy() if metadata else {}
        full_metadata["saved_at"] = datetime.now().isoformat()
        full_metadata["model_file"] = "model.zip"

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)

        return model_path

    def save_final_model(
        self,
        agent: BaseAlgorithm,
        vec_env: Optional[VecNormalize] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save the final trained model.

        Args:
            agent: The trained agent
            vec_env: Optional VecNormalize environment
            metadata: Optional training metadata

        Returns:
            Path to the saved model
        """
        return self.save_model(
            agent=agent,
            vec_env=vec_env,
            name="final_model",
            metadata=metadata,
        )

    def create_training_callbacks(
        self,
        eval_env: Optional[Any] = None,
        eval_freq: Optional[int] = None,
        n_eval_episodes: int = 10,
        save_freq: Optional[int] = None,
        additional_callbacks: Optional[List[BaseCallback]] = None,
        verbose: int = 1,
    ) -> CallbackList:
        """
        Create callbacks for training.

        Args:
            eval_env: Environment for periodic evaluation
            eval_freq: How often to evaluate (in timesteps)
            n_eval_episodes: Number of evaluation episodes
            save_freq: How often to save checkpoints (in timesteps)
            additional_callbacks: Extra callbacks to include
            verbose: Verbosity level

        Returns:
            CallbackList containing all callbacks
        """
        callbacks = []

        # Evaluation callback with best model saving
        if eval_env is not None and eval_freq is not None:
            best_model_dir = self._save_path / "best_model"
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(self._save_path / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
                verbose=verbose,
            )
            callbacks.append(eval_callback)
            self._best_model_path = best_model_dir / "best_model.zip"

        # Periodic checkpoint callback
        if save_freq is not None:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self._save_path / "checkpoints"),
                name_prefix=self._name_prefix,
                save_replay_buffer=True,
                save_vecnormalize=True,
                verbose=verbose,
            )
            callbacks.append(checkpoint_callback)

        # Add any additional callbacks
        if additional_callbacks:
            callbacks.extend(additional_callbacks)

        return CallbackList(callbacks) if callbacks else CallbackList([])

    def list_checkpoints(self) -> List[Path]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint directories
        """
        checkpoints = []

        # Check main save path
        for item in self._save_path.iterdir():
            if item.is_dir():
                model_file = item / "model.zip"
                if model_file.exists():
                    checkpoints.append(item)

        # Check checkpoints subdirectory
        checkpoints_dir = self._save_path / "checkpoints"
        if checkpoints_dir.exists():
            for item in checkpoints_dir.glob(f"{self._name_prefix}_*.zip"):
                checkpoints.append(item)

        return sorted(checkpoints)

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the most recent checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def load_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Load metadata for a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory or file

        Returns:
            Metadata dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        # Handle both directory and file paths
        if checkpoint_path.is_file():
            checkpoint_path = checkpoint_path.parent

        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)

        return {}

    def cleanup_old_checkpoints(
        self,
        keep_last: int = 5,
        keep_best: bool = True,
    ) -> int:
        """
        Remove old checkpoints to save disk space.

        Args:
            keep_last: Number of recent checkpoints to keep
            keep_best: Whether to always keep best_model

        Returns:
            Number of checkpoints removed
        """
        checkpoints_dir = self._save_path / "checkpoints"
        if not checkpoints_dir.exists():
            return 0

        # Get checkpoint files sorted by modification time
        checkpoint_files = sorted(
            checkpoints_dir.glob(f"{self._name_prefix}_*.zip"),
            key=lambda p: p.stat().st_mtime,
        )

        # Determine which to remove
        to_remove = checkpoint_files[:-keep_last] if len(checkpoint_files) > keep_last else []

        removed = 0
        for checkpoint in to_remove:
            try:
                checkpoint.unlink()
                # Also remove associated files
                for ext in [".pkl", "_vecnormalize.pkl"]:
                    assoc_file = checkpoint.with_suffix(ext)
                    if assoc_file.exists():
                        assoc_file.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Failed to remove {checkpoint}: {e}")

        if removed > 0:
            logger.info(f"Removed {removed} old checkpoints")

        return removed
