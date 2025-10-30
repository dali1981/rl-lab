"""
Custom callbacks for training with CheckpointManager integration.
"""

import logging
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback

from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class CheckpointManagerCallback(CheckpointCallback):
    """
    Enhanced CheckpointCallback that saves metadata with each checkpoint.

    Extends SB3's CheckpointCallback to:
    - Save checkpoint metadata (policy type, versions, config)
    - Ensure VecNormalize stats are saved
    - Make loading more robust
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = True,
        verbose: int = 0,
        metadata: Optional[dict] = None,
    ):
        """
        Args:
            save_freq: Save frequency (in timesteps)
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint files
            save_replay_buffer: Save replay buffer (for off-policy algorithms)
            save_vecnormalize: Save VecNormalize stats
            verbose: Verbosity level
            metadata: Additional metadata to save with each checkpoint
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.checkpoint_manager = CheckpointManager()
        self.custom_metadata = metadata or {}

    def _on_step(self) -> bool:
        """Called at every step - saves checkpoint if needed"""
        # Let parent handle the standard saving
        result = super()._on_step()

        # Save metadata if we just saved a checkpoint
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = self._checkpoint_path(extension="")  # Get path without extension
            metadata_path = Path(checkpoint_path).with_suffix('.metadata.json')

            # Create metadata
            metadata = self.checkpoint_manager._create_metadata(
                self.model,
                custom_metadata=self.custom_metadata
            )

            # Save metadata
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            if self.verbose >= 2:
                logger.info(f"Saved checkpoint metadata to {metadata_path}")

        return result


class BestModelCallback(EvalCallback):
    """
    Enhanced EvalCallback that saves VecNormalize with best model.

    Fixes the issue where best_model.zip is saved without VecNormalize stats,
    making it impossible to load correctly.
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        log_path: Optional[str] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        callback_on_new_best=None,
        callback_after_eval=None,
        metadata: Optional[dict] = None,
    ):
        """Enhanced EvalCallback with metadata support"""
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.checkpoint_manager = CheckpointManager()
        self.custom_metadata = metadata or {}

    def _on_step(self) -> bool:
        """Called at every step - evaluates and saves best model"""
        # Let parent handle evaluation
        result = super()._on_step()

        # Check if a new best model was saved
        if self.best_model_save_path is not None and hasattr(self, 'last_mean_reward'):
            best_model_path = Path(self.best_model_save_path) / "best_model"

            if best_model_path.with_suffix('.zip').exists():
                # Check if we need to save metadata/vecnormalize
                metadata_path = best_model_path.with_suffix('.metadata.json')

                if not metadata_path.exists() or self._just_saved_best():
                    # Save metadata
                    metadata = self.checkpoint_manager._create_metadata(
                        self.model,
                        custom_metadata={
                            **self.custom_metadata,
                            'best_mean_reward': float(self.best_mean_reward),
                            'evaluation': {
                                'mean_reward': float(self.last_mean_reward),
                                'std_reward': float(self.last_std_reward) if hasattr(self, 'last_std_reward') else None,
                            }
                        }
                    )

                    import json
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    if self.verbose >= 1:
                        logger.info(f"Saved best model metadata to {metadata_path}")

                    # CRITICAL: Save VecNormalize stats with best model (if used)
                    vec_normalize = self.model.get_vec_normalize_env()
                    if vec_normalize is not None:
                        vecnorm_path = Path(self.best_model_save_path) / "vecnormalize.pkl"
                        vec_normalize.save(vecnorm_path)
                        if self.verbose >= 1:
                            logger.info(f"✓ Saved VecNormalize stats to {vecnorm_path}")
                    else:
                        if self.verbose >= 2:
                            logger.debug("VecNormalize not used, skipping stats save")

        return result

    def _just_saved_best(self) -> bool:
        """Check if we just saved a new best model"""
        # This is called right after parent's _on_step, so check if it was just updated
        return hasattr(self, '_is_new_best') and self._is_new_best
