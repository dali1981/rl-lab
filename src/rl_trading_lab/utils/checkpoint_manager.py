"""
Checkpoint Manager for RL Trading Models

Provides unified interface for saving and loading model checkpoints with:
- Automatic metadata tracking (policy type, versions, config)
- Custom policy support (Transformer, custom feature extractors)
- VecNormalize stats management
- Version compatibility checking
- Robust error handling

Usage:
    # Saving (automatic during training)
    manager = CheckpointManager(checkpoint_dir)
    manager.save_checkpoint(model, vec_env, metadata)

    # Loading
    model, vec_env = manager.load_best_model(env)
    model, vec_env = manager.load_checkpoint("rl_model_10000_steps.zip", env)
"""

import json
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import sys

import torch
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import stable_baselines3 as sb3

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with metadata for robust save/load operations.

    Handles:
    - Model weights (.zip)
    - VecNormalize stats (.pkl)
    - Metadata (.json) - policy type, versions, config
    """

    METADATA_VERSION = "1.0"

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory containing checkpoints (e.g., "checkpoints/PPO_xxx")
        """
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

    def save_checkpoint(
        self,
        model: BaseAlgorithm,
        save_path: Path,
        vec_env: Optional[VecNormalize] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model checkpoint with metadata.

        Args:
            model: Trained SB3 model
            save_path: Path to save model (without extension)
            vec_env: VecNormalize wrapper to save stats
            metadata: Additional metadata to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_path.with_suffix('.zip')
        model.save(model_path)
        logger.info(f"Saved model to {model_path}")

        # Save VecNormalize if available
        if vec_env is not None:
            vecnorm_path = save_path.parent / f"{save_path.stem}_vecnormalize.pkl"
            vec_env.save(vecnorm_path)
            logger.info(f"Saved VecNormalize to {vecnorm_path}")

        # Create and save metadata
        checkpoint_metadata = self._create_metadata(model, metadata)
        metadata_path = save_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        env: Any,
        verbose: int = 1,
    ) -> Tuple[BaseAlgorithm, Optional[VecNormalize]]:
        """
        Load model checkpoint with VecNormalize stats.

        Args:
            checkpoint_path: Path to model.zip file
            env: Environment to wrap (will be wrapped with VecNormalize)
            verbose: Verbosity level

        Returns:
            (model, vec_env) tuple
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load metadata if available
        metadata_path = checkpoint_path.with_suffix('.metadata.json')
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")

            # Import custom classes if needed
            self._import_custom_classes(metadata)
        else:
            logger.warning(f"No metadata found for {checkpoint_path}, attempting to load without it")

        # Prepare custom_objects for loading
        custom_objects = self._get_custom_objects(metadata)

        # Load model
        if verbose >= 1:
            logger.info(f"Loading model from {checkpoint_path}")

        try:
            if custom_objects:
                model = self._load_with_custom_objects(checkpoint_path, custom_objects)
            else:
                # Determine algorithm from metadata or path
                algo_class = self._infer_algorithm_class(checkpoint_path, metadata)
                model = algo_class.load(checkpoint_path)

            if verbose >= 1:
                logger.info(f"✓ Model loaded successfully")
                logger.info(f"  Policy: {type(model.policy).__name__}")
                logger.info(f"  Device: {model.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Wrap environment with VecNormalize
        vec_env = self._wrap_env_with_vecnormalize(env, model)

        # Load VecNormalize stats
        vecnorm_loaded = self._load_vecnormalize_stats(checkpoint_path, vec_env, verbose)

        if not vecnorm_loaded and verbose >= 1:
            logger.warning("⚠️  VecNormalize stats not loaded - predictions may be suboptimal")

        return model, vec_env

    def load_best_model(
        self,
        env: Any,
        checkpoint_dir: Optional[Path] = None,
        verbose: int = 1,
    ) -> Tuple[BaseAlgorithm, Optional[VecNormalize]]:
        """
        Load the best model from checkpoint directory.

        Args:
            env: Environment to wrap
            checkpoint_dir: Override checkpoint directory
            verbose: Verbosity level

        Returns:
            (model, vec_env) tuple
        """
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.checkpoint_dir

        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided")

        # Look for best_model
        best_model_path = checkpoint_dir / "best_model" / "best_model.zip"

        if best_model_path.exists():
            if verbose >= 1:
                logger.info(f"Loading best model from {best_model_path}")
            return self.load_checkpoint(best_model_path, env, verbose)

        # Fallback: find latest checkpoint
        checkpoint_files = list((checkpoint_dir / "checkpoints").glob("rl_model_*_steps.zip"))
        if checkpoint_files:
            # Sort by step number
            checkpoint_files.sort(key=lambda p: int(p.stem.split('_')[2]))
            latest = checkpoint_files[-1]
            if verbose >= 1:
                logger.warning(f"best_model not found, using latest checkpoint: {latest}")
            return self.load_checkpoint(latest, env, verbose)

        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    def list_checkpoints(
        self,
        checkpoint_dir: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of checkpoint info dicts
        """
        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.checkpoint_dir

        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided")

        checkpoints = []

        # Check best_model
        best_model = checkpoint_dir / "best_model" / "best_model.zip"
        if best_model.exists():
            checkpoints.append({
                'path': best_model,
                'type': 'best',
                'metadata': self._load_metadata(best_model),
            })

        # List periodic checkpoints
        checkpoint_files = sorted(
            (checkpoint_dir / "checkpoints").glob("rl_model_*_steps.zip"),
            key=lambda p: int(p.stem.split('_')[2])
        )

        for ckpt in checkpoint_files:
            checkpoints.append({
                'path': ckpt,
                'type': 'periodic',
                'steps': int(ckpt.stem.split('_')[2]),
                'metadata': self._load_metadata(ckpt),
            })

        return checkpoints

    # ========================================================================
    # Internal Methods
    # ========================================================================

    def _create_metadata(
        self,
        model: BaseAlgorithm,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create metadata dict for checkpoint"""
        metadata = {
            'version': self.METADATA_VERSION,
            'timestamp': datetime.now().isoformat(),
            'algorithm': type(model).__name__,
            'policy_class': {
                'module': type(model.policy).__module__,
                'name': type(model.policy).__name__,
            },
            'observation_space': {
                'shape': model.observation_space.shape,
                'dtype': str(model.observation_space.dtype),
            },
            'action_space': {
                'type': type(model.action_space).__name__,
            },
            'versions': {
                'stable_baselines3': sb3.__version__,
                'torch': torch.__version__,
                'numpy': np.__version__,
                'python': sys.version,
            },
        }

        # Add action space details
        if hasattr(model.action_space, 'n'):
            metadata['action_space']['n'] = int(model.action_space.n)
        elif hasattr(model.action_space, 'shape'):
            metadata['action_space']['shape'] = model.action_space.shape

        # Check for custom feature extractor
        if hasattr(model.policy, 'features_extractor'):
            extractor = model.policy.features_extractor
            metadata['feature_extractor'] = {
                'module': type(extractor).__module__,
                'name': type(extractor).__name__,
            }

        # Add custom metadata
        if custom_metadata:
            metadata['custom'] = custom_metadata

        return metadata

    def _import_custom_classes(self, metadata: Optional[Dict[str, Any]]) -> None:
        """Import custom policy and feature extractor classes"""
        if not metadata:
            return

        # Import policy class
        if 'policy_class' in metadata:
            policy_info = metadata['policy_class']
            module_name = policy_info['module']
            class_name = policy_info['name']

            # Only import if it's a custom class
            if 'rl_trading_lab' in module_name:
                try:
                    module = importlib.import_module(module_name)
                    policy_class = getattr(module, class_name)
                    logger.info(f"Imported custom policy: {class_name}")
                except Exception as e:
                    logger.warning(f"Could not import policy {class_name}: {e}")

        # Import feature extractor
        if 'feature_extractor' in metadata:
            extractor_info = metadata['feature_extractor']
            module_name = extractor_info['module']
            class_name = extractor_info['name']

            if 'rl_trading_lab' in module_name:
                try:
                    module = importlib.import_module(module_name)
                    extractor_class = getattr(module, class_name)
                    logger.info(f"Imported custom feature extractor: {class_name}")
                except Exception as e:
                    logger.warning(f"Could not import extractor {class_name}: {e}")

    def _get_custom_objects(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get custom_objects dict for SB3 loading"""
        custom_objects = {}

        if not metadata:
            # Fallback: try common custom classes
            try:
                from rl_trading_lab.models import TransformerActorCriticPolicy, TransformerFeatureExtractor
                custom_objects['TransformerActorCriticPolicy'] = TransformerActorCriticPolicy
                custom_objects['TransformerFeatureExtractor'] = TransformerFeatureExtractor
            except ImportError:
                pass
            return custom_objects

        # Import based on metadata
        if 'policy_class' in metadata:
            policy_info = metadata['policy_class']
            if 'rl_trading_lab' in policy_info['module']:
                try:
                    module = importlib.import_module(policy_info['module'])
                    custom_objects[policy_info['name']] = getattr(module, policy_info['name'])
                except Exception:
                    pass

        if 'feature_extractor' in metadata:
            extractor_info = metadata['feature_extractor']
            if 'rl_trading_lab' in extractor_info['module']:
                try:
                    module = importlib.import_module(extractor_info['module'])
                    custom_objects[extractor_info['name']] = getattr(module, extractor_info['name'])
                except Exception:
                    pass

        return custom_objects

    def _load_with_custom_objects(self, path: Path, custom_objects: Dict) -> BaseAlgorithm:
        """Load model with custom_objects"""
        # Try to determine algorithm
        algo_class = self._infer_algorithm_class(path)
        return algo_class.load(path, custom_objects=custom_objects)

    def _infer_algorithm_class(
        self,
        path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Infer SB3 algorithm class from path or metadata"""
        # From metadata
        if metadata and 'algorithm' in metadata:
            algo_name = metadata['algorithm']
            return getattr(sb3, algo_name)

        # From path
        path_str = str(path)
        for algo in ['PPO', 'A2C', 'DQN', 'SAC', 'TD3']:
            if algo in path_str:
                return getattr(sb3, algo)

        # Default to PPO
        logger.warning("Could not infer algorithm, defaulting to PPO")
        return sb3.PPO

    def _wrap_env_with_vecnormalize(self, env: Any, model: BaseAlgorithm) -> VecNormalize:
        """Wrap environment with VecNormalize to match training"""
        # Wrap with Monitor
        monitored_env = Monitor(env)
        env_func = lambda e=monitored_env: e
        vec_env = DummyVecEnv([env_func])

        # Wrap with VecNormalize
        gamma = model.gamma if hasattr(model, 'gamma') else 0.99
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize during eval
            clip_obs=10.0,
            training=False,
            gamma=gamma,
        )

        return vec_env

    def _load_vecnormalize_stats(
        self,
        checkpoint_path: Path,
        vec_env: VecNormalize,
        verbose: int = 1
    ) -> bool:
        """Load VecNormalize stats from various possible locations"""
        # Try multiple naming conventions
        possible_paths = [
            # Same directory as model, with _vecnormalize suffix
            checkpoint_path.parent / f"{checkpoint_path.stem}_vecnormalize.pkl",
            # CheckpointCallback naming: rl_model_vecnormalize_{steps}_steps.pkl
            checkpoint_path.parent / checkpoint_path.name.replace('rl_model_', 'rl_model_vecnormalize_').replace('.zip', '.pkl'),
            # best_model directory
            checkpoint_path.parent / "vecnormalize.pkl",
        ]

        for norm_path in possible_paths:
            if norm_path.exists():
                try:
                    vec_env = VecNormalize.load(norm_path, vec_env)
                    if verbose >= 1:
                        logger.info(f"✓ Loaded VecNormalize from {norm_path.name}")
                    return True
                except Exception as e:
                    logger.warning(f"Could not load VecNormalize from {norm_path}: {e}")

        return False

    def _load_metadata(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata for a checkpoint"""
        metadata_path = checkpoint_path.with_suffix('.metadata.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
