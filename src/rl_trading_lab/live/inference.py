"""
Model inference engine for live trading.

Loads trained RL models and makes real-time predictions on incoming features.
"""

import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
except ImportError:
    PPO = A2C = DQN = VecNormalize = DummyVecEnv = None

logger = logging.getLogger(__name__)


class ModelInferenceEngine:
    """
    Loads and runs trained RL models for real-time inference.

    Supports PPO, A2C, and DQN models with optional VecNormalize wrapper
    for observation normalization.

    Example:
        >>> engine = ModelInferenceEngine(
        ...     model_path="checkpoints/PPO_returns_20251028_143659/best_model.zip",
        ...     vecnormalize_path="checkpoints/PPO_returns_20251028_143659/vecnormalize.pkl"
        ... )
        >>>
        >>> # Predict action from features
        >>> action, confidence = engine.predict(features)
        >>> print(f"Action: {action}, Confidence: {confidence}")
    """

    # Action mapping (assuming 0=HOLD, 1=BUY, 2=SELL)
    ACTION_NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

    def __init__(
        self,
        model_path: str,
        vecnormalize_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to trained model (.zip file)
            vecnormalize_path: Path to VecNormalize wrapper (.pkl file)
            model_type: Model type ("PPO", "A2C", "DQN"). Auto-detected if None
        """
        if PPO is None:
            raise ImportError(
                "stable-baselines3 is required. Install with: uv add stable-baselines3"
            )

        self.model_path = Path(model_path)
        self.vecnormalize_path = Path(vecnormalize_path) if vecnormalize_path else None
        self.model_type = model_type

        # Load model
        self.model = None
        self.vecnormalize = None
        self._load_model()

        # Prediction statistics
        self.total_predictions = 0
        self.action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL

        logger.info(
            f"Initialized ModelInferenceEngine with {self.model_type} "
            f"from {self.model_path.name}"
        )

    def _load_model(self):
        """Load the trained model and normalization wrapper."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Auto-detect model type from path if not specified
        if self.model_type is None:
            path_str = str(self.model_path)
            if "PPO" in path_str or "ppo" in path_str:
                self.model_type = "PPO"
            elif "A2C" in path_str or "a2c" in path_str:
                self.model_type = "A2C"
            elif "DQN" in path_str or "dqn" in path_str:
                self.model_type = "DQN"
            else:
                # Default to PPO
                logger.warning("Could not auto-detect model type, defaulting to PPO")
                self.model_type = "PPO"

        # Load model
        try:
            if self.model_type == "PPO":
                self.model = PPO.load(self.model_path)
            elif self.model_type == "A2C":
                self.model = A2C.load(self.model_path)
            elif self.model_type == "DQN":
                self.model = DQN.load(self.model_path)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")

            logger.info(f"Loaded {self.model_type} model from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Load VecNormalize wrapper if provided
        if self.vecnormalize_path is not None:
            if self.vecnormalize_path.exists():
                try:
                    with open(self.vecnormalize_path, "rb") as f:
                        self.vecnormalize = pickle.load(f)
                    logger.info(f"Loaded VecNormalize from {self.vecnormalize_path}")
                except Exception as e:
                    logger.warning(f"Could not load VecNormalize: {e}")
                    self.vecnormalize = None
            else:
                logger.warning(f"VecNormalize path specified but not found: {self.vecnormalize_path}")
                self.vecnormalize = None
        else:
            logger.info("VecNormalize not used (path not provided)")

    def predict(
        self,
        features: pd.DataFrame,
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """
        Predict action from features.

        Args:
            features: DataFrame with feature columns
            deterministic: Use deterministic policy (True for live trading)

        Returns:
            Tuple of (action, confidence)
            - action: Integer action (0=HOLD, 1=BUY, 2=SELL)
            - confidence: Confidence score (probability or Q-value)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Convert features to numpy array
            obs = self._prepare_observation(features)

            # Normalize if VecNormalize is available
            if self.vecnormalize is not None:
                obs = self.vecnormalize.normalize_obs(obs)

            # Predict action
            action, _states = self.model.predict(obs, deterministic=deterministic)

            # Handle both scalar and array actions
            if isinstance(action, np.ndarray):
                action = int(action[0])
            else:
                action = int(action)

            # Compute confidence (model-dependent)
            confidence = self._compute_confidence(obs, action)

            # Update statistics
            self.total_predictions += 1
            self.action_counts[action] = self.action_counts.get(action, 0) + 1

            return action, confidence

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def _prepare_observation(self, features: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation array from features DataFrame.

        Args:
            features: DataFrame with features

        Returns:
            Numpy array ready for model input
        """
        # Drop non-feature columns
        non_feature_cols = ["timestamp", "symbol"]
        feature_cols = [col for col in features.columns if col not in non_feature_cols]

        # Extract feature values
        obs = features[feature_cols].values

        # Ensure 2D shape (batch_size, features)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        return obs.astype(np.float32)

    def _compute_confidence(self, obs: np.ndarray, action: int) -> float:
        """
        Compute confidence score for the predicted action.

        Args:
            obs: Observation array
            action: Predicted action

        Returns:
            Confidence score (0-1)
        """
        try:
            # For policy-based methods (PPO, A2C), get action probabilities
            if self.model_type in ["PPO", "A2C"]:
                # Get action probabilities
                obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
                distribution = self.model.policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.detach().cpu().numpy()[0]
                return float(probs[action])

            # For DQN, use Q-values
            elif self.model_type == "DQN":
                q_values = self.model.q_net(
                    self.model.policy.obs_to_tensor(obs)[0]
                ).detach().cpu().numpy()[0]
                # Normalize Q-values to 0-1 range using softmax
                exp_q = np.exp(q_values - np.max(q_values))
                probs = exp_q / exp_q.sum()
                return float(probs[action])

        except Exception as e:
            logger.warning(f"Could not compute confidence: {e}")

        # Default confidence
        return 0.5

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        return self.ACTION_NAMES.get(action, "UNKNOWN")

    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        return {
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "total_predictions": self.total_predictions,
            "action_counts": self.action_counts,
            "action_distribution": {
                self.get_action_name(k): v / max(self.total_predictions, 1)
                for k, v in self.action_counts.items()
            },
        }


class MultiSymbolInferenceEngine:
    """
    Manages inference for multiple symbols with different models.

    Example:
        >>> engine = MultiSymbolInferenceEngine()
        >>> engine.add_model(
        ...     "BTCUSDT",
        ...     "checkpoints/PPO_returns_20251028_143659/best_model.zip",
        ...     "checkpoints/PPO_returns_20251028_143659/vecnormalize.pkl"
        ... )
        >>> engine.add_model(
        ...     "ETHUSDT",
        ...     "checkpoints/A2C_returns_20251028_143912/best_model.zip"
        ... )
        >>>
        >>> # Predict for specific symbol
        >>> action, confidence = engine.predict("BTCUSDT", btc_features)
    """

    def __init__(self):
        self.engines: Dict[str, ModelInferenceEngine] = {}

    def add_model(
        self,
        symbol: str,
        model_path: str,
        vecnormalize_path: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """Add a model for a specific symbol."""
        engine = ModelInferenceEngine(
            model_path=model_path,
            vecnormalize_path=vecnormalize_path,
            model_type=model_type,
        )
        self.engines[symbol] = engine
        logger.info(f"Added model for {symbol}")

    def predict(
        self,
        symbol: str,
        features: pd.DataFrame,
        deterministic: bool = True,
    ) -> Tuple[int, float]:
        """Predict action for a specific symbol."""
        if symbol not in self.engines:
            raise ValueError(f"No model loaded for symbol: {symbol}")

        return self.engines[symbol].predict(features, deterministic)

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        return ModelInferenceEngine.ACTION_NAMES.get(action, "UNKNOWN")

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all models."""
        return {
            symbol: engine.get_stats()
            for symbol, engine in self.engines.items()
        }

    def has_model(self, symbol: str) -> bool:
        """Check if a model exists for the symbol."""
        return symbol in self.engines