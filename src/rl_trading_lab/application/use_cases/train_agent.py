"""
TrainAgentUseCase - Use case for training RL agents.

This use case orchestrates the entire training workflow:
1. Create training and evaluation environments
2. Configure the RL agent
3. Set up checkpointing and experiment tracking
4. Execute training loop
5. Save final model

Per Martin (Clean Architecture), use cases contain
application-specific business rules.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl_trading_lab.application.ports.experiment_tracker import (
    ExperimentTrackerPort,
    NoOpExperimentTracker,
)
from rl_trading_lab.application.services.agent_service import AgentService
from rl_trading_lab.application.services.checkpoint_service import CheckpointService
from rl_trading_lab.application.services.environment_service import EnvironmentService

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Result of the training use case.

    Contains paths to saved models and training metrics.
    """

    final_model_path: Path
    total_timesteps: int
    best_model_path: Optional[Path] = None
    training_time_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TrainingResult(timesteps={self.total_timesteps}, "
            f"final_model={self.final_model_path.name})"
        )


@dataclass
class TrainingConfig:
    """Configuration for the training use case."""

    # Data
    data_path: str
    observation_features: List[str]

    # Environment
    initial_balance: float = 10000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_pct: float = 0.95
    lookback_window: int = 20
    min_episode_length: int = 100
    reward_type: str = "returns"
    max_drawdown_pct: float = 0.30
    min_portfolio_pct: float = 0.20

    # Agent
    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    device: str = "auto"

    # Training
    total_timesteps: int = 100000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    save_freq: int = 50000

    # Paths
    save_path: str = "checkpoints"
    tensorboard_log: Optional[str] = None

    # Experiment
    run_name: Optional[str] = None
    progress_bar: bool = True


class TrainAgentUseCase:
    """
    Use case: Train an RL agent on trading environment.

    This use case orchestrates:
    - Environment creation (train and eval)
    - Agent configuration and creation
    - Training with callbacks for checkpointing
    - Experiment tracking (if configured)
    - Final model saving

    Example:
        >>> from rl_trading_lab.application.use_cases import TrainAgentUseCase
        >>> from rl_trading_lab.application.ports.data_loader import DataLoaderPort
        >>>
        >>> # Setup dependencies
        >>> data_loader: DataLoaderPort = ...
        >>> env_service = EnvironmentService(data_loader)
        >>> agent_service = AgentService()
        >>> checkpoint_service = CheckpointService(save_path=Path("models"))
        >>>
        >>> # Create use case
        >>> train_use_case = TrainAgentUseCase(
        ...     environment_service=env_service,
        ...     agent_service=agent_service,
        ...     checkpoint_service=checkpoint_service,
        ... )
        >>>
        >>> # Execute training
        >>> config = TrainingConfig(
        ...     data_path="data/btc_features.parquet",
        ...     observation_features=["close", "volume", "rsi"],
        ...     total_timesteps=100000,
        ... )
        >>> result = train_use_case.execute(config)
        >>> print(f"Model saved to: {result.final_model_path}")
    """

    def __init__(
        self,
        environment_service: EnvironmentService,
        agent_service: AgentService,
        checkpoint_service: CheckpointService,
        experiment_tracker: Optional[ExperimentTrackerPort] = None,
    ):
        """
        Initialize the training use case.

        Args:
            environment_service: Service for creating environments
            agent_service: Service for agent management
            checkpoint_service: Service for model persistence
            experiment_tracker: Optional tracker for experiment logging
        """
        self._env_service = environment_service
        self._agent_service = agent_service
        self._checkpoint_service = checkpoint_service
        self._tracker = experiment_tracker or NoOpExperimentTracker()

    def execute(self, config: TrainingConfig) -> TrainingResult:
        """
        Execute the training use case.

        Args:
            config: Training configuration

        Returns:
            TrainingResult with paths and metrics
        """
        import time

        start_time = time.time()

        # Start experiment tracking
        run_name = config.run_name or f"{config.algorithm}_{config.reward_type}"
        self._tracker.start_run(
            run_name=run_name,
            params=self._extract_params(config),
        )

        try:
            # 1. Create environments
            logger.info("Creating training environment...")
            train_env = self._env_service.create_training_env(
                data_path=config.data_path,
                observation_features=config.observation_features,
                initial_balance=config.initial_balance,
                commission_rate=config.commission_rate,
                slippage_rate=config.slippage_rate,
                max_position_pct=config.max_position_pct,
                lookback_window=config.lookback_window,
                min_episode_length=config.min_episode_length,
                reward_type=config.reward_type,
                max_drawdown_pct=config.max_drawdown_pct,
                min_portfolio_pct=config.min_portfolio_pct,
                randomize_start=True,
            )

            logger.info("Creating evaluation environment...")
            eval_env = self._env_service.create_eval_env(
                data_path=config.data_path,
                observation_features=config.observation_features,
                initial_balance=config.initial_balance,
                commission_rate=config.commission_rate,
                slippage_rate=config.slippage_rate,
                max_position_pct=config.max_position_pct,
                lookback_window=config.lookback_window,
                min_episode_length=config.min_episode_length,
                reward_type=config.reward_type,
                max_drawdown_pct=config.max_drawdown_pct,
                min_portfolio_pct=config.min_portfolio_pct,
                randomize_start=True,
            )

            # 2. Create agent
            logger.info(f"Creating {config.algorithm} agent...")
            agent, train_vec_env = self._agent_service.create_agent(
                algorithm=config.algorithm,
                env=train_env,
                hyperparameters=config.hyperparameters,
                policy=config.policy,
                device=config.device,
                tensorboard_log=config.tensorboard_log,
            )

            # Wrap eval env with normalization from train env
            eval_vec_env = self._agent_service.wrap_eval_environment(
                env=eval_env,
                training_vec_env=train_vec_env,
            )

            # 3. Create callbacks
            callbacks = self._checkpoint_service.create_training_callbacks(
                eval_env=eval_vec_env,
                eval_freq=config.eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                save_freq=config.save_freq,
            )

            # 4. Train
            logger.info(f"Starting training for {config.total_timesteps} timesteps...")
            agent.learn(
                total_timesteps=config.total_timesteps,
                callback=callbacks,
                progress_bar=config.progress_bar,
            )

            # 5. Save final model
            final_path = self._checkpoint_service.save_final_model(
                agent=agent,
                vec_env=train_vec_env,
                metadata={
                    "algorithm": config.algorithm,
                    "total_timesteps": config.total_timesteps,
                    "reward_type": config.reward_type,
                },
            )

            # Calculate training time
            training_time = time.time() - start_time

            # Log final metrics
            self._tracker.log_metrics({
                "training_time_seconds": training_time,
                "total_timesteps": config.total_timesteps,
            })

            logger.info(f"Training completed in {training_time:.1f}s")

            return TrainingResult(
                final_model_path=final_path,
                total_timesteps=config.total_timesteps,
                best_model_path=self._checkpoint_service.best_model_path,
                training_time_seconds=training_time,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._tracker.end_run(status="FAILED")
            raise

        finally:
            self._tracker.end_run()

    def _extract_params(self, config: TrainingConfig) -> Dict[str, Any]:
        """Extract parameters for experiment tracking."""
        params = {
            "algorithm": config.algorithm,
            "policy": config.policy,
            "total_timesteps": config.total_timesteps,
            "reward_type": config.reward_type,
            "initial_balance": config.initial_balance,
            "commission_rate": config.commission_rate,
            "lookback_window": config.lookback_window,
            "max_drawdown_pct": config.max_drawdown_pct,
        }

        # Add hyperparameters (flattened)
        for key, value in config.hyperparameters.items():
            if not isinstance(value, (dict, list)):
                params[f"agent.{key}"] = value

        return params
