"""Composition-root helpers for the canonical training entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from rl_trading_lab.application.ports.data_loader import ParquetDataLoader
from rl_trading_lab.application.services.agent_service import AgentService
from rl_trading_lab.application.services.checkpoint_service import CheckpointService
from rl_trading_lab.application.services.environment_service import EnvironmentService
from rl_trading_lab.application.use_cases.train_agent import (
    TrainAgentUseCase,
    TrainingConfig as UseCaseTrainingConfig,
)
from rl_trading_lab.config import RootConfig
from rl_trading_lab.infrastructure.adapters.mlflow_tracker import create_mlflow_tracker


def _drop_none_values(value: Any) -> Any:
    """Recursively drop None values from nested dict/list structures."""
    if isinstance(value, dict):
        return {k: _drop_none_values(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none_values(v) for v in value if v is not None]
    return value


def _extract_agent_settings(config: RootConfig) -> Tuple[str, Dict[str, Any]]:
    """Extract policy and algorithm hyperparameters for the use case."""
    hyperparameters = config.agent.hyperparameters.model_dump()
    policy = hyperparameters.pop("policy", "MlpPolicy")
    return policy, _drop_none_values(hyperparameters)


def build_training_use_case(config: RootConfig) -> TrainAgentUseCase:
    """Assemble the canonical training use case with concrete services."""
    data_loader = ParquetDataLoader(
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        required_columns=config.env.required_columns,
    )

    environment_service = EnvironmentService(data_loader=data_loader)

    agent_service = AgentService(
        vec_normalize_enabled=config.env.vec_normalize.enabled,
        norm_obs=config.env.vec_normalize.norm_obs,
        norm_reward=config.env.vec_normalize.norm_reward,
        clip_obs=config.env.vec_normalize.clip_obs,
        clip_reward=config.env.vec_normalize.clip_reward,
    )

    checkpoint_service = CheckpointService(
        save_path=Path(config.training.save_path),
        name_prefix=f"{config.agent.name.lower()}_model",
    )

    experiment_tracker = create_mlflow_tracker(
        tracking_uri=config.logging.mlflow.tracking_uri,
        experiment_name=config.logging.mlflow.experiment_name,
        enabled=config.logging.mlflow.enabled,
    )

    return TrainAgentUseCase(
        environment_service=environment_service,
        agent_service=agent_service,
        checkpoint_service=checkpoint_service,
        experiment_tracker=experiment_tracker,
    )


def to_use_case_training_config(config: RootConfig) -> UseCaseTrainingConfig:
    """Convert root config to TrainAgentUseCase config."""
    policy, hyperparameters = _extract_agent_settings(config)
    env = config.env.environment_params

    return UseCaseTrainingConfig(
        data_path=config.data.train_data_path,
        observation_features=config.observation.input_features,
        initial_balance=env.initial_balance,
        commission_rate=env.commission_rate,
        slippage_rate=env.slippage_rate,
        max_position_pct=env.max_position_pct,
        lookback_window=env.lookback_window,
        min_episode_length=env.min_episode_length,
        reward_type=env.reward_type,
        max_drawdown_pct=0.30,
        min_portfolio_pct=0.20,
        algorithm=config.agent.name,
        policy=policy,
        hyperparameters=hyperparameters,
        device=config.experiment.device,
        total_timesteps=config.training.total_timesteps,
        eval_freq=config.training.eval_freq,
        n_eval_episodes=config.training.n_eval_episodes,
        save_freq=config.training.save_freq,
        save_path=config.training.save_path,
        tensorboard_log=(
            config.logging.tensorboard.log_dir if config.logging.tensorboard.enabled else None
        ),
        run_name=config.experiment.run_name,
        progress_bar=config.logging.console.progress_bar,
    )
