#!/usr/bin/env python
"""
Canonical training entrypoint for RL Trading Lab.

This script is the canonical core runtime path and delegates orchestration to
application-layer use cases.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from rich.console import Console

from rl_trading_lab.application.use_cases.train_agent import TrainAgentUseCase
from rl_trading_lab.config import load_config
from rl_trading_lab.runtime import build_training_use_case, to_use_case_training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run canonical training pipeline through the application use case."""
    config = load_config(cfg)

    console.print("[bold blue]RL Trading Lab[/bold blue]")
    console.print(f"Agent: [cyan]{config.agent.name}[/cyan]")
    console.print(
        f"Reward: [cyan]{config.env.environment_params.reward_type}[/cyan], "
        f"Timesteps: [cyan]{config.training.total_timesteps}[/cyan], "
        f"Device: [cyan]{config.experiment.device}[/cyan]\n"
    )

    use_case = build_training_use_case(config)
    if not isinstance(use_case, TrainAgentUseCase):
        raise TypeError("Canonical training runtime must resolve to TrainAgentUseCase")

    training_config = to_use_case_training_config(config)

    result = use_case.execute(training_config)

    console.print("[bold green]Training complete.[/bold green]")
    console.print(f"Final model: [green]{result.final_model_path}[/green]")
    if result.best_model_path:
        console.print(f"Best model: [green]{result.best_model_path}[/green]")


if __name__ == "__main__":
    main()
