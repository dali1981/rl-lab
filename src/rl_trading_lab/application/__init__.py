"""
Application Layer - Use cases and application services.

This layer orchestrates domain logic to fulfill use cases:
- Use Cases: Application-specific workflows (train agent, evaluate, backtest)
- Services: Application-level coordination (environment creation, agent management)
- Ports: Interfaces for external services (experiment tracking, data loading)

The application layer depends on the domain layer but not on infrastructure.
Infrastructure adapters implement the ports defined here.

Per Martin (Clean Architecture):
"The application-specific rules of the application. The dependencies of
elements in this layer should point inward, toward the domain layer."
"""

from rl_trading_lab.application.use_cases import (
    TrainAgentUseCase,
    TrainingResult,
    EvaluateAgentUseCase,
    EvaluationResult,
)

__all__ = [
    # Use Cases
    "TrainAgentUseCase",
    "TrainingResult",
    "EvaluateAgentUseCase",
    "EvaluationResult",
]
