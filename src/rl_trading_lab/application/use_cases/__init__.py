"""
Use Cases - Application-specific workflows.

Use cases orchestrate the flow of data to and from domain entities,
and direct those entities to use their domain logic to achieve
the goals of the use case.

Per Martin (Clean Architecture):
"The use cases of an application are the details that make the
application valuable. They know about the entities, but entities
don't know about them."

Use cases in this package:
- TrainAgentUseCase: Train an RL agent on trading environment
- EvaluateAgentUseCase: Evaluate a trained agent's performance
- RunBacktestUseCase: Run detailed backtest with metrics collection
"""

from rl_trading_lab.application.use_cases.train_agent import (
    TrainAgentUseCase,
    TrainingResult,
)
from rl_trading_lab.application.use_cases.evaluate_agent import (
    EvaluateAgentUseCase,
    EvaluationResult,
)

__all__ = [
    "TrainAgentUseCase",
    "TrainingResult",
    "EvaluateAgentUseCase",
    "EvaluationResult",
]
