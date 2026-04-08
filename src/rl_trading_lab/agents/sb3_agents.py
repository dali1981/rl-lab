"""
Compatibility facade for legacy Trainer import paths.

Authoritative trainer orchestration now lives in:
- rl_trading_lab.agents.trainer.Trainer
- rl_trading_lab.agents.trainer_factory.TrainerFactory
"""

from rl_trading_lab.agents.trainer import MASKABLE_AVAILABLE, Trainer

__all__ = ["Trainer", "MASKABLE_AVAILABLE"]
