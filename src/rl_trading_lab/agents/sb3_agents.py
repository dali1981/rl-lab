"""
Compatibility facade for legacy Trainer import paths.

Authoritative trainer orchestration now lives in:
- rl_trading_lab.agents.trainer.Trainer
- rl_trading_lab.agents.trainer_factory.TrainerFactory
"""

from __future__ import annotations

import warnings

from rl_trading_lab.agents.trainer import MASKABLE_AVAILABLE, Trainer as _Trainer


class Trainer(_Trainer):
    """Compatibility shim that forwards to the authoritative trainer."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "rl_trading_lab.agents.sb3_agents.Trainer is deprecated; "
            "use rl_trading_lab.agents.trainer.Trainer or TrainerFactory",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["Trainer", "MASKABLE_AVAILABLE"]
