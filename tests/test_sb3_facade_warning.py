"""DAL-137 compatibility facade behavior checks."""

from __future__ import annotations

import warnings


def test_sb3_facade_trainer_emits_deprecation_warning(monkeypatch) -> None:
    from rl_trading_lab.agents import sb3_agents
    from rl_trading_lab.agents import trainer as trainer_module

    called = {"value": False}

    def _fake_init(self, *args, **kwargs):
        called["value"] = True

    monkeypatch.setattr(trainer_module.Trainer, "__init__", _fake_init)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        sb3_agents.Trainer()

    assert called["value"], "Facade Trainer must delegate initialization to authoritative Trainer"
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "sb3_agents.Trainer is deprecated" in str(w.message)
        for w in captured
    ), "Facade Trainer must emit DeprecationWarning on direct use"
