"""Deterministic tests for canonical training entrypoint runtime mapping."""

from types import SimpleNamespace

from rl_trading_lab.runtime.training_entrypoint import _extract_agent_settings


class _DummyHyperparameters:
    def model_dump(self):
        return {
            "policy": "MlpPolicy",
            "learning_rate": 1e-4,
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": "ReLU",
                "features_extractor_class": None,
                "features_extractor_kwargs": None,
            },
            "clip_range_vf": None,
        }


def test_extract_agent_settings_strips_none_values():
    """Nested None values are removed before agent construction."""
    config = SimpleNamespace(
        agent=SimpleNamespace(hyperparameters=_DummyHyperparameters())
    )

    policy, hyperparameters = _extract_agent_settings(config)

    assert policy == "MlpPolicy"
    assert "policy" not in hyperparameters
    assert hyperparameters["learning_rate"] == 1e-4
    assert hyperparameters["policy_kwargs"] == {
        "net_arch": [256, 256],
        "activation_fn": "ReLU",
    }
    assert "clip_range_vf" not in hyperparameters
