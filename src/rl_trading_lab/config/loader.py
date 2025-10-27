"""Configuration loader that converts Hydra DictConfig to Pydantic models."""

from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from rl_trading_lab.config.main import RootConfig


def load_config(cfg: DictConfig) -> RootConfig:
    """Load and validate configuration from Hydra DictConfig.

    Args:
        cfg: Hydra DictConfig object loaded from YAML files

    Returns:
        RootConfig: Validated Pydantic config object

    Raises:
        ValidationError: If configuration is invalid or missing required fields
    """
    try:
        # Convert OmegaConf DictConfig to plain dict
        config_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

        # Validate and create Pydantic model
        config = RootConfig(**config_dict)

        return config

    except ValidationError as e:
        print("\n" + "=" * 80)
        print("CONFIGURATION VALIDATION ERROR")
        print("=" * 80)
        print("\nThe configuration is invalid. Please check your YAML files.")
        print("\nDetailed errors:")
        print(e)
        print("=" * 80 + "\n")
        raise
