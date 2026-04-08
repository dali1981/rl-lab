"""
Application Ports - Interfaces for external services.

Ports define the contracts that infrastructure adapters must implement.
This follows the Ports & Adapters (Hexagonal) architecture pattern.

Ports in this package:
- ExperimentTrackerPort: Interface for experiment tracking (MLflow, W&B)
- DataLoaderPort: Interface for loading market data
- FeatureEngineeringPort: Interface for feature transformation pipelines
"""

from rl_trading_lab.application.ports.experiment_tracker import ExperimentTrackerPort
from rl_trading_lab.application.ports.data_loader import DataLoaderPort
from rl_trading_lab.application.ports.feature_engineering import (
    FeatureEngineeringPort,
    PassthroughFeatures,
)
from rl_trading_lab.application.ports.environment_factories import (
    MarketDataAdapterFactory,
    EnvAdapterFactory,
)

__all__ = [
    "ExperimentTrackerPort",
    "DataLoaderPort",
    "FeatureEngineeringPort",
    "PassthroughFeatures",
    "MarketDataAdapterFactory",
    "EnvAdapterFactory",
]
