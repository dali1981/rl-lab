"""
Live trading module for RL Trading Lab.

This module provides all components needed for live trading on Binance testnet/live:
- Real-time data streaming via WebSocket
- Dollar volume bar creation
- Feature computation with rolling windows
- Model inference for action prediction
- Order execution and portfolio management
- Safety guards and risk management

Example usage:
    >>> from rl_trading_lab.live import (
    ...     StreamConsumer,
    ...     FeatureComputer,
    ...     ModelInferenceEngine,
    ...     OrderExecutor,
    ...     PortfolioManager
    ... )
"""

from .stream_consumer import StreamConsumer, MultiSymbolStreamConsumer
from .feature_computer import FeatureComputer, MultiSymbolFeatureComputer
from .inference import ModelInferenceEngine, MultiSymbolInferenceEngine
from .executor import OrderExecutor, Action
from .portfolio import PortfolioManager
from .safety import SafetyGuard, CircuitBreakerState, ConnectionMonitor
from .dashboard import TradingDashboard

__all__ = [
    # Streaming
    "StreamConsumer",
    "MultiSymbolStreamConsumer",
    # Features
    "FeatureComputer",
    "MultiSymbolFeatureComputer",
    # Inference
    "ModelInferenceEngine",
    "MultiSymbolInferenceEngine",
    # Execution
    "OrderExecutor",
    "Action",
    # Portfolio
    "PortfolioManager",
    # Safety
    "SafetyGuard",
    "CircuitBreakerState",
    "ConnectionMonitor",
    # Dashboard
    "TradingDashboard",
]
