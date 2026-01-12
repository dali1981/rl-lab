"""
EnvironmentService - Application service for environment creation.

This service assembles the domain with appropriate services and wraps
it in the Gymnasium adapter for use with RL frameworks.

Per Fowler (PoEAA), application services coordinate domain objects
to perform application-specific operations.
"""

import logging
from typing import List, Optional

import gymnasium as gym

from rl_trading_lab.application.ports.data_loader import DataLoaderPort
from rl_trading_lab.domain.trading_domain import TradingDomain, TradingDomainConfig
from rl_trading_lab.domain.services.position_sizing import (
    FixedPercentagePositionSizing,
    PositionSizingService,
)
from rl_trading_lab.domain.services.reward_calculation import (
    ReturnsRewardCalculation,
    PnLRewardCalculation,
    RewardCalculationService,
)
from rl_trading_lab.domain.services.risk_management import (
    RiskLimits,
    RiskManagementService,
    StandardRiskManagement,
)
from rl_trading_lab.infrastructure.adapters.market_data_adapter import ParquetMarketDataAdapter
from rl_trading_lab.infrastructure.adapters.gym_adapter import GymTradingEnvAdapter

logger = logging.getLogger(__name__)


class EnvironmentService:
    """
    Application service for creating trading environments.

    Responsibilities:
    - Loads market data using DataLoaderPort
    - Assembles TradingDomain with appropriate services
    - Wraps domain in GymTradingEnvAdapter for RL frameworks

    This service encapsulates the complexity of:
    - Choosing appropriate domain services based on configuration
    - Proper initialization and wiring of components
    - Consistent environment creation for train/eval/test

    Example:
        >>> data_loader = ParquetDataLoader(val_split=0.1, test_split=0.1)
        >>> env_service = EnvironmentService(data_loader)
        >>>
        >>> train_env = env_service.create_training_env(
        ...     data_path=Path("data/btc_features.parquet"),
        ...     observation_features=["close", "volume", "rsi"],
        ...     initial_balance=10000,
        ...     reward_type="returns",
        ... )
    """

    def __init__(self, data_loader: DataLoaderPort):
        """
        Initialize the environment service.

        Args:
            data_loader: Port for loading market data
        """
        self._data_loader = data_loader

    def create_training_env(
        self,
        data_path: str,
        observation_features: List[str],
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_pct: float = 0.95,
        lookback_window: int = 20,
        min_episode_length: int = 100,
        reward_type: str = "returns",
        max_drawdown_pct: float = 0.30,
        min_portfolio_pct: float = 0.20,
        randomize_start: bool = True,
        position_sizing: Optional[PositionSizingService] = None,
        reward_calculator: Optional[RewardCalculationService] = None,
        risk_manager: Optional[RiskManagementService] = None,
    ) -> gym.Env:
        """
        Create a training environment with randomized starts.

        Args:
            data_path: Path to market data file
            observation_features: Feature columns for observations
            initial_balance: Starting portfolio balance
            commission_rate: Trading commission rate
            slippage_rate: Slippage rate
            max_position_pct: Maximum position as fraction of cash
            lookback_window: Number of historical bars in observations
            min_episode_length: Minimum episode length for random starts
            reward_type: Type of reward ("returns" or "pnl")
            max_drawdown_pct: Maximum drawdown before termination
            min_portfolio_pct: Minimum portfolio value as % of initial
            randomize_start: Whether to randomize episode start points
            position_sizing: Custom position sizing service
            reward_calculator: Custom reward calculation service
            risk_manager: Custom risk management service

        Returns:
            Gymnasium-compatible training environment
        """
        return self._create_env(
            data_path=data_path,
            mode="train",
            observation_features=observation_features,
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_pct=max_position_pct,
            lookback_window=lookback_window,
            min_episode_length=min_episode_length,
            reward_type=reward_type,
            max_drawdown_pct=max_drawdown_pct,
            min_portfolio_pct=min_portfolio_pct,
            randomize_start=randomize_start,
            position_sizing=position_sizing,
            reward_calculator=reward_calculator,
            risk_manager=risk_manager,
        )

    def create_eval_env(
        self,
        data_path: str,
        observation_features: List[str],
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_pct: float = 0.95,
        lookback_window: int = 20,
        min_episode_length: int = 100,
        reward_type: str = "returns",
        max_drawdown_pct: float = 0.30,
        min_portfolio_pct: float = 0.20,
        randomize_start: bool = True,
    ) -> gym.Env:
        """
        Create an evaluation environment.

        Similar to training but uses validation data split.
        Randomization is typically enabled for diverse evaluation.
        """
        return self._create_env(
            data_path=data_path,
            mode="eval",
            observation_features=observation_features,
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_pct=max_position_pct,
            lookback_window=lookback_window,
            min_episode_length=min_episode_length,
            reward_type=reward_type,
            max_drawdown_pct=max_drawdown_pct,
            min_portfolio_pct=min_portfolio_pct,
            randomize_start=randomize_start,
        )

    def create_test_env(
        self,
        data_path: str,
        observation_features: List[str],
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        max_position_pct: float = 0.95,
        lookback_window: int = 20,
        min_episode_length: int = 100,
        reward_type: str = "returns",
        max_drawdown_pct: float = 0.30,
        min_portfolio_pct: float = 0.20,
    ) -> gym.Env:
        """
        Create a test environment.

        Uses test data split with deterministic start (no randomization)
        for reproducible evaluation.
        """
        return self._create_env(
            data_path=data_path,
            mode="test",
            observation_features=observation_features,
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_pct=max_position_pct,
            lookback_window=lookback_window,
            min_episode_length=min_episode_length,
            reward_type=reward_type,
            max_drawdown_pct=max_drawdown_pct,
            min_portfolio_pct=min_portfolio_pct,
            randomize_start=False,  # Deterministic for testing
        )

    def _create_env(
        self,
        data_path: str,
        mode: str,
        observation_features: List[str],
        initial_balance: float,
        commission_rate: float,
        slippage_rate: float,
        max_position_pct: float,
        lookback_window: int,
        min_episode_length: int,
        reward_type: str,
        max_drawdown_pct: float,
        min_portfolio_pct: float,
        randomize_start: bool,
        position_sizing: Optional[PositionSizingService] = None,
        reward_calculator: Optional[RewardCalculationService] = None,
        risk_manager: Optional[RiskManagementService] = None,
    ) -> gym.Env:
        """
        Internal method to create an environment.

        Assembles all components and returns a wrapped gym.Env.
        """
        # Load data for mode
        df = self._data_loader.load(data_path, mode=mode)
        logger.info(f"Loaded {mode} data: {len(df)} rows")

        # Create market data adapter
        market_data = ParquetMarketDataAdapter(df)

        # Create or use provided domain services
        pos_sizing = position_sizing or FixedPercentagePositionSizing()

        if reward_calculator:
            reward_calc = reward_calculator
        elif reward_type == "returns":
            reward_calc = ReturnsRewardCalculation()
        else:
            reward_calc = PnLRewardCalculation()

        risk_mgmt = risk_manager or StandardRiskManagement(
            RiskLimits(
                max_drawdown_pct=max_drawdown_pct,
                min_portfolio_pct=min_portfolio_pct,
            )
        )

        # Create domain configuration
        config = TradingDomainConfig(
            initial_balance=initial_balance,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            max_position_pct=max_position_pct,
            lookback_window=lookback_window,
            min_episode_length=min_episode_length,
        )

        # Create trading domain
        domain = TradingDomain(
            market_data=market_data,
            observation_features=observation_features,
            config=config,
            position_sizing=pos_sizing,
            reward_calculator=reward_calc,
            risk_manager=risk_mgmt,
        )

        # Wrap in gym adapter
        env = GymTradingEnvAdapter(
            domain=domain,
            randomize_start=randomize_start,
            min_episode_length=min_episode_length,
        )

        logger.info(
            f"Created {mode} environment: "
            f"features={len(observation_features)}, "
            f"lookback={lookback_window}, "
            f"reward={reward_type}, "
            f"randomize={randomize_start}"
        )

        return env
