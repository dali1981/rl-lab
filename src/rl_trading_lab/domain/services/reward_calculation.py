"""
Reward Calculation Service - Domain service for calculating step rewards.

This service encapsulates the logic for computing rewards based on
portfolio value changes. Different reward formulations can significantly
impact agent learning and behavior.

Note: Sharpe ratio is NOT implemented as a step reward because:
- Requires many samples for meaningful variance estimation
- Short windows create non-stationary signals
- Agent would optimize short-term Sharpe, not episode Sharpe
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class RewardType(Enum):
    """Supported reward types."""

    RETURNS = "returns"
    PNL = "pnl"


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for reward calculation."""

    clip_min: float = -10.0
    clip_max: float = 10.0
    scale_factor: float = 1.0


class RewardCalculationService(Protocol):
    """
    Protocol for reward calculation strategies.

    The reward signal is critical for RL agent learning. Different
    formulations lead to different agent behaviors:
    - Returns: Percentage-based, scale-invariant
    - PnL: Absolute dollar amounts, can be dominated by large positions
    """

    def calculate(
        self,
        prev_value: float,
        current_value: float,
    ) -> float:
        """
        Calculate reward for a single step.

        Args:
            prev_value: Portfolio value at previous step
            current_value: Portfolio value at current step

        Returns:
            Reward value (clipped to configured bounds)
        """
        ...

    @property
    def reward_type(self) -> RewardType:
        """The type of reward this service calculates."""
        ...


class ReturnsRewardCalculation(RewardCalculationService):
    """
    Reward based on percentage returns.

    Formula: reward = (current_value - prev_value) / prev_value

    Advantages:
    - Scale-invariant (works regardless of portfolio size)
    - Bounded by nature (typically small percentages)
    - Intuitive interpretation

    Example:
        >>> calc = ReturnsRewardCalculation()
        >>> reward = calc.calculate(prev_value=10000, current_value=10100)
        >>> print(f"Reward: {reward:.4f}")  # 0.01 (1% return)
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize returns-based reward calculation.

        Args:
            config: Optional configuration for clipping and scaling
        """
        self._config = config or RewardConfig()

    def calculate(
        self,
        prev_value: float,
        current_value: float,
    ) -> float:
        """Calculate percentage return as reward."""
        if prev_value <= 0:
            return 0.0

        returns = (current_value - prev_value) / prev_value
        reward = returns * self._config.scale_factor

        # Clip to bounds
        return max(self._config.clip_min, min(self._config.clip_max, reward))

    @property
    def reward_type(self) -> RewardType:
        return RewardType.RETURNS


class PnLRewardCalculation(RewardCalculationService):
    """
    Reward based on absolute P&L (profit and loss).

    Formula: reward = current_value - prev_value

    Advantages:
    - Direct measure of dollar gains/losses
    - Simple to interpret

    Disadvantages:
    - Not scale-invariant (larger portfolios get larger rewards)
    - May need careful scaling for stable learning

    Example:
        >>> calc = PnLRewardCalculation()
        >>> reward = calc.calculate(prev_value=10000, current_value=10100)
        >>> print(f"Reward: {reward:.2f}")  # 100.00 ($100 gain)
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize PnL-based reward calculation.

        Args:
            config: Optional configuration for clipping and scaling
        """
        self._config = config or RewardConfig()

    def calculate(
        self,
        prev_value: float,
        current_value: float,
    ) -> float:
        """Calculate absolute P&L as reward."""
        pnl = current_value - prev_value
        reward = pnl * self._config.scale_factor

        # Clip to bounds
        return max(self._config.clip_min, min(self._config.clip_max, reward))

    @property
    def reward_type(self) -> RewardType:
        return RewardType.PNL


class RiskAdjustedRewardCalculation(RewardCalculationService):
    """
    Reward adjusted for risk (volatility-scaled returns).

    This approach scales returns by recent volatility to encourage
    consistent performance rather than high-variance strategies.

    Formula: reward = returns / (volatility + epsilon)

    Note: Requires maintaining a rolling window of returns.
    """

    def __init__(
        self,
        window_size: int = 20,
        epsilon: float = 1e-8,
        config: RewardConfig = None,
    ):
        """
        Initialize risk-adjusted reward calculation.

        Args:
            window_size: Number of periods for volatility calculation
            epsilon: Small value to prevent division by zero
            config: Optional configuration for clipping and scaling
        """
        self._window_size = window_size
        self._epsilon = epsilon
        self._config = config or RewardConfig()
        self._returns_history: list = []

    def calculate(
        self,
        prev_value: float,
        current_value: float,
    ) -> float:
        """Calculate risk-adjusted return as reward."""
        if prev_value <= 0:
            return 0.0

        # Calculate return
        returns = (current_value - prev_value) / prev_value

        # Update history
        self._returns_history.append(returns)
        if len(self._returns_history) > self._window_size:
            self._returns_history.pop(0)

        # Calculate volatility (standard deviation of returns)
        if len(self._returns_history) < 2:
            volatility = self._epsilon
        else:
            mean = sum(self._returns_history) / len(self._returns_history)
            variance = sum((r - mean) ** 2 for r in self._returns_history) / len(
                self._returns_history
            )
            volatility = variance**0.5

        # Risk-adjusted return
        reward = returns / (volatility + self._epsilon)
        reward *= self._config.scale_factor

        # Clip to bounds
        return max(self._config.clip_min, min(self._config.clip_max, reward))

    @property
    def reward_type(self) -> RewardType:
        return RewardType.RETURNS  # Conceptually returns-based

    def reset(self):
        """Reset the returns history (call on episode reset)."""
        self._returns_history.clear()


def create_reward_service(
    reward_type: str,
    config: RewardConfig = None,
) -> RewardCalculationService:
    """
    Factory function to create reward calculation service.

    Args:
        reward_type: Type of reward ("returns" or "pnl")
        config: Optional reward configuration

    Returns:
        RewardCalculationService instance

    Raises:
        ValueError: If reward_type is not supported
    """
    if reward_type == "returns":
        return ReturnsRewardCalculation(config)
    elif reward_type == "pnl":
        return PnLRewardCalculation(config)
    else:
        raise ValueError(
            f"Unsupported reward_type: '{reward_type}'. "
            f"Valid options: 'returns', 'pnl'"
        )
