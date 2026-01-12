"""
Risk Management Service - Domain service for risk checks and termination.

This service encapsulates the logic for:
- Checking if an episode should terminate due to risk limits
- Tracking peak portfolio value for drawdown calculations
- Validating trades against risk constraints

Risk management is critical for protecting capital and ensuring
the agent learns sustainable trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from rl_trading_lab.domain.value_objects.trade import CompletedTrade


@dataclass(frozen=True)
class RiskLimits:
    """
    Configuration for risk management limits.

    All percentages are expressed as decimals (0.30 = 30%).
    """

    max_drawdown_pct: float = 0.30  # Maximum drawdown before termination
    min_portfolio_pct: float = 0.20  # Minimum portfolio value as % of initial
    max_consecutive_losses: Optional[int] = None  # Optional consecutive loss limit
    max_position_hold_bars: Optional[int] = None  # Optional max holding period


@dataclass(frozen=True)
class RiskCheckResult:
    """Result of a risk check."""

    should_terminate: bool
    reason: Optional[str] = None
    current_drawdown: float = 0.0
    peak_value: float = 0.0


class RiskManagementService(Protocol):
    """
    Protocol for risk management strategies.

    Risk management services check various conditions that might
    require terminating an episode or blocking certain actions.
    """

    def check_termination(
        self,
        portfolio_value: float,
        initial_balance: float,
        current_bar: int,
        trade_history: List[CompletedTrade],
    ) -> RiskCheckResult:
        """
        Check if episode should terminate due to risk limits.

        Args:
            portfolio_value: Current portfolio value
            initial_balance: Initial portfolio value
            current_bar: Current bar index
            trade_history: List of completed trades

        Returns:
            RiskCheckResult indicating whether to terminate and why
        """
        ...

    def reset(self) -> None:
        """Reset state for new episode (e.g., peak value tracking)."""
        ...

    def update_peak(self, portfolio_value: float) -> None:
        """Update peak portfolio value for drawdown calculation."""
        ...


class StandardRiskManagement(RiskManagementService):
    """
    Standard risk management with configurable limits.

    Checks:
    1. Minimum portfolio value (stop-loss on total capital)
    2. Maximum drawdown from peak
    3. Optional consecutive loss limit
    4. Optional maximum holding period

    Example:
        >>> risk_mgr = StandardRiskManagement(RiskLimits(
        ...     max_drawdown_pct=0.20,
        ...     min_portfolio_pct=0.50,
        ... ))
        >>> result = risk_mgr.check_termination(
        ...     portfolio_value=8000,
        ...     initial_balance=10000,
        ...     current_bar=100,
        ...     trade_history=[],
        ... )
        >>> if result.should_terminate:
        ...     print(f"Terminating: {result.reason}")
    """

    def __init__(self, limits: RiskLimits = None):
        """
        Initialize standard risk management.

        Args:
            limits: Risk limits configuration, uses defaults if not provided
        """
        self._limits = limits or RiskLimits()
        self._peak_value: float = 0.0

    def check_termination(
        self,
        portfolio_value: float,
        initial_balance: float,
        current_bar: int,
        trade_history: List[CompletedTrade],
    ) -> RiskCheckResult:
        """Check all risk conditions and return result."""
        # Update peak
        self._peak_value = max(self._peak_value, portfolio_value)

        # Check minimum portfolio value
        min_value = initial_balance * self._limits.min_portfolio_pct
        if portfolio_value < min_value:
            loss_pct = (1 - self._limits.min_portfolio_pct) * 100
            return RiskCheckResult(
                should_terminate=True,
                reason=f"Portfolio below {self._limits.min_portfolio_pct:.0%} of initial (lost {loss_pct:.0f}%)",
                current_drawdown=self._calculate_drawdown(portfolio_value),
                peak_value=self._peak_value,
            )

        # Check maximum drawdown
        if self._peak_value > 0:
            drawdown = self._calculate_drawdown(portfolio_value)
            if drawdown > self._limits.max_drawdown_pct:
                return RiskCheckResult(
                    should_terminate=True,
                    reason=f"Drawdown {drawdown:.1%} exceeded limit {self._limits.max_drawdown_pct:.0%}",
                    current_drawdown=drawdown,
                    peak_value=self._peak_value,
                )

        # Check consecutive losses
        if self._limits.max_consecutive_losses is not None:
            consecutive = self._count_consecutive_losses(trade_history)
            if consecutive >= self._limits.max_consecutive_losses:
                return RiskCheckResult(
                    should_terminate=True,
                    reason=f"Exceeded {self._limits.max_consecutive_losses} consecutive losses",
                    current_drawdown=self._calculate_drawdown(portfolio_value),
                    peak_value=self._peak_value,
                )

        # All checks passed
        return RiskCheckResult(
            should_terminate=False,
            current_drawdown=self._calculate_drawdown(portfolio_value),
            peak_value=self._peak_value,
        )

    def _calculate_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_value <= 0:
            return 0.0
        return (self._peak_value - portfolio_value) / self._peak_value

    def _count_consecutive_losses(self, trade_history: List[CompletedTrade]) -> int:
        """Count consecutive losing trades from most recent."""
        count = 0
        for trade in reversed(trade_history):
            if trade.net_pnl < 0:
                count += 1
            else:
                break
        return count

    def reset(self) -> None:
        """Reset peak value for new episode."""
        self._peak_value = 0.0

    def update_peak(self, portfolio_value: float) -> None:
        """Update peak portfolio value."""
        self._peak_value = max(self._peak_value, portfolio_value)

    @property
    def peak_value(self) -> float:
        """Current peak portfolio value."""
        return self._peak_value

    @property
    def limits(self) -> RiskLimits:
        """Current risk limits."""
        return self._limits


class ConservativeRiskManagement(RiskManagementService):
    """
    Conservative risk management with tighter limits.

    Uses tighter default limits suitable for:
    - Paper trading evaluation
    - Risk-averse strategies
    - Initial agent training
    """

    def __init__(self, limits: RiskLimits = None):
        """Initialize with conservative defaults."""
        default_limits = RiskLimits(
            max_drawdown_pct=0.15,  # 15% max drawdown
            min_portfolio_pct=0.70,  # Stop at 30% loss
            max_consecutive_losses=5,  # Stop after 5 consecutive losses
        )
        self._impl = StandardRiskManagement(limits or default_limits)

    def check_termination(
        self,
        portfolio_value: float,
        initial_balance: float,
        current_bar: int,
        trade_history: List[CompletedTrade],
    ) -> RiskCheckResult:
        return self._impl.check_termination(
            portfolio_value, initial_balance, current_bar, trade_history
        )

    def reset(self) -> None:
        self._impl.reset()

    def update_peak(self, portfolio_value: float) -> None:
        self._impl.update_peak(portfolio_value)


class NoRiskManagement(RiskManagementService):
    """
    No-op risk management (never terminates).

    Useful for:
    - Backtesting full datasets without early termination
    - Testing/debugging
    - Environments where external termination logic is used
    """

    def check_termination(
        self,
        portfolio_value: float,
        initial_balance: float,
        current_bar: int,
        trade_history: List[CompletedTrade],
    ) -> RiskCheckResult:
        return RiskCheckResult(
            should_terminate=False,
            current_drawdown=0.0,
            peak_value=portfolio_value,
        )

    def reset(self) -> None:
        pass

    def update_peak(self, portfolio_value: float) -> None:
        pass
