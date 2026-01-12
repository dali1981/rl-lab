"""
Domain Services - Stateless operations that don't belong to entities.

Per Evans (DDD, pp. 104-107):
"When a significant process or transformation in the domain is not a natural
responsibility of an Entity or Value Object, add an operation to the model
as a standalone interface declared as a Service."

Domain services in this module:
- PositionSizingService: Calculate position sizes based on available capital
- RewardCalculationService: Calculate step rewards based on value changes
- RiskManagementService: Check termination conditions and risk limits
"""

from rl_trading_lab.domain.services.position_sizing import (
    PositionSizingService,
    FixedPercentagePositionSizing,
)
from rl_trading_lab.domain.services.reward_calculation import (
    RewardCalculationService,
    RewardType,
    ReturnsRewardCalculation,
    PnLRewardCalculation,
)
from rl_trading_lab.domain.services.risk_management import (
    RiskManagementService,
    RiskLimits,
    StandardRiskManagement,
)

__all__ = [
    # Position Sizing
    "PositionSizingService",
    "FixedPercentagePositionSizing",
    # Reward Calculation
    "RewardCalculationService",
    "RewardType",
    "ReturnsRewardCalculation",
    "PnLRewardCalculation",
    # Risk Management
    "RiskManagementService",
    "RiskLimits",
    "StandardRiskManagement",
]
