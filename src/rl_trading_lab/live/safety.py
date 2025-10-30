"""
Safety guards and circuit breakers for live trading.

Implements multiple layers of protection to prevent catastrophic losses
and ensure safe operation of the trading system.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Trading halted
    MANUAL = "manual"  # Manually triggered stop


class SafetyGuard:
    """
    Comprehensive safety guard system for live trading.

    Monitors multiple risk metrics and triggers circuit breakers
    when safety thresholds are exceeded.

    Example:
        >>> guard = SafetyGuard(
        ...     max_drawdown=0.20,
        ...     max_trades_per_hour=10,
        ...     initial_balance=10000
        ... )
        >>>
        >>> # Check if trading is allowed
        >>> if guard.can_trade("BTCUSDT"):
        ...     # Execute trade
        ...     guard.record_trade("BTCUSDT", -50)  # Lost $50
        >>>
        >>> # Check circuit breaker state
        >>> print(guard.get_state())
    """

    def __init__(
        self,
        max_drawdown: float = 0.20,
        max_trades_per_hour: int = 20,
        max_trades_per_day: int = 100,
        initial_balance: float = 10000.0,
        min_balance_threshold: float = 100.0,
        max_consecutive_losses: int = 5,
        max_position_pct: float = 0.95,
        enable_circuit_breaker: bool = True,
    ):
        """
        Initialize safety guard.

        Args:
            max_drawdown: Maximum allowed drawdown (0-1)
            max_trades_per_hour: Maximum trades per hour
            max_trades_per_day: Maximum trades per day
            initial_balance: Starting balance
            min_balance_threshold: Minimum balance before stopping
            max_consecutive_losses: Stop after this many consecutive losses
            max_position_pct: Maximum % of balance in positions
            enable_circuit_breaker: Enable automatic circuit breaker
        """
        self.max_drawdown = max_drawdown
        self.max_trades_per_hour = max_trades_per_hour
        self.max_trades_per_day = max_trades_per_day
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.min_balance_threshold = min_balance_threshold
        self.max_consecutive_losses = max_consecutive_losses
        self.max_position_pct = max_position_pct
        self.enable_circuit_breaker = enable_circuit_breaker

        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_losses = 0
        self.peak_balance = initial_balance
        self.trade_history: List[Dict] = []

        # Per-symbol tracking
        self.symbol_trade_counts: Dict[str, List[datetime]] = {}

        # Violation tracking
        self.violations: List[Dict] = []

        logger.info(
            f"Initialized SafetyGuard (max_dd={max_drawdown*100}%, "
            f"max_trades_hour={max_trades_per_hour})"
        )

    def can_trade(self, symbol: str) -> tuple[bool, str]:
        """
        Check if trading is allowed for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (allowed, reason)
        """
        # Check circuit breaker state
        if self.state != CircuitBreakerState.CLOSED:
            return False, f"Circuit breaker is {self.state.value}"

        # Check balance threshold
        if self.current_balance < self.min_balance_threshold:
            reason = f"Balance ${self.current_balance:.2f} below threshold ${self.min_balance_threshold:.2f}"
            self._trigger_circuit_breaker(reason)
            return False, reason

        # Check drawdown
        current_drawdown = self.get_drawdown()
        if current_drawdown > self.max_drawdown:
            reason = f"Drawdown {current_drawdown*100:.1f}% exceeds max {self.max_drawdown*100:.1f}%"
            self._trigger_circuit_breaker(reason)
            return False, reason

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            reason = f"Consecutive losses ({self.consecutive_losses}) exceeded max ({self.max_consecutive_losses})"
            self._trigger_circuit_breaker(reason)
            return False, reason

        # Check trade rate limits
        now = datetime.now()

        # Initialize symbol tracking
        if symbol not in self.symbol_trade_counts:
            self.symbol_trade_counts[symbol] = []

        # Clean old trades (older than 1 day)
        self.symbol_trade_counts[symbol] = [
            ts for ts in self.symbol_trade_counts[symbol]
            if now - ts < timedelta(days=1)
        ]

        # Check hourly limit
        recent_hour = [
            ts for ts in self.symbol_trade_counts[symbol]
            if now - ts < timedelta(hours=1)
        ]
        if len(recent_hour) >= self.max_trades_per_hour:
            return False, f"Hourly trade limit reached ({len(recent_hour)}/{self.max_trades_per_hour})"

        # Check daily limit
        recent_day = self.symbol_trade_counts[symbol]
        if len(recent_day) >= self.max_trades_per_day:
            return False, f"Daily trade limit reached ({len(recent_day)}/{self.max_trades_per_day})"

        return True, "OK"

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        new_balance: float,
        trade_details: Optional[Dict] = None,
    ):
        """
        Record a trade and update safety metrics.

        Args:
            symbol: Trading symbol
            pnl: Trade PnL (can be negative)
            new_balance: Updated portfolio balance
            trade_details: Optional trade details for logging
        """
        # Update balance
        self.current_balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Record trade timestamp
        if symbol not in self.symbol_trade_counts:
            self.symbol_trade_counts[symbol] = []
        self.symbol_trade_counts[symbol].append(datetime.now())

        # Log trade
        trade_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "pnl": pnl,
            "balance": new_balance,
            "consecutive_losses": self.consecutive_losses,
            **(trade_details or {}),
        }
        self.trade_history.append(trade_record)

        logger.info(
            f"Recorded trade: {symbol} PnL=${pnl:.2f} "
            f"Balance=${new_balance:.2f} "
            f"ConsecLosses={self.consecutive_losses}"
        )

    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_balance == 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def check_position_size(self, position_value: float) -> tuple[bool, str]:
        """
        Check if position size is within limits.

        Args:
            position_value: Total value of position

        Returns:
            Tuple of (allowed, reason)
        """
        if self.current_balance == 0:
            return False, "Zero balance"

        position_pct = position_value / self.current_balance

        if position_pct > self.max_position_pct:
            return False, f"Position {position_pct*100:.1f}% exceeds max {self.max_position_pct*100:.1f}%"

        return True, "OK"

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker and halt trading."""
        if not self.enable_circuit_breaker:
            logger.warning(f"Circuit breaker disabled, would trigger: {reason}")
            return

        self.state = CircuitBreakerState.OPEN

        violation = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "balance": self.current_balance,
            "drawdown": self.get_drawdown(),
            "consecutive_losses": self.consecutive_losses,
        }
        self.violations.append(violation)

        logger.critical(f"🚨 CIRCUIT BREAKER TRIGGERED: {reason}")
        logger.critical(f"   Balance: ${self.current_balance:.2f}")
        logger.critical(f"   Drawdown: {self.get_drawdown()*100:.1f}%")
        logger.critical(f"   Consecutive losses: {self.consecutive_losses}")

    def manual_stop(self, reason: str = "Manual stop"):
        """Manually trigger circuit breaker."""
        self.state = CircuitBreakerState.MANUAL

        violation = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "balance": self.current_balance,
            "type": "manual",
        }
        self.violations.append(violation)

        logger.warning(f"🛑 Manual stop triggered: {reason}")

    def reset(self, require_manual: bool = True):
        """
        Reset circuit breaker.

        Args:
            require_manual: If True, only reset manual stops
        """
        if require_manual and self.state != CircuitBreakerState.MANUAL:
            logger.error("Cannot reset circuit breaker: not in manual state")
            return False

        logger.info("Resetting circuit breaker to CLOSED state")
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_losses = 0
        return True

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state

    def get_stats(self) -> Dict:
        """Get safety statistics."""
        # Calculate trade counts
        now = datetime.now()
        all_trades = []
        for trades in self.symbol_trade_counts.values():
            all_trades.extend(trades)

        trades_last_hour = len([
            ts for ts in all_trades if now - ts < timedelta(hours=1)
        ])
        trades_today = len([
            ts for ts in all_trades if now - ts < timedelta(days=1)
        ])

        return {
            "state": self.state.value,
            "balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown": self.get_drawdown(),
            "drawdown_pct": self.get_drawdown() * 100,
            "consecutive_losses": self.consecutive_losses,
            "trades_last_hour": trades_last_hour,
            "trades_today": trades_today,
            "violations": len(self.violations),
            "last_violation": self.violations[-1] if self.violations else None,
        }

    def get_violations(self) -> List[Dict]:
        """Get all safety violations."""
        return self.violations.copy()


class ConnectionMonitor:
    """
    Monitors WebSocket connections and triggers reconnection.

    Example:
        >>> monitor = ConnectionMonitor(max_reconnects=5, timeout=30)
        >>> monitor.record_heartbeat()
        >>> if monitor.should_reconnect():
        ...     # Reconnect WebSocket
    """

    def __init__(
        self,
        max_reconnects: int = 5,
        timeout: int = 30,
        heartbeat_interval: int = 10,
    ):
        """
        Initialize connection monitor.

        Args:
            max_reconnects: Maximum reconnection attempts
            timeout: Connection timeout in seconds
            heartbeat_interval: Expected heartbeat interval
        """
        self.max_reconnects = max_reconnects
        self.timeout = timeout
        self.heartbeat_interval = heartbeat_interval

        self.reconnect_count = 0
        self.last_heartbeat = datetime.now()
        self.connected = True

    def record_heartbeat(self):
        """Record successful heartbeat."""
        self.last_heartbeat = datetime.now()
        self.connected = True

    def should_reconnect(self) -> tuple[bool, str]:
        """
        Check if reconnection is needed.

        Returns:
            Tuple of (should_reconnect, reason)
        """
        now = datetime.now()
        time_since_heartbeat = (now - self.last_heartbeat).total_seconds()

        if time_since_heartbeat > self.timeout:
            if self.reconnect_count >= self.max_reconnects:
                return True, f"Max reconnects ({self.max_reconnects}) exceeded"

            self.reconnect_count += 1
            return True, f"Timeout ({time_since_heartbeat:.0f}s > {self.timeout}s)"

        return False, "OK"

    def reset_reconnects(self):
        """Reset reconnection counter after successful reconnection."""
        self.reconnect_count = 0
        logger.info("Connection monitor reset")

    def get_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            "connected": self.connected,
            "reconnect_count": self.reconnect_count,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "seconds_since_heartbeat": (datetime.now() - self.last_heartbeat).total_seconds(),
        }
