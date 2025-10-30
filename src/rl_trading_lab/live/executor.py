"""
Order execution manager for Binance testnet/live trading.

Translates RL model actions into actual market orders and manages positions.
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
from enum import Enum

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None
    BinanceAPIException = None

logger = logging.getLogger(__name__)


class Action(Enum):
    """Trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2


class OrderExecutor:
    """
    Executes trading orders on Binance (testnet or live).

    Translates RL actions (BUY/SELL/HOLD) into market orders and manages
    position state per symbol.

    Example:
        >>> executor = OrderExecutor(
        ...     api_key="your_key",
        ...     api_secret="your_secret",
        ...     testnet=True
        ... )
        >>>
        >>> # Execute BUY action
        >>> result = executor.execute_action(
        ...     symbol="BTCUSDT",
        ...     action=Action.BUY,
        ...     price=67000.0,
        ...     max_position_size=1000
        ... )
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        commission_rate: float = 0.001,  # 0.1% commission
        min_order_size: float = 10.0,  # USD
    ):
        """
        Initialize the order executor.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Use testnet endpoint
            commission_rate: Trading commission rate
            min_order_size: Minimum order size in USD
        """
        if Client is None:
            raise ImportError(
                "python-binance is required. Install with: uv add python-binance"
            )

        self.testnet = testnet
        self.commission_rate = commission_rate
        self.min_order_size = min_order_size

        # Initialize Binance client
        if testnet:
            self.client = Client(
                api_key, api_secret, testnet=True, tld="vision"
            )
        else:
            self.client = Client(api_key, api_secret)

        # Position tracking per symbol
        self.positions: Dict[str, Dict] = {}

        # Execution statistics
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_commission = 0.0

        logger.info(
            f"Initialized OrderExecutor "
            f"({'testnet' if testnet else 'live'}, commission={commission_rate*100}%)"
        )

    def execute_action(
        self,
        symbol: str,
        action: int,  # or Action enum
        price: float,
        max_position_size: float,
    ) -> Dict:
        """
        Execute a trading action.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            action: Action to execute (0=HOLD, 1=BUY, 2=SELL)
            price: Current market price
            max_position_size: Maximum position size in USD

        Returns:
            Dict with execution result:
            {
                "action": str,
                "executed": bool,
                "order_id": str or None,
                "quantity": float,
                "price": float,
                "commission": float,
                "message": str
            }
        """
        # Convert int to Action
        if isinstance(action, int):
            action = Action(action)

        # Initialize result
        result = {
            "action": action.name,
            "executed": False,
            "order_id": None,
            "quantity": 0.0,
            "price": price,
            "commission": 0.0,
            "message": "",
        }

        # Get current position
        position = self.get_position(symbol)

        # Execute action
        if action == Action.HOLD:
            result["message"] = "HOLD - No action taken"
            return result

        elif action == Action.BUY:
            # Only buy if not already long
            if position["quantity"] > 0:
                result["message"] = "Already long - skipping BUY"
                return result

            # Execute buy order
            return self._execute_buy(symbol, price, max_position_size)

        elif action == Action.SELL:
            # Only sell if currently long
            if position["quantity"] == 0:
                result["message"] = "No position - skipping SELL"
                return result

            # Execute sell order
            return self._execute_sell(symbol, price)

        return result

    def _execute_buy(
        self,
        symbol: str,
        price: float,
        max_position_size: float,
    ) -> Dict:
        """Execute a BUY order."""
        result = {
            "action": "BUY",
            "executed": False,
            "order_id": None,
            "quantity": 0.0,
            "price": price,
            "commission": 0.0,
            "message": "",
        }

        try:
            # Calculate quantity based on max position size
            quantity = max_position_size / price

            # Get symbol info for precision
            symbol_info = self._get_symbol_info(symbol)
            if symbol_info:
                quantity = self._round_quantity(quantity, symbol_info)

            # Check minimum order size
            order_value = quantity * price
            if order_value < self.min_order_size:
                result["message"] = f"Order too small (${order_value:.2f} < ${self.min_order_size})"
                return result

            # Place market buy order
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )

            # Calculate commission
            commission = order_value * self.commission_rate

            # Update position
            self.positions[symbol] = {
                "quantity": quantity,
                "entry_price": price,
                "entry_time": datetime.now(),
                "value": order_value,
            }

            # Update statistics
            self.total_orders += 1
            self.successful_orders += 1
            self.total_commission += commission

            result.update({
                "executed": True,
                "order_id": order.get("orderId"),
                "quantity": quantity,
                "commission": commission,
                "message": f"BUY {quantity:.8f} {symbol} @ ${price:.2f}",
            })

            logger.info(result["message"])

        except BinanceAPIException as e:
            self.failed_orders += 1
            result["message"] = f"BinanceAPIException: {e.message}"
            logger.error(result["message"])

        except Exception as e:
            self.failed_orders += 1
            result["message"] = f"Error executing BUY: {e}"
            logger.error(result["message"])

        return result

    def _execute_sell(self, symbol: str, price: float) -> Dict:
        """Execute a SELL order."""
        result = {
            "action": "SELL",
            "executed": False,
            "order_id": None,
            "quantity": 0.0,
            "price": price,
            "commission": 0.0,
            "pnl": 0.0,
            "message": "",
        }

        try:
            position = self.positions.get(symbol)
            if not position or position["quantity"] == 0:
                result["message"] = "No position to sell"
                return result

            quantity = position["quantity"]
            entry_price = position["entry_price"]

            # Place market sell order
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )

            # Calculate PnL and commission
            order_value = quantity * price
            commission = order_value * self.commission_rate
            entry_value = position["value"]
            pnl = order_value - entry_value - commission * 2  # Buy + sell commission

            # Clear position
            self.positions[symbol] = {
                "quantity": 0.0,
                "entry_price": 0.0,
                "entry_time": None,
                "value": 0.0,
            }

            # Update statistics
            self.total_orders += 1
            self.successful_orders += 1
            self.total_commission += commission

            result.update({
                "executed": True,
                "order_id": order.get("orderId"),
                "quantity": quantity,
                "commission": commission,
                "pnl": pnl,
                "message": f"SELL {quantity:.8f} {symbol} @ ${price:.2f} (PnL: ${pnl:.2f})",
            })

            logger.info(result["message"])

        except BinanceAPIException as e:
            self.failed_orders += 1
            result["message"] = f"BinanceAPIException: {e.message}"
            logger.error(result["message"])

        except Exception as e:
            self.failed_orders += 1
            result["message"] = f"Error executing SELL: {e}"
            logger.error(result["message"])

        return result

    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for a symbol.

        Returns:
            Dict with position info:
            {
                "quantity": float,
                "entry_price": float,
                "entry_time": datetime or None,
                "value": float
            }
        """
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0.0,
                "entry_price": 0.0,
                "entry_time": None,
                "value": 0.0,
            }
        return self.positions[symbol]

    def get_account_balance(self) -> Dict:
        """Get account balance from Binance."""
        try:
            account = self.client.get_account()
            balances = {
                asset["asset"]: float(asset["free"]) + float(asset["locked"])
                for asset in account["balances"]
                if float(asset["free"]) + float(asset["locked"]) > 0
            }
            return balances
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}

    def _get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading rules."""
        try:
            info = self.client.get_symbol_info(symbol)
            return info
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

    def _round_quantity(self, quantity: float, symbol_info: Dict) -> float:
        """Round quantity to symbol's step size."""
        try:
            for filter in symbol_info["filters"]:
                if filter["filterType"] == "LOT_SIZE":
                    step_size = float(filter["stepSize"])
                    # Round down to step size
                    precision = len(str(step_size).split(".")[-1].rstrip("0"))
                    quantity = round(quantity - (quantity % step_size), precision)
                    break
        except Exception as e:
            logger.warning(f"Could not round quantity: {e}")
        return quantity

    def get_stats(self) -> Dict:
        """Get execution statistics."""
        success_rate = (
            self.successful_orders / max(self.total_orders, 1)
        )
        return {
            "total_orders": self.total_orders,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": success_rate,
            "total_commission": self.total_commission,
            "active_positions": sum(
                1 for p in self.positions.values() if p["quantity"] > 0
            ),
        }
