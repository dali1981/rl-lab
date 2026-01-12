"""
Real-time WebSocket stream consumer for live trading.

This module streams real-time trade data from Binance and creates dollar volume bars
on-the-fly for immediate use in trading decisions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import deque
import pandas as pd

try:
    from binance import AsyncClient, BinanceSocketManager
except ImportError:
    AsyncClient = None
    BinanceSocketManager = None

from ..data.bar_processor import BarProcessor

logger = logging.getLogger(__name__)


class StreamConsumer:
    """
    Real-time trade stream consumer with dollar volume bar creation.

    This class:
    1. Connects to Binance WebSocket for live trade data
    2. Buffers trades per symbol
    3. Creates dollar volume bars when threshold is reached
    4. Calls callback function with new bars

    Example:
        >>> async def on_bar(symbol: str, bar: pd.DataFrame):
        ...     print(f"New bar for {symbol}: {bar}")
        >>>
        >>> consumer = StreamConsumer(
        ...     symbols=["BTCUSDT", "ETHUSDT"],
        ...     on_bar_callback=on_bar,
        ...     dollar_volume_thresholds={"BTCUSDT": 1000000, "ETHUSDT": 500000}
        ... )
        >>> await consumer.start()
    """

    def __init__(
        self,
        symbols: List[str],
        on_bar_callback: Callable[[str, pd.DataFrame], None],
        dollar_volume_thresholds: Optional[Dict[str, float]] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        max_buffer_size: int = 10000,
    ):
        """
        Initialize the stream consumer.

        Args:
            symbols: List of symbols to stream (e.g., ["BTCUSDT", "ETHUSDT"])
            on_bar_callback: Async callback function called when new bar is created
                            Signature: async def callback(symbol: str, bar: pd.DataFrame)
            dollar_volume_thresholds: Dollar volume threshold per symbol
            api_key: Binance API key (optional for public streams)
            api_secret: Binance API secret (optional for public streams)
            testnet: Use Binance testnet endpoint
            max_buffer_size: Maximum trades to buffer per symbol before forcing a bar
        """
        if AsyncClient is None:
            raise ImportError(
                "python-binance is required for streaming. Install with: uv add python-binance"
            )

        self.symbols = symbols
        self.on_bar_callback = on_bar_callback
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.max_buffer_size = max_buffer_size

        # Initialize dollar volume thresholds
        self.dollar_volume_thresholds = dollar_volume_thresholds or {}
        default_threshold = 100000  # $100k default

        # Trade buffers per symbol
        self.trade_buffers: Dict[str, List[dict]] = {
            symbol: [] for symbol in symbols
        }

        # Accumulated dollar volume per symbol
        self.accumulated_volume: Dict[str, float] = {
            symbol: 0.0 for symbol in symbols
        }

        # Bar processors
        self.bar_processors: Dict[str, BarProcessor] = {}
        for symbol in symbols:
            threshold = self.dollar_volume_thresholds.get(symbol, default_threshold)
            self.bar_processors[symbol] = BarProcessor(
                symbol=symbol,
                threshold=threshold
            )

        # Connection management
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        self.running = False

        logger.info(
            f"Initialized StreamConsumer for {len(symbols)} symbols: {', '.join(symbols)}"
        )

    async def start(self):
        """Start streaming and processing trades."""
        if self.running:
            logger.warning("StreamConsumer is already running")
            return

        self.running = True
        logger.info("Starting StreamConsumer...")

        try:
            # Create async client
            if self.testnet:
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=True,
                )
            else:
                self.client = await AsyncClient.create(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                )

            self.bsm = BinanceSocketManager(self.client)

            # Create multiplex stream for aggregated trades
            streams = [f"{symbol.lower()}@aggTrade" for symbol in self.symbols]

            async with self.bsm.multiplex_socket(streams) as stream:
                logger.info(f"Connected to WebSocket streams: {', '.join(self.symbols)}")

                while self.running:
                    try:
                        msg = await stream.recv()

                        if "data" in msg:
                            await self._process_trade(msg["data"])

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        await asyncio.sleep(1)
                        continue

        except Exception as e:
            logger.error(f"Fatal error in StreamConsumer: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self):
        """Stop streaming and close connections."""
        logger.info("Stopping StreamConsumer...")
        self.running = False

        if self.client:
            await self.client.close_connection()
            self.client = None

        self.bsm = None
        logger.info("StreamConsumer stopped")

    async def _process_trade(self, trade_data: dict):
        """
        Process a single trade message.

        Args:
            trade_data: Raw trade data from WebSocket
        """
        symbol = trade_data["s"]

        if symbol not in self.symbols:
            return

        # Parse trade
        trade = {
            "timestamp": pd.Timestamp(trade_data["T"], unit="ms"),
            "price": float(trade_data["p"]),
            "quantity": float(trade_data["q"]),
            "is_buyer_maker": trade_data["m"],
        }

        # Add to buffer
        self.trade_buffers[symbol].append(trade)

        # Update accumulated dollar volume
        dollar_volume = trade["price"] * trade["quantity"]
        self.accumulated_volume[symbol] += dollar_volume

        # Get threshold for this symbol
        threshold = self.dollar_volume_thresholds.get(
            symbol,
            self.bar_processors[symbol].threshold
        )

        # Check if we should create a bar
        should_create_bar = (
            self.accumulated_volume[symbol] >= threshold or
            len(self.trade_buffers[symbol]) >= self.max_buffer_size
        )

        if should_create_bar:
            await self._create_bar(symbol)

    async def _create_bar(self, symbol: str):
        """
        Create a bar from buffered trades.

        Args:
            symbol: Trading symbol
        """
        if not self.trade_buffers[symbol]:
            return

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trade_buffers[symbol])

        # Create OHLCV bar
        bar = pd.DataFrame([{
            "timestamp": trades_df["timestamp"].iloc[0],
            "open": trades_df["price"].iloc[0],
            "high": trades_df["price"].max(),
            "low": trades_df["price"].min(),
            "close": trades_df["price"].iloc[-1],
            "volume": trades_df["quantity"].sum(),
            "dollar_volume": self.accumulated_volume[symbol],
            "num_trades": len(trades_df),
        }])

        logger.debug(
            f"Created bar for {symbol}: "
            f"${self.accumulated_volume[symbol]:,.0f} volume, "
            f"{len(trades_df)} trades"
        )

        # Call the callback
        try:
            if asyncio.iscoroutinefunction(self.on_bar_callback):
                await self.on_bar_callback(symbol, bar)
            else:
                self.on_bar_callback(symbol, bar)
        except Exception as e:
            logger.error(f"Error in on_bar_callback for {symbol}: {e}")

        # Reset buffer and volume
        self.trade_buffers[symbol] = []
        self.accumulated_volume[symbol] = 0.0


class MultiSymbolStreamConsumer:
    """
    Manages multiple StreamConsumers for different symbols with independent bar callbacks.

    This is useful when you want different handling for different symbols.

    Example:
        >>> manager = MultiSymbolStreamConsumer()
        >>> manager.add_symbol("BTCUSDT", on_btc_bar, threshold=1000000)
        >>> manager.add_symbol("ETHUSDT", on_eth_bar, threshold=500000)
        >>> await manager.start_all()
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.consumers: Dict[str, StreamConsumer] = {}

    def add_symbol(
        self,
        symbol: str,
        on_bar_callback: Callable[[str, pd.DataFrame], None],
        threshold: float = 100000,
    ):
        """Add a symbol with its own callback."""
        consumer = StreamConsumer(
            symbols=[symbol],
            on_bar_callback=on_bar_callback,
            dollar_volume_thresholds={symbol: threshold},
            api_key=self.api_key,
            api_secret=self.api_secret,
        )
        self.consumers[symbol] = consumer

    async def start_all(self):
        """Start all consumers concurrently."""
        tasks = [consumer.start() for consumer in self.consumers.values()]
        await asyncio.gather(*tasks)

    async def stop_all(self):
        """Stop all consumers."""
        tasks = [consumer.stop() for consumer in self.consumers.values()]
        await asyncio.gather(*tasks)