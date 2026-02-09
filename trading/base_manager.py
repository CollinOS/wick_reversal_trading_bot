"""
Base trading manager with shared logic for live and paper trading.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Optional

from config.settings import StrategyConfig, SymbolConfig
from config.paths import CANDLE_CACHE_FILE
from core.types import Candle
from main import WickReversalStrategy
from data.ingestion import HyperliquidProvider
from execution.orders import HyperliquidExecutor

logger = logging.getLogger(__name__)


class BaseTradingManager:
    """Base class for trading managers with shared infrastructure."""

    # Maximum age (in minutes) for candle data before symbol is considered "not ready"
    MAX_DATA_AGE_MINUTES = 30  # 30 minutes - cache should be fresh from live_monitor

    # Path to cache file written by live_monitor
    CACHE_FILE = CANDLE_CACHE_FILE

    def __init__(
        self,
        config: StrategyConfig,
        data_provider: HyperliquidProvider,
        executor: HyperliquidExecutor,
        initial_capital: float
    ):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.initial_capital = initial_capital

        # Cached candles loaded from file (written by live_monitor)
        self.cached_candles: Dict[str, list] = {}
        self.cache_loaded_at: Optional[datetime] = None

        self.strategy: Optional[WickReversalStrategy] = None
        self.current_symbols: Set[str] = set()
        self.symbol_tasks: dict = {}
        self.is_running = False

        # Track symbols that don't have fresh enough data
        self.symbols_not_ready: Set[str] = set()

    def load_candle_cache(self) -> bool:
        """Load candles from cache file written by live_monitor.

        Returns True if cache is fresh and valid, False otherwise.
        """
        if not self.CACHE_FILE.exists():
            print(f"  WARNING: Cache file not found: {self.CACHE_FILE}")
            print(f"  Make sure live_monitor.py is running!")
            return False

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            # Check cache age
            saved_at = datetime.fromisoformat(data.get("saved_at", "2000-01-01"))
            cache_age_minutes = (datetime.utcnow() - saved_at).total_seconds() / 60

            if cache_age_minutes > self.MAX_DATA_AGE_MINUTES:
                print(f"  WARNING: Cache is {cache_age_minutes:.0f} minutes old (max: {self.MAX_DATA_AGE_MINUTES})")
                print(f"  Make sure live_monitor.py is running!")
                return False

            # Load candles
            candle_data = data.get("candles", {})
            total_candles = 0

            for symbol, candle_list in candle_data.items():
                candles = []
                for c in candle_list:
                    candles.append(Candle(
                        timestamp=datetime.fromisoformat(c["timestamp"]),
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"]
                    ))
                self.cached_candles[symbol] = candles
                total_candles += len(candles)

            self.cache_loaded_at = datetime.utcnow()
            print(f"  Loaded {total_candles} candles for {len(self.cached_candles)} symbols from cache")
            print(f"  Cache age: {cache_age_minutes:.1f} minutes")
            return True

        except Exception as e:
            print(f"  ERROR loading cache: {e}")
            return False

    def get_cached_candles(self, symbol: str, count: int = 100) -> list:
        """Get candles for a symbol from the loaded cache."""
        if symbol not in self.cached_candles:
            return []
        return self.cached_candles[symbol][-count:]

    def reload_symbol_from_cache(self, symbol: str) -> bool:
        """Reload a single symbol's candles from the cache file.

        Used when a new symbol is added - live_monitor may have fresh data for it.
        """
        if not self.CACHE_FILE.exists():
            return False

        try:
            with open(self.CACHE_FILE, 'r') as f:
                data = json.load(f)

            candle_data = data.get("candles", {})
            if symbol not in candle_data:
                return False

            candles = []
            for c in candle_data[symbol]:
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(c["timestamp"]),
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"]
                ))

            self.cached_candles[symbol] = candles
            return len(candles) > 0

        except Exception as e:
            logger.warning(f"Failed to reload {symbol} from cache: {e}")
            return False

    async def start(self, symbols: list):
        """Start trading with initial symbols."""
        self.current_symbols = set(symbols)

        await self._pre_start(symbols)

        # Create strategy
        self.strategy = WickReversalStrategy(
            config=self.config,
            data_provider=self.data_provider,
            executor=self.executor
        )

        await self._configure_strategy()

        await self.strategy.initialize(initial_capital=self.initial_capital)
        self.is_running = True

        logger.info(f"Started trading: {', '.join(symbols)}")

        # Start streaming for each symbol
        for symbol in symbols:
            await self._start_symbol_stream(symbol)

        # Also stream BTC for correlation
        await self._start_btc_stream()

        await self._post_start()

    async def _pre_start(self, symbols: list):
        """Hook for subclass pre-start logic."""
        pass

    async def _configure_strategy(self):
        """Hook for subclass to configure strategy after creation."""
        pass

    async def _post_start(self):
        """Hook for subclass post-start logic."""
        pass

    async def _start_symbol_stream(self, symbol: str):
        """Start candle stream for a symbol."""
        if symbol in self.symbol_tasks:
            return

        # Hook for pre-stream setup
        self._on_pre_stream(symbol)

        # Pre-load historical candles to seed ATR and other indicators
        data_ready = await self._preload_historical_data(symbol)

        # If data is not ready, add to ignored positions (will be removed when fresh data arrives)
        if not data_ready:
            self.strategy.ignored_positions.add(symbol)

        # Track candles received to know when symbol becomes ready
        candles_received = [0]  # Use list to allow modification in nested function

        async def stream_symbol():
            try:
                async for candle in self.data_provider.subscribe_candles(
                    symbol, self.config.timeframe.value
                ):
                    if not self.is_running:
                        break
                    if symbol not in self.current_symbols:
                        break

                    candles_received[0] += 1

                    # Check if symbol was not ready but now has fresh data
                    if symbol in self.symbols_not_ready:
                        # After receiving enough fresh candles for ATR (15+), enable trading
                        if candles_received[0] >= 15:
                            self.symbols_not_ready.discard(symbol)
                            self.strategy.ignored_positions.discard(symbol)
                            print(f"\n  {symbol}: Fresh data received ({candles_received[0]} candles) - TRADING ENABLED\n")

                    orderbook = await self.data_provider.get_orderbook_snapshot(symbol)
                    await self.strategy.process_candle(symbol, candle, orderbook)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error streaming {symbol}: {e}")

        task = asyncio.create_task(stream_symbol())
        self.symbol_tasks[symbol] = task
        logger.debug(f"Started stream: {symbol}")

    def _on_pre_stream(self, symbol: str):
        """Hook called before starting a symbol stream."""
        pass

    async def _preload_historical_data(self, symbol: str, num_candles: int = 100):
        """Pre-load historical candles to seed indicators (ATR, VWAP, etc.).

        Uses candles from cache file (written by live_monitor).
        Returns True if data is fresh enough for trading, False otherwise.
        """
        try:
            # Get candles from loaded cache
            candles = self.get_cached_candles(symbol, num_candles)

            # If not in cache, try reloading from file (live_monitor may have added it)
            if not candles:
                if self.reload_symbol_from_cache(symbol):
                    candles = self.get_cached_candles(symbol, num_candles)

            if candles:
                newest = candles[-1]
                data_age_minutes = (datetime.utcnow() - newest.timestamp).total_seconds() / 60

                # Add candles to the data aggregator
                for candle in candles:
                    self.strategy.data_aggregator.add_candle(symbol, candle)

                # Hook for additional data processing
                self._on_data_preloaded(symbol, candles)

                if data_age_minutes > self.MAX_DATA_AGE_MINUTES:
                    print(f"  {symbol}: {len(candles)} candles "
                          f"({data_age_minutes:.0f}min old - TRADING DISABLED until fresh data)")
                    self.symbols_not_ready.add(symbol)
                    return False
                elif data_age_minutes > 10:
                    print(f"  {symbol}: {len(candles)} candles "
                          f"(newest is {data_age_minutes:.0f}min old)")
                else:
                    print(f"  {symbol}: {len(candles)} candles (current)")
                return True
            else:
                print(f"  {symbol}: No cached data - TRADING DISABLED until fresh data")
                self.symbols_not_ready.add(symbol)
                return False

        except Exception as e:
            logger.warning(f"Failed to preload historical data for {symbol}: {e}")
            self.symbols_not_ready.add(symbol)
            return False

    def _on_data_preloaded(self, symbol: str, candles: list):
        """Hook called after historical data is preloaded for a symbol."""
        pass

    async def _start_btc_stream(self):
        """Start BTC stream for correlation filter."""
        if "BTC-PERP" in self.symbol_tasks:
            return

        async def stream_btc():
            try:
                async for candle in self.data_provider.subscribe_candles(
                    "BTC-PERP", self.config.timeframe.value
                ):
                    if not self.is_running:
                        break
                    self.strategy.btc_price_history.append(candle.close)
                    if len(self.strategy.btc_price_history) > 100:
                        self.strategy.btc_price_history = self.strategy.btc_price_history[-100:]
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error streaming BTC: {e}")

        task = asyncio.create_task(stream_btc())
        self.symbol_tasks["BTC-PERP"] = task

    async def _stop_symbol_stream(self, symbol: str):
        """Stop candle stream for a symbol."""
        if symbol not in self.symbol_tasks:
            return

        task = self.symbol_tasks[symbol]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.symbol_tasks[symbol]
        logger.debug(f"Stopped stream: {symbol}")

    async def update_symbols(self, new_symbols: Set[str]):
        """Update trading symbols dynamically."""
        if new_symbols == self.current_symbols:
            return

        added = new_symbols - self.current_symbols
        removed = self.current_symbols - new_symbols

        # Stop removed symbols
        for symbol in removed:
            await self._stop_symbol_stream(symbol)

        # Update config
        self.config.symbols = [SymbolConfig(symbol=s) for s in new_symbols]

        # Start new symbols
        for symbol in added:
            await self._start_symbol_stream(symbol)

        self.current_symbols = new_symbols

    async def stop(self):
        """Stop all trading."""
        self.is_running = False

        # Cancel all symbol tasks
        for symbol, task in list(self.symbol_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.symbol_tasks.clear()

        if self.strategy:
            await self.strategy.shutdown()

    def get_status(self) -> dict:
        """Get current trading status."""
        if self.strategy:
            return self.strategy.get_status()
        return {}
