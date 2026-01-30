"""
Persistent Candle Cache

Maintains a rolling buffer of candles for all symbols, persisted to disk.
Survives bot restarts - only needs to fill the gap since last shutdown.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import deque
from dataclasses import asdict

import aiohttp

from core.types import Candle

logger = logging.getLogger(__name__)


class CandleCache:
    """
    Persistent candle cache that:
    - Subscribes to real-time candle updates via WebSocket
    - Maintains rolling buffers of candles per symbol
    - Persists to disk for survival across restarts
    - Provides instant access to historical data
    """

    CACHE_FILE = "candle_cache.json"
    MAX_CANDLES_PER_SYMBOL = 200  # ~16 hours of 5m candles
    SAVE_INTERVAL_SECONDS = 60  # Save to disk every minute

    def __init__(
        self,
        cache_dir: str = "data",
        timeframe: str = "5m",
        testnet: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / self.CACHE_FILE
        self.timeframe = timeframe
        self.testnet = testnet

        # In-memory candle storage: symbol -> deque of Candles
        self.candles: Dict[str, deque] = {}

        # Track subscribed symbols
        self.subscribed_symbols: Set[str] = set()

        # WebSocket state
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._save_task = None
        self._stream_task = None

        # Hyperliquid endpoints
        if testnet:
            self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws"
            self.api_url = "https://api.hyperliquid-testnet.xyz/info"
        else:
            self.ws_url = "wss://api.hyperliquid.xyz/ws"
            self.api_url = "https://api.hyperliquid.xyz/info"

        # Load existing cache from disk
        self._load_from_disk()

    def _load_from_disk(self):
        """Load cached candles from disk."""
        if not self.cache_file.exists():
            logger.info("No existing candle cache found, starting fresh")
            return

        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)

            loaded_count = 0
            for symbol, candle_list in data.get("candles", {}).items():
                self.candles[symbol] = deque(maxlen=self.MAX_CANDLES_PER_SYMBOL)
                for c in candle_list:
                    candle = Candle(
                        timestamp=datetime.fromisoformat(c["timestamp"]),
                        open=c["open"],
                        high=c["high"],
                        low=c["low"],
                        close=c["close"],
                        volume=c["volume"]
                    )
                    self.candles[symbol].append(candle)
                loaded_count += len(candle_list)

            cache_time = data.get("saved_at", "unknown")
            logger.info(f"Loaded {loaded_count} candles for {len(self.candles)} symbols from cache (saved at {cache_time})")

        except Exception as e:
            logger.warning(f"Failed to load candle cache: {e}")

    def _save_to_disk(self):
        """Save current candles to disk."""
        try:
            data = {
                "saved_at": datetime.utcnow().isoformat(),
                "timeframe": self.timeframe,
                "candles": {}
            }

            for symbol, candle_deque in self.candles.items():
                data["candles"][symbol] = [
                    {
                        "timestamp": c.timestamp.isoformat(),
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume
                    }
                    for c in candle_deque
                ]

            with open(self.cache_file, 'w') as f:
                json.dump(data, f)

            total_candles = sum(len(d) for d in self.candles.values())
            logger.debug(f"Saved {total_candles} candles to cache")

        except Exception as e:
            logger.error(f"Failed to save candle cache: {e}")

    async def start(self, symbols: Optional[List[str]] = None):
        """
        Start the candle cache.

        Args:
            symbols: Optional list of symbols to subscribe to.
                     If None, fetches all available perp markets.
        """
        self._running = True
        self._session = aiohttp.ClientSession()

        # Get all perp symbols if not specified
        if symbols is None:
            symbols = await self._fetch_all_perp_symbols()

        logger.info(f"Starting candle cache for {len(symbols)} symbols")

        # Initialize buffers for new symbols
        for symbol in symbols:
            if symbol not in self.candles:
                self.candles[symbol] = deque(maxlen=self.MAX_CANDLES_PER_SYMBOL)

        self.subscribed_symbols = set(symbols)

        # Start background tasks
        self._stream_task = asyncio.create_task(self._stream_candles())
        self._save_task = asyncio.create_task(self._periodic_save())

    async def stop(self):
        """Stop the candle cache and save to disk."""
        self._running = False

        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass

        if self._ws:
            await self._ws.close()

        if self._session:
            await self._session.close()

        # Final save
        self._save_to_disk()
        logger.info("Candle cache stopped and saved")

    async def _fetch_all_perp_symbols(self, max_retries: int = 5) -> List[str]:
        """Fetch all available perpetual market symbols from Hyperliquid with retry logic."""
        for attempt in range(max_retries):
            try:
                async with self._session.post(
                    self.api_url,
                    json={"type": "meta"}
                ) as resp:
                    if resp.status == 429:
                        # Rate limited
                        wait_time = (2 ** attempt) * 2
                        logger.warning(f"Rate limited fetching perp symbols, waiting {wait_time}s ({attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue

                    data = await resp.json()

                symbols = []
                if "universe" in data:
                    for asset in data["universe"]:
                        symbols.append(f"{asset['name']}-PERP")

                logger.info(f"Found {len(symbols)} perpetual markets")
                return symbols

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Error fetching perp symbols: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch perp symbols after {max_retries} attempts: {e}")
                    return []

        logger.error(f"Failed to fetch perp symbols after {max_retries} attempts")
        return []

    async def _stream_candles(self):
        """Stream candles for all subscribed symbols via WebSocket."""
        while self._running:
            try:
                async with self._session.ws_connect(
                    self.ws_url,
                    heartbeat=30,
                    receive_timeout=60
                ) as ws:
                    self._ws = ws

                    # Subscribe to candles for all symbols
                    for symbol in self.subscribed_symbols:
                        coin = symbol.replace("-PERP", "")
                        subscribe_msg = {
                            "method": "subscribe",
                            "subscription": {
                                "type": "candle",
                                "coin": coin,
                                "interval": self.timeframe
                            }
                        }
                        await ws.send_json(subscribe_msg)

                    logger.info(f"Subscribed to {len(self.subscribed_symbols)} candle streams")

                    # Process incoming messages
                    async for msg in ws:
                        if not self._running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._process_ws_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                if self._running:
                    await asyncio.sleep(5)  # Reconnect delay

    async def _process_ws_message(self, data: str):
        """Process incoming WebSocket message."""
        try:
            msg = json.loads(data)

            # Log first few messages to debug format
            if not hasattr(self, '_msg_count'):
                self._msg_count = 0
            self._msg_count += 1
            if self._msg_count <= 3:
                logger.info(f"WebSocket message sample: {str(msg)[:200]}")

            if msg.get("channel") == "candle":
                candle_data = msg.get("data", {})
                coin = candle_data.get("s", "")
                symbol = f"{coin}-PERP"

                if symbol in self.subscribed_symbols:
                    # Parse candle (use UTC timestamp for consistency)
                    candle = Candle(
                        timestamp=datetime.utcfromtimestamp(candle_data["t"] / 1000),
                        open=float(candle_data["o"]),
                        high=float(candle_data["h"]),
                        low=float(candle_data["l"]),
                        close=float(candle_data["c"]),
                        volume=float(candle_data["v"])
                    )

                    # Add to buffer (avoid duplicates by checking timestamp)
                    self._add_candle(symbol, candle)

                    # Track candles received
                    if not hasattr(self, '_candles_received'):
                        self._candles_received = 0
                    self._candles_received += 1
                    if self._candles_received % 100 == 0:
                        logger.info(f"Candle cache: received {self._candles_received} candle updates")

        except Exception as e:
            logger.warning(f"Error processing WS message: {e} - data: {str(data)[:100]}")

    def _add_candle(self, symbol: str, candle: Candle):
        """Add a candle to the buffer, avoiding duplicates."""
        if symbol not in self.candles:
            self.candles[symbol] = deque(maxlen=self.MAX_CANDLES_PER_SYMBOL)

        buffer = self.candles[symbol]

        # Check if this candle already exists (by timestamp)
        if buffer and buffer[-1].timestamp >= candle.timestamp:
            # Update existing candle if same timestamp (candle in progress)
            if buffer[-1].timestamp == candle.timestamp:
                buffer[-1] = candle
            return

        # New candle - add to buffer
        buffer.append(candle)

    async def _periodic_save(self):
        """Periodically save cache to disk."""
        while self._running:
            await asyncio.sleep(self.SAVE_INTERVAL_SECONDS)
            self._save_to_disk()

    def get_candles(self, symbol: str, count: int = 100) -> List[Candle]:
        """
        Get recent candles for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC-PERP')
            count: Number of candles to return

        Returns:
            List of Candle objects, oldest first
        """
        if symbol not in self.candles:
            return []

        buffer = self.candles[symbol]
        return list(buffer)[-count:]

    def get_newest_candle_time(self, symbol: str) -> Optional[datetime]:
        """Get timestamp of the newest candle for a symbol."""
        if symbol not in self.candles or not self.candles[symbol]:
            return None
        return self.candles[symbol][-1].timestamp

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        stats = {
            "symbols": len(self.candles),
            "total_candles": sum(len(d) for d in self.candles.values()),
            "symbols_with_data": {}
        }

        for symbol, buffer in self.candles.items():
            if buffer:
                newest = buffer[-1].timestamp
                age_minutes = (datetime.utcnow() - newest).total_seconds() / 60
                stats["symbols_with_data"][symbol] = {
                    "candles": len(buffer),
                    "newest_age_minutes": round(age_minutes, 1)
                }

        return stats

    def has_sufficient_data(self, symbol: str, min_candles: int = 50) -> bool:
        """Check if we have enough recent data for a symbol."""
        if symbol not in self.candles:
            return False

        buffer = self.candles[symbol]
        if len(buffer) < min_candles:
            return False

        # Check if newest candle is recent (within 10 minutes)
        newest = buffer[-1].timestamp
        age = datetime.utcnow() - newest
        return age.total_seconds() < 600  # 10 minutes

    def get_freshness_report(self) -> dict:
        """Get a report on data freshness for debugging."""
        now = datetime.utcnow()
        fresh_count = 0  # < 10 min old
        stale_count = 0  # > 10 min old
        very_stale_count = 0  # > 5 hours old

        for symbol, buffer in self.candles.items():
            if not buffer:
                continue
            age_minutes = (now - buffer[-1].timestamp).total_seconds() / 60
            if age_minutes < 10:
                fresh_count += 1
            elif age_minutes < 300:
                stale_count += 1
            else:
                very_stale_count += 1

        return {
            "fresh": fresh_count,
            "stale": stale_count,
            "very_stale": very_stale_count,
            "candles_received": getattr(self, '_candles_received', 0)
        }
