"""
Data Ingestion Module
Handles fetching and processing of market data from exchanges.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Dict, Optional, AsyncGenerator
from collections import deque
import logging

from core.types import Candle, MarketData


logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for exchange data providers."""
    
    @abstractmethod
    async def connect(self):
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source."""
        pass
    
    @abstractmethod
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Fetch historical candlestick data."""
        pass
    
    @abstractmethod
    async def subscribe_candles(
        self,
        symbol: str,
        timeframe: str
    ) -> AsyncGenerator[Candle, None]:
        """Subscribe to real-time candlestick updates."""
        pass
    
    @abstractmethod
    async def get_orderbook_snapshot(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict:
        """Get current order book snapshot."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker data."""
        pass


class HyperliquidProvider(DataProvider):
    """
    Data provider for Hyperliquid DEX.
    
    Hyperliquid is recommended for this strategy due to:
    - Low latency perpetual futures
    - Good liquidity on alt-perps
    - No KYC required
    - Transparent on-chain order book
    - Low fees (0.02% maker, 0.05% taker)
    """
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.base_url = (
            "https://api.hyperliquid-testnet.xyz" if testnet
            else "https://api.hyperliquid.xyz"
        )
        self.ws_url = (
            "wss://api.hyperliquid-testnet.xyz/ws" if testnet
            else "wss://api.hyperliquid.xyz/ws"
        )
        self._session = None
        self._ws = None
    
    async def connect(self):
        """Initialize HTTP session and WebSocket connection."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info(f"Connected to Hyperliquid ({'testnet' if self.testnet else 'mainnet'})")
    
    async def disconnect(self):
        """Close connections."""
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("Disconnected from Hyperliquid")
    
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Fetch historical candles from Hyperliquid."""
        
        # Convert timeframe to Hyperliquid format
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        hl_timeframe = tf_map.get(timeframe, "5m")
        
        # Hyperliquid uses Unix timestamps in milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol.replace("-PERP", ""),
                "interval": hl_timeframe,
                "startTime": start_ms,
                "endTime": end_ms
            }
        }
        
        async with self._session.post(
            f"{self.base_url}/info",
            json=payload
        ) as response:
            data = await response.json()

        if data is None:
            logger.warning(f"Hyperliquid returned no data for {symbol} ({timeframe})")
            return []
        if isinstance(data, dict) and "error" in data:
            raise ValueError(f"Hyperliquid error for {symbol}: {data.get('error')}")
        if not isinstance(data, list):
            raise ValueError(f"Unexpected Hyperliquid response for {symbol}: {data}")
        
        candles = []
        for c in data:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(c['t'] / 1000),
                open=float(c['o']),
                high=float(c['h']),
                low=float(c['l']),
                close=float(c['c']),
                volume=float(c['v'])
            ))
        
        return candles
    
    async def subscribe_candles(
        self,
        symbol: str,
        timeframe: str
    ) -> AsyncGenerator[Candle, None]:
        """Subscribe to real-time candle updates via WebSocket."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url) as ws:
                # Subscribe to candles
                await ws.send_json({
                    "method": "subscribe",
                    "subscription": {
                        "type": "candle",
                        "coin": symbol.replace("-PERP", ""),
                        "interval": timeframe
                    }
                })
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if data.get('channel') == 'candle':
                            c = data['data']
                            yield Candle(
                                timestamp=datetime.fromtimestamp(c['t'] / 1000),
                                open=float(c['o']),
                                high=float(c['h']),
                                low=float(c['l']),
                                close=float(c['c']),
                                volume=float(c['v'])
                            )
    
    async def get_orderbook_snapshot(
        self,
        symbol: str,
        depth: int = 20
    ) -> Dict:
        """Get order book snapshot."""
        payload = {
            "type": "l2Book",
            "coin": symbol.replace("-PERP", "")
        }
        
        async with self._session.post(
            f"{self.base_url}/info",
            json=payload
        ) as response:
            data = await response.json()
        
        return {
            "bids": [(float(b['px']), float(b['sz'])) for b in data['levels'][0][:depth]],
            "asks": [(float(a['px']), float(a['sz'])) for a in data['levels'][1][:depth]],
            "timestamp": datetime.utcnow()
        }
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker."""
        payload = {
            "type": "allMids"
        }
        
        async with self._session.post(
            f"{self.base_url}/info",
            json=payload
        ) as response:
            data = await response.json()
        
        coin = symbol.replace("-PERP", "")
        mid_price = float(data.get(coin, 0))
        
        return {
            "symbol": symbol,
            "mid_price": mid_price,
            "timestamp": datetime.utcnow()
        }


class BybitProvider(DataProvider):
    """
    Data provider for Bybit exchange.
    
    Bybit is a good alternative for this strategy:
    - Wide selection of altcoin perps
    - Good API and WebSocket support
    - Reasonable fees
    - Testnet available
    """
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        self.base_url = (
            "https://api-testnet.bybit.com" if testnet
            else "https://api.bybit.com"
        )
        self.ws_url = (
            "wss://stream-testnet.bybit.com/v5/public/linear" if testnet
            else "wss://stream.bybit.com/v5/public/linear"
        )
        self._session = None
    
    async def connect(self):
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info(f"Connected to Bybit ({'testnet' if self.testnet else 'mainnet'})")
    
    async def disconnect(self):
        if self._session:
            await self._session.close()
    
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Fetch historical candles from Bybit."""
        
        # Convert timeframe
        tf_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "4h": "240", "1d": "D"}
        bybit_tf = tf_map.get(timeframe, "5")
        
        params = {
            "category": "linear",
            "symbol": symbol.replace("-PERP", "USDT"),
            "interval": bybit_tf,
            "start": int(start_time.timestamp() * 1000),
            "end": int(end_time.timestamp() * 1000),
            "limit": 1000
        }
        
        async with self._session.get(
            f"{self.base_url}/v5/market/kline",
            params=params
        ) as response:
            data = await response.json()
        
        candles = []
        for c in reversed(data.get('result', {}).get('list', [])):
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5])
            ))
        
        return candles
    
    async def subscribe_candles(
        self,
        symbol: str,
        timeframe: str
    ) -> AsyncGenerator[Candle, None]:
        """Subscribe to real-time candles."""
        import aiohttp
        
        tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60"}
        bybit_tf = tf_map.get(timeframe, "5")
        bybit_symbol = symbol.replace("-PERP", "USDT")
        
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url) as ws:
                await ws.send_json({
                    "op": "subscribe",
                    "args": [f"kline.{bybit_tf}.{bybit_symbol}"]
                })
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = msg.json()
                        if 'data' in data:
                            for c in data['data']:
                                yield Candle(
                                    timestamp=datetime.fromtimestamp(c['start'] / 1000),
                                    open=float(c['open']),
                                    high=float(c['high']),
                                    low=float(c['low']),
                                    close=float(c['close']),
                                    volume=float(c['volume'])
                                )
    
    async def get_orderbook_snapshot(self, symbol: str, depth: int = 20) -> Dict:
        """Get order book snapshot."""
        params = {
            "category": "linear",
            "symbol": symbol.replace("-PERP", "USDT"),
            "limit": depth
        }
        
        async with self._session.get(
            f"{self.base_url}/v5/market/orderbook",
            params=params
        ) as response:
            data = await response.json()
        
        result = data.get('result', {})
        return {
            "bids": [(float(b[0]), float(b[1])) for b in result.get('b', [])],
            "asks": [(float(a[0]), float(a[1])) for a in result.get('a', [])],
            "timestamp": datetime.utcnow()
        }
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker."""
        params = {
            "category": "linear",
            "symbol": symbol.replace("-PERP", "USDT")
        }
        
        async with self._session.get(
            f"{self.base_url}/v5/market/tickers",
            params=params
        ) as response:
            data = await response.json()
        
        ticker = data.get('result', {}).get('list', [{}])[0]
        return {
            "symbol": symbol,
            "last_price": float(ticker.get('lastPrice', 0)),
            "bid": float(ticker.get('bid1Price', 0)),
            "ask": float(ticker.get('ask1Price', 0)),
            "volume_24h": float(ticker.get('volume24h', 0)),
            "timestamp": datetime.utcnow()
        }


class SimulatedDataProvider(DataProvider):
    """
    Simulated data provider for backtesting.
    Replays historical data as if it were live.
    """
    
    def __init__(self, historical_data: Dict[str, List[Candle]]):
        """
        Initialize with pre-loaded historical data.
        
        Args:
            historical_data: Dict mapping symbol to list of candles
        """
        self.historical_data = historical_data
        self.current_index: Dict[str, int] = {}
        self.is_connected = False
    
    async def connect(self):
        self.is_connected = True
        for symbol in self.historical_data:
            self.current_index[symbol] = 0
        logger.info("Simulated data provider connected")
    
    async def disconnect(self):
        self.is_connected = False
    
    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Return candles within the specified time range."""
        if symbol not in self.historical_data:
            return []
        
        return [
            c for c in self.historical_data[symbol]
            if start_time <= c.timestamp <= end_time
        ]
    
    async def subscribe_candles(
        self,
        symbol: str,
        timeframe: str
    ) -> AsyncGenerator[Candle, None]:
        """Yield historical candles one by one (for backtesting)."""
        if symbol not in self.historical_data:
            return
        
        for candle in self.historical_data[symbol]:
            yield candle
    
    async def get_orderbook_snapshot(self, symbol: str, depth: int = 20) -> Dict:
        """Generate synthetic order book based on current candle."""
        if symbol not in self.historical_data:
            return {"bids": [], "asks": [], "timestamp": datetime.utcnow()}
        
        idx = self.current_index.get(symbol, 0)
        if idx >= len(self.historical_data[symbol]):
            idx = len(self.historical_data[symbol]) - 1
        
        candle = self.historical_data[symbol][idx]
        mid = candle.close
        spread = mid * 0.0005  # 0.05% spread
        
        # Generate synthetic levels
        bids = [(mid - spread - i * spread, 1000 + i * 100) for i in range(depth)]
        asks = [(mid + spread + i * spread, 1000 + i * 100) for i in range(depth)]
        
        return {
            "bids": bids,
            "asks": asks,
            "timestamp": candle.timestamp
        }
    
    async def get_ticker(self, symbol: str) -> Dict:
        """Get ticker from current candle."""
        if symbol not in self.historical_data:
            return {}
        
        idx = self.current_index.get(symbol, 0)
        candle = self.historical_data[symbol][idx]
        
        return {
            "symbol": symbol,
            "mid_price": candle.close,
            "timestamp": candle.timestamp
        }
    
    def advance(self, symbol: str):
        """Advance to next candle (for backtesting)."""
        if symbol in self.current_index:
            self.current_index[symbol] += 1


class DataAggregator:
    """
    Aggregates raw candle data with technical indicators.
    Produces MarketData objects ready for signal generation.
    """
    
    def __init__(self, config):
        self.config = config
        self.candle_buffers: Dict[str, deque] = {}
        self.vwap_data: Dict[str, Dict] = {}
    
    def _ensure_buffer(self, symbol: str, size: int = 200):
        """Ensure candle buffer exists for symbol."""
        if symbol not in self.candle_buffers:
            self.candle_buffers[symbol] = deque(maxlen=size)
    
    def add_candle(self, symbol: str, candle: Candle):
        """Add a new candle to the buffer."""
        self._ensure_buffer(symbol)
        self.candle_buffers[symbol].append(candle)
    
    def calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range."""
        buffer = self.candle_buffers.get(symbol, [])
        if len(buffer) < period + 1:
            return 0.0
        
        candles = list(buffer)[-period-1:]
        true_ranges = []
        
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_prev_close = abs(candles[i].high - candles[i-1].close)
            low_prev_close = abs(candles[i].low - candles[i-1].close)
            true_ranges.append(max(high_low, high_prev_close, low_prev_close))
        
        return sum(true_ranges) / len(true_ranges)
    
    def calculate_vwap(self, symbol: str, period: int = 20) -> float:
        """Calculate Volume Weighted Average Price (rolling)."""
        buffer = self.candle_buffers.get(symbol, [])
        if len(buffer) < period:
            return 0.0
        
        candles = list(buffer)[-period:]
        
        total_vp = sum(c.close * c.volume for c in candles)
        total_volume = sum(c.volume for c in candles)
        
        if total_volume == 0:
            return candles[-1].close
        
        return total_vp / total_volume
    
    def calculate_volume_sma(self, symbol: str, period: int = 20) -> float:
        """Calculate simple moving average of volume."""
        buffer = self.candle_buffers.get(symbol, [])
        if len(buffer) < period:
            return 0.0
        
        volumes = [c.volume for c in list(buffer)[-period:]]
        return sum(volumes) / len(volumes)
    
    def calculate_atr_baseline(self, symbol: str, period: int = 50) -> float:
        """Calculate longer-term ATR baseline."""
        return self.calculate_atr(symbol, period)
    
    def get_market_data(
        self,
        symbol: str,
        candle: Candle,
        orderbook: Optional[Dict] = None
    ) -> MarketData:
        """
        Aggregate all market data into a MarketData object.
        """
        self.add_candle(symbol, candle)
        
        atr = self.calculate_atr(symbol, self.config.signal.atr_period)
        vwap = self.calculate_vwap(symbol, self.config.signal.vwap_rolling_period)
        volume_sma = self.calculate_volume_sma(symbol, self.config.filters.volume_baseline_period)
        atr_baseline = self.calculate_atr_baseline(symbol, self.config.filters.atr_baseline_period)
        
        # Calculate ratios
        volume_ratio = candle.volume / volume_sma if volume_sma > 0 else 1.0
        atr_ratio = atr / atr_baseline if atr_baseline > 0 else 1.0
        
        # Order book data
        bid_price = ask_price = spread = bid_depth = ask_depth = None
        if orderbook:
            if orderbook.get('bids'):
                bid_price = orderbook['bids'][0][0]
            if orderbook.get('asks'):
                ask_price = orderbook['asks'][0][0]
            if bid_price and ask_price:
                spread = (ask_price - bid_price) / ((ask_price + bid_price) / 2)
            
            # Calculate depth at configured distance from mid
            mid = (bid_price + ask_price) / 2 if bid_price and ask_price else candle.close
            depth_threshold = mid * self.config.filters.orderbook_depth_pct
            
            bid_depth = sum(
                price * size for price, size in orderbook.get('bids', [])
                if mid - price <= depth_threshold
            )
            ask_depth = sum(
                price * size for price, size in orderbook.get('asks', [])
                if price - mid <= depth_threshold
            )
        
        return MarketData(
            symbol=symbol,
            timestamp=candle.timestamp,
            candle=candle,
            atr=atr,
            vwap=vwap,
            volume_sma=volume_sma,
            volume_ratio=volume_ratio,
            atr_baseline=atr_baseline,
            atr_ratio=atr_ratio,
            bid_price=bid_price,
            ask_price=ask_price,
            spread=spread,
            bid_depth_usd=bid_depth,
            ask_depth_usd=ask_depth
        )
    
    def get_candle_history(self, symbol: str, count: int = 100) -> List[Candle]:
        """Get recent candle history."""
        buffer = self.candle_buffers.get(symbol, [])
        return list(buffer)[-count:]


def create_data_provider(provider_name: str, testnet: bool = True) -> DataProvider:
    """Factory function to create data providers."""
    providers = {
        "hyperliquid": HyperliquidProvider,
        "bybit": BybitProvider,
    }
    
    if provider_name.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return providers[provider_name.lower()](testnet=testnet)
