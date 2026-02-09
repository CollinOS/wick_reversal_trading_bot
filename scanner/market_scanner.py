"""
Market Scanner for Wick Reversal Strategy

Scans all Hyperliquid perpetual markets and ranks them by profitability
potential for the wick reversal strategy.
"""

import asyncio
import aiohttp
import csv
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from config.settings import StrategyConfig, SignalConfig, FilterConfig
from core.types import Candle, MarketData, SignalType
from signals.detection import WickAnalyzer
from backtest.engine import BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class MarketScore:
    """Scoring results for a single market."""
    symbol: str

    # Signal metrics
    total_signals: int = 0
    long_signals: int = 0
    short_signals: int = 0
    signals_per_day: float = 0.0

    # Win rate (from mini-backtest)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl_pct: float = 0.0

    # Liquidity metrics
    avg_volume_usd: float = 0.0
    avg_spread_pct: float = 0.0

    # Volatility metrics
    avg_atr_pct: float = 0.0
    avg_wick_size_pct: float = 0.0

    # Quality metrics
    avg_signal_strength: float = 0.0
    atr_ratio_signals: int = 0  # High-value signals

    # Overall score (computed)
    overall_score: float = 0.0

    # Additional context
    candles_analyzed: int = 0
    days_analyzed: float = 0.0
    current_price: float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "overall_score": round(self.overall_score, 2),
            "total_signals": self.total_signals,
            "signals_per_day": round(self.signals_per_day, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "profit_factor": round(self.profit_factor, 2),
            "total_pnl_pct": round(self.total_pnl_pct, 2),
            "avg_volume_usd": round(self.avg_volume_usd, 0),
            "avg_atr_pct": round(self.avg_atr_pct * 100, 3),
            "avg_signal_strength": round(self.avg_signal_strength, 3),
            "atr_ratio_signals": self.atr_ratio_signals,
            "current_price": self.current_price,
            "candles_analyzed": self.candles_analyzed,
        }


class HyperliquidScanner:
    """Scans Hyperliquid markets for wick reversal opportunities."""

    BASE_URL = "https://api.hyperliquid.xyz"

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.wick_analyzer = WickAnalyzer(config.signal)
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        """Initialize HTTP session."""
        self._session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close HTTP session."""
        if self._session:
            await self._session.close()

    async def get_all_markets(self) -> List[Dict]:
        """Fetch list of all perpetual markets from Hyperliquid."""
        payload = {"type": "meta"}

        async with self._session.post(
            f"{self.BASE_URL}/info",
            json=payload
        ) as response:
            data = await response.json()

        markets = []
        universe = data.get("universe", [])

        for asset in universe:
            markets.append({
                "symbol": f"{asset['name']}-PERP",
                "name": asset["name"],
                "sz_decimals": asset.get("szDecimals", 0),
            })

        return markets

    async def get_market_info(self) -> Dict[str, Dict]:
        """Get current market info (price, volume, etc.) for all assets."""
        payload = {"type": "allMids"}

        async with self._session.post(
            f"{self.BASE_URL}/info",
            json=payload
        ) as response:
            mids = await response.json()

        # Get 24h stats
        payload = {"type": "metaAndAssetCtxs"}
        async with self._session.post(
            f"{self.BASE_URL}/info",
            json=payload
        ) as response:
            data = await response.json()

        market_info = {}
        asset_ctxs = data[1] if len(data) > 1 else []
        universe = data[0].get("universe", []) if data else []

        for i, asset in enumerate(universe):
            name = asset["name"]
            ctx = asset_ctxs[i] if i < len(asset_ctxs) else {}

            market_info[name] = {
                "mid_price": float(mids.get(name, 0)),
                "mark_price": float(ctx.get("markPx", 0)),
                "open_interest": float(ctx.get("openInterest", 0)),
                "funding_rate": float(ctx.get("funding", 0)),
                "volume_24h": float(ctx.get("dayNtlVlm", 0)),
            }

        return market_info

    async def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        days: int
    ) -> List[Candle]:
        """Fetch historical candles for a symbol."""
        coin = symbol.replace("-PERP", "")

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": timeframe,
                "startTime": int(start_time.timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000)
            }
        }

        try:
            async with self._session.post(
                f"{self.BASE_URL}/info",
                json=payload
            ) as response:
                data = await response.json()

            if not isinstance(data, list):
                return []

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

            return sorted(candles, key=lambda x: x.timestamp)

        except Exception as e:
            logger.warning(f"Failed to fetch candles for {symbol}: {e}")
            return []

    def calculate_atr(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate ATR from candles."""
        if len(candles) < period + 1:
            return 0.0

        true_ranges = []
        for i in range(1, len(candles)):
            high_low = candles[i].high - candles[i].low
            high_prev_close = abs(candles[i].high - candles[i-1].close)
            low_prev_close = abs(candles[i].low - candles[i-1].close)
            true_ranges.append(max(high_low, high_prev_close, low_prev_close))

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0

        return sum(true_ranges[-period:]) / period

    def calculate_vwap(self, candles: List[Candle], period: int = 20) -> float:
        """Calculate rolling VWAP."""
        if len(candles) < period:
            return candles[-1].close if candles else 0

        recent = candles[-period:]
        total_vp = sum(c.close * c.volume for c in recent)
        total_volume = sum(c.volume for c in recent)

        return total_vp / total_volume if total_volume > 0 else recent[-1].close

    def analyze_signals(
        self,
        candles: List[Candle]
    ) -> Tuple[List[Dict], Dict]:
        """Analyze candles for wick signals and collect statistics."""
        signals = []
        stats = {
            "total_wick_sizes": [],
            "signal_strengths": [],
            "atr_ratio_count": 0,
        }

        if len(candles) < 50:
            return signals, stats

        for i in range(50, len(candles)):
            window = candles[max(0, i-50):i+1]
            candle = candles[i]

            atr = self.calculate_atr(window, 14)
            vwap = self.calculate_vwap(window, 20)

            if atr <= 0:
                continue

            # Calculate volume ratio
            volumes = [c.volume for c in window[-20:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            volume_ratio = candle.volume / avg_volume if avg_volume > 0 else 1

            # Analyze wick
            signal_type, strength, criteria = self.wick_analyzer.analyze_wick(
                candle, atr, vwap, volume_ratio
            )

            if signal_type != SignalType.NO_SIGNAL and strength > 0:
                # Calculate entry/exit levels for simulated trade
                if signal_type == SignalType.UPPER_WICK_SHORT:
                    entry = candle.close
                    stop = candle.high + (atr * 0.4)
                    target = entry - (atr * 1.2)
                    side = "short"
                else:
                    entry = candle.close
                    stop = candle.low - (atr * 0.4)
                    target = entry + (atr * 1.2)
                    side = "long"

                signals.append({
                    "index": i,
                    "timestamp": candle.timestamp,
                    "type": signal_type,
                    "side": side,
                    "strength": strength,
                    "criteria": criteria,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "atr": atr,
                })

                stats["signal_strengths"].append(strength)

                # Track high-value ATR ratio signals
                if any("atr_ratio" in c for c in criteria):
                    stats["atr_ratio_count"] += 1

            # Track wick sizes for all candles
            upper_wick_pct = candle.upper_wick / candle.close if candle.close > 0 else 0
            lower_wick_pct = candle.lower_wick / candle.close if candle.close > 0 else 0
            stats["total_wick_sizes"].append(max(upper_wick_pct, lower_wick_pct))

        return signals, stats

    def simulate_trades(
        self,
        signals: List[Dict],
        candles: List[Candle],
        max_hold_candles: int = 12
    ) -> List[Dict]:
        """Simulate trades based on signals to estimate win rate."""
        trades = []

        for signal in signals:
            idx = signal["index"]
            entry = signal["entry"]
            stop = signal["stop"]
            target = signal["target"]
            side = signal["side"]

            # Simulate forward from signal
            exit_price = None
            exit_reason = None

            for j in range(1, min(max_hold_candles + 1, len(candles) - idx)):
                future_candle = candles[idx + j]

                if side == "long":
                    # Check stop loss
                    if future_candle.low <= stop:
                        exit_price = stop
                        exit_reason = "stop_loss"
                        break
                    # Check take profit
                    if future_candle.high >= target:
                        exit_price = target
                        exit_reason = "take_profit"
                        break
                else:  # short
                    # Check stop loss
                    if future_candle.high >= stop:
                        exit_price = stop
                        exit_reason = "stop_loss"
                        break
                    # Check take profit
                    if future_candle.low <= target:
                        exit_price = target
                        exit_reason = "take_profit"
                        break

            # Time exit if no other exit
            if exit_price is None and idx + max_hold_candles < len(candles):
                exit_price = candles[idx + max_hold_candles].close
                exit_reason = "time_exit"

            if exit_price is not None:
                if side == "long":
                    pnl_pct = (exit_price - entry) / entry
                else:
                    pnl_pct = (entry - exit_price) / entry

                trades.append({
                    "entry": entry,
                    "exit": exit_price,
                    "pnl_pct": pnl_pct,
                    "exit_reason": exit_reason,
                    "won": pnl_pct > 0,
                    "strength": signal["strength"],
                    "criteria": signal["criteria"],
                })

        return trades

    async def score_market(
        self,
        symbol: str,
        market_info: Dict,
        days: int,
        timeframe: str = "5m"
    ) -> Optional[MarketScore]:
        """Calculate comprehensive score for a single market."""
        coin = symbol.replace("-PERP", "")
        info = market_info.get(coin, {})

        # Skip if no price data
        if info.get("mid_price", 0) == 0:
            return None

        # Fetch historical candles
        candles = await self.get_historical_candles(symbol, timeframe, days)

        if len(candles) < 100:
            logger.debug(f"Skipping {symbol}: insufficient data ({len(candles)} candles)")
            return None

        # Analyze signals
        signals, stats = self.analyze_signals(candles)

        if len(signals) < 5:
            logger.debug(f"Skipping {symbol}: too few signals ({len(signals)})")
            return None

        # Simulate trades
        trades = self.simulate_trades(signals, candles)

        if len(trades) < 5:
            return None

        # Calculate metrics
        score = MarketScore(symbol=symbol)
        score.candles_analyzed = len(candles)
        score.current_price = info.get("mid_price", 0)

        # Time range
        if candles:
            time_range = (candles[-1].timestamp - candles[0].timestamp).total_seconds()
            score.days_analyzed = time_range / 86400

        # Signal metrics
        score.total_signals = len(signals)
        score.long_signals = len([s for s in signals if s["side"] == "long"])
        score.short_signals = len([s for s in signals if s["side"] == "short"])
        score.signals_per_day = len(signals) / max(score.days_analyzed, 1)

        # Trade metrics
        winning_trades = [t for t in trades if t["won"]]
        losing_trades = [t for t in trades if not t["won"]]

        score.win_rate = len(winning_trades) / len(trades) if trades else 0

        total_wins = sum(t["pnl_pct"] for t in winning_trades)
        total_losses = abs(sum(t["pnl_pct"] for t in losing_trades))
        score.profit_factor = total_wins / total_losses if total_losses > 0 else total_wins
        score.total_pnl_pct = sum(t["pnl_pct"] for t in trades) * 100

        # Volume metrics
        volumes_usd = [c.volume * c.close for c in candles]
        score.avg_volume_usd = sum(volumes_usd) / len(volumes_usd) if volumes_usd else 0

        # Volatility metrics
        if candles:
            atr = self.calculate_atr(candles, 14)
            score.avg_atr_pct = atr / candles[-1].close if candles[-1].close > 0 else 0

        if stats["total_wick_sizes"]:
            score.avg_wick_size_pct = sum(stats["total_wick_sizes"]) / len(stats["total_wick_sizes"])

        # Quality metrics
        if stats["signal_strengths"]:
            score.avg_signal_strength = sum(stats["signal_strengths"]) / len(stats["signal_strengths"])
        score.atr_ratio_signals = stats["atr_ratio_count"]

        # Calculate overall score
        score.overall_score = self._calculate_overall_score(score)

        return score

    def _calculate_overall_score(self, score: MarketScore) -> float:
        """
        Calculate overall score from individual metrics.

        Weights adjusted for wick reversal strategy:
        - Signal frequency: 35% (MOST important - need volatile assets with lots of wicks)
        - Win rate: 25%
        - Profit factor: 20%
        - ATR/Volatility: 15% (higher volatility = more opportunities)
        - Signal strength: 5%
        """
        # Normalize each component to 0-100 scale

        # Signal frequency score - MOST IMPORTANT for this strategy
        if score.signals_per_day < 1:
            freq_score = score.signals_per_day * 20
        elif score.signals_per_day < 2:
            freq_score = 20 + (score.signals_per_day - 1) * 30
        else:
            freq_score = min(100, 50 + (score.signals_per_day - 2) * 10)

        # Win rate score (target: 65-85%)
        wr_score = min(100, max(0, (score.win_rate - 0.5) * 200))

        # Profit factor score (target: 1.5-3.0)
        pf_score = min(100, max(0, (score.profit_factor - 1.0) * 50))

        # Volatility score
        vol_score = min(100, score.avg_atr_pct * 100 * 20)

        # Signal strength score
        strength_score = min(100, score.avg_signal_strength * 150)

        # Weighted combination
        overall = (
            freq_score * 0.35 +
            wr_score * 0.25 +
            pf_score * 0.20 +
            vol_score * 0.15 +
            strength_score * 0.05
        )

        return overall

    async def scan_all_markets(
        self,
        days: int = 30,
        timeframe: str = "5m",
        min_signals: int = 10,
        min_volume: float = 50000,
        exclude_stables: bool = True,
        max_concurrent: int = 5
    ) -> List[MarketScore]:
        """Scan all markets and return scored results."""
        # Get all markets
        markets = await self.get_all_markets()
        market_info = await self.get_market_info()

        logger.info(f"Found {len(markets)} markets on Hyperliquid")

        # Filter markets
        stables = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "USDD", "TUSD", "BUSD"}
        filtered_markets = []

        for m in markets:
            coin = m["name"]
            info = market_info.get(coin, {})

            if exclude_stables and coin in stables:
                continue

            if info.get("volume_24h", 0) < min_volume:
                continue

            filtered_markets.append(m)

        logger.info(f"Scanning {len(filtered_markets)} markets after filtering")

        # Scan markets with concurrency limit
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scan_with_semaphore(market):
            async with semaphore:
                symbol = market["symbol"]
                try:
                    logger.info(f"Scanning {symbol}...")
                    score = await self.score_market(symbol, market_info, days, timeframe)
                    if score and score.total_signals >= min_signals:
                        return score
                except Exception as e:
                    logger.warning(f"Error scanning {symbol}: {e}")
                return None

        tasks = [scan_with_semaphore(m) for m in filtered_markets]
        scores = await asyncio.gather(*tasks)

        # Filter and sort results
        results = [s for s in scores if s is not None]
        results.sort(key=lambda x: x.overall_score, reverse=True)

        return results


def print_results(results: List[MarketScore], top_n: int = 20):
    """Print formatted results table."""
    print("\n" + "=" * 105)
    print("MARKET SCAN RESULTS - WICK REVERSAL STRATEGY")
    print("=" * 105)
    print(f"\n{'Rank':<5} {'Symbol':<15} {'Score':<8} {'Sig/Day':<8} {'Win%':<8} {'PF':<8} "
          f"{'ATR%':<8} {'Vol(k)':<10} {'PnL%':<8}")
    print("-" * 105)

    for i, score in enumerate(results[:top_n], 1):
        print(f"{i:<5} {score.symbol:<15} {score.overall_score:<8.1f} "
              f"{score.signals_per_day:<8.1f} {score.win_rate*100:<8.1f} "
              f"{score.profit_factor:<8.2f} {score.avg_atr_pct*100:<8.2f} "
              f"{score.avg_volume_usd/1000:<10.0f} {score.total_pnl_pct:<8.1f}")

    print("-" * 105)
    print(f"\nTotal markets analyzed: {len(results)}")
    print(f"Showing top {min(top_n, len(results))} by overall score\n")

    # Print legend
    print("Legend:")
    print("  Score   = Overall profitability score (0-100) - prioritizes signal frequency")
    print("  Sig/Day = Signals per day (MOST IMPORTANT - need volatile assets)")
    print("  Win%    = Historical win rate from simplified simulation")
    print("  PF      = Profit factor (gross wins / gross losses)")
    print("  ATR%    = Average True Range as % of price (volatility)")
    print("  Vol(k)  = Average volume in thousands USD")
    print("  PnL%    = Total simulated PnL percentage")
    print("\n  NOTE: Run full backtest on top picks to get accurate win rates!")


def export_results(results: List[MarketScore], filepath: str):
    """Export results to CSV."""
    with open(filepath, 'w', newline='') as f:
        if not results:
            return

        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for score in results:
            writer.writerow(score.to_dict())

    print(f"\nResults exported to: {filepath}")
