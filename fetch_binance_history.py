#!/usr/bin/env python3
"""
Fetch historical candles from Binance Futures and save to JSON for backtesting.

Usage:
    python fetch_binance_history.py --symbols DOGE-PERP SHIB-PERP PEPE-PERP --timeframe 5m --days 180
"""

import argparse
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List

from core.types import Candle


BASE_URL = "https://fapi.binance.com"


def candle_to_dict(candle: Candle) -> dict:
    return {
        "timestamp": candle.timestamp.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


def convert_symbol(symbol: str) -> str:
    """Convert our symbol format to Binance format."""
    # DOGE-PERP -> DOGEUSDT, SHIB-PERP -> SHIBUSDT, etc.
    return symbol.replace("-PERP", "USDT")


async def fetch_candles_chunk(
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
) -> List[Candle]:
    """Fetch a single chunk of candles from Binance."""
    binance_symbol = convert_symbol(symbol)

    params = {
        "symbol": binance_symbol,
        "interval": timeframe,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1500,  # Binance max
    }

    async with session.get(f"{BASE_URL}/fapi/v1/klines", params=params) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise Exception(f"Binance API error {resp.status}: {text[:200]}")

        data = await resp.json()

    candles = []
    for c in data:
        # Binance format: [open_time, open, high, low, close, volume, close_time, ...]
        candles.append(Candle(
            timestamp=datetime.utcfromtimestamp(c[0] / 1000),
            open=float(c[1]),
            high=float(c[2]),
            low=float(c[3]),
            close=float(c[4]),
            volume=float(c[5]),
        ))

    return candles


async def fetch_symbol_history(
    session: aiohttp.ClientSession,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    sleep_seconds: float = 0.2,
) -> List[Candle]:
    """Fetch full history for a symbol with pagination."""
    all_candles: Dict[datetime, Candle] = {}

    # Timeframe to milliseconds
    tf_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000
    }
    interval_ms = tf_ms.get(timeframe, 300_000)

    # Chunk size: 1400 candles worth (under 1500 limit)
    chunk_ms = interval_ms * 1400

    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    cursor = start_ms

    while cursor < end_ms:
        chunk_end = min(cursor + chunk_ms, end_ms)

        try:
            candles = await fetch_candles_chunk(session, symbol, timeframe, cursor, chunk_end)

            for c in candles:
                all_candles[c.timestamp] = c

            cursor_date = datetime.utcfromtimestamp(cursor / 1000).date()
            print(f"  {symbol}: fetched to {cursor_date}, {len(all_candles)} total candles")

        except Exception as e:
            print(f"  {symbol}: error at {datetime.utcfromtimestamp(cursor/1000).date()}: {e}")

        cursor = chunk_end
        await asyncio.sleep(sleep_seconds)

    return [all_candles[t] for t in sorted(all_candles)]


async def fetch_historical_data(
    symbols: List[str],
    timeframe: str,
    days: int,
    sleep_seconds: float,
    include_btc: bool,
) -> dict:
    """Fetch historical data for all symbols."""

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    print(f"Fetching {days} days of {timeframe} data from Binance Futures")
    print(f"From: {start_time.date()} To: {end_time.date()}")
    print("-" * 40)

    async with aiohttp.ClientSession() as session:
        all_data = {}

        for symbol in symbols:
            print(f"Fetching {symbol}...")
            candles = await fetch_symbol_history(
                session, symbol, timeframe, start_time, end_time, sleep_seconds
            )
            all_data[symbol] = candles
            print(f"  Complete: {len(candles)} candles\n")

        btc_candles = []
        if include_btc:
            print("Fetching BTC-PERP for correlation filter...")
            btc_candles = await fetch_symbol_history(
                session, "BTC-PERP", timeframe, start_time, end_time, sleep_seconds
            )
            print(f"  Complete: {len(btc_candles)} candles\n")

    return {
        "meta": {
            "provider": "Binance",
            "timeframe": timeframe,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "days": days,
        },
        "symbols": {s: [candle_to_dict(c) for c in candles] for s, candles in all_data.items()},
        "btc": [candle_to_dict(c) for c in btc_candles],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance Futures historical candles")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["DOGE-PERP", "SHIB-PERP", "PEPE-PERP"],
        help="Symbols to fetch (use -PERP suffix, will be converted to USDT)",
    )
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=180, help="Days of history (default: 180)")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between API requests")
    parser.add_argument("--no-btc", action="store_true", help="Skip BTC data")
    parser.add_argument("--out", default="historical_data.json", help="Output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()

    payload = asyncio.run(
        fetch_historical_data(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            sleep_seconds=args.sleep,
            include_btc=not args.no_btc,
        )
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    total_candles = sum(len(v) for v in payload["symbols"].values())
    print("-" * 40)
    print(f"Wrote {args.out}")
    print(f"Total: {total_candles} candles across {len(payload['symbols'])} symbols")


if __name__ == "__main__":
    main()
