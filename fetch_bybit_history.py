#!/usr/bin/env python3
"""
Fetch historical candles from Bybit and save to JSON for backtesting.

Usage:
    python fetch_bybit_history.py --symbols DOGE-PERP SHIB-PERP PEPE-PERP --timeframe 5m --days 180
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List

from core.types import Candle
from data.ingestion import BybitProvider
from config.paths import HISTORICAL_DATA_FILE


def candle_to_dict(candle: Candle) -> dict:
    return {
        "timestamp": candle.timestamp.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
    }


async def fetch_symbol_history(
    provider: BybitProvider,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    sleep_seconds: float = 0.15,
) -> List[Candle]:
    """Fetch candles with pagination (Bybit returns max 1000 per request)."""
    all_candles: Dict[datetime, Candle] = {}

    # Timeframe to minutes for calculating chunk size
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    minutes = tf_minutes.get(timeframe, 5)

    # Fetch ~900 candles per request to stay under 1000 limit
    chunk_size = timedelta(minutes=minutes * 900)
    cursor = start_time

    while cursor < end_time:
        chunk_end = min(cursor + chunk_size, end_time)

        try:
            candles = await provider.get_historical_candles(
                symbol, timeframe, cursor, chunk_end
            )

            for c in candles:
                all_candles[c.timestamp] = c

            print(f"  {symbol}: fetched to {chunk_end.date()}, {len(all_candles)} total candles")

        except Exception as e:
            print(f"  {symbol}: error fetching {cursor.date()} to {chunk_end.date()}: {e}")

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
    provider = BybitProvider(testnet=False)
    await provider.connect()

    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        print(f"Fetching {days} days of {timeframe} data")
        print(f"From: {start_time.date()} To: {end_time.date()}")
        print("-" * 40)

        all_data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            candles = await fetch_symbol_history(
                provider, symbol, timeframe, start_time, end_time, sleep_seconds
            )
            all_data[symbol] = candles
            print(f"  Complete: {len(candles)} candles\n")

        btc_candles = []
        if include_btc:
            print("Fetching BTC-PERP for correlation filter...")
            btc_candles = await fetch_symbol_history(
                provider, "BTC-PERP", timeframe, start_time, end_time, sleep_seconds
            )
            print(f"  Complete: {len(btc_candles)} candles\n")

        return {
            "meta": {
                "provider": "Bybit",
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "days": days,
            },
            "symbols": {s: [candle_to_dict(c) for c in candles] for s, candles in all_data.items()},
            "btc": [candle_to_dict(c) for c in btc_candles],
        }
    finally:
        await provider.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Bybit historical candles")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["DOGE-PERP", "TAO-PERP", "1000PEPE-PERP"],
        help="Symbols to fetch (use -PERP suffix, will be converted to USDT)",
    )
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (e.g., 5m, 15m, 1h)")
    parser.add_argument("--days", type=int, default=180, help="Number of days of history (default: 180)")
    parser.add_argument("--sleep", type=float, default=0.15, help="Sleep between API requests")
    parser.add_argument("--no-btc", action="store_true", help="Skip BTC-PERP data")
    parser.add_argument("--out", default=str(HISTORICAL_DATA_FILE), help="Output JSON file")
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
