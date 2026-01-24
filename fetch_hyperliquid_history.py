#!/usr/bin/env python3
"""
Fetch historical candles from Hyperliquid and save to JSON for backtesting.

python fetch_hyperliquid_history.py --symbols DOGE-PERP kSHIB-PERP kPEPE-PERP --timeframe 5m --days 90 --out
  historical_data.json
"""

import argparse
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List

from core.types import Candle
from data.ingestion import HyperliquidProvider


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
    provider: HyperliquidProvider,
    symbol: str,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    chunk_days: int,
    sleep_seconds: float,
) -> List[Candle]:
    """Fetch candles in chunks to avoid API limits."""
    merged: Dict[datetime, Candle] = {}
    cursor = start_time

    while cursor < end_time:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_time)
        candles = await provider.get_historical_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=cursor,
            end_time=chunk_end,
        )

        for candle in candles:
            merged[candle.timestamp] = candle

        cursor = chunk_end
        if sleep_seconds > 0:
            await asyncio.sleep(sleep_seconds)

    return [merged[t] for t in sorted(merged)]




async def fetch_historical_data(
    symbols: List[str],
    timeframe: str,
    days: int,
    chunk_days: int,
    sleep_seconds: float,
    include_btc: bool,
) -> dict:
    provider = HyperliquidProvider(testnet=False)
    await provider.connect()

    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        all_data = {}
        for symbol in symbols:
            print(f"Fetching {symbol}...")
            try:
                candles = await fetch_symbol_history(
                    provider,
                    symbol,
                    timeframe,
                    start_time,
                    end_time,
                    chunk_days,
                    sleep_seconds,
                )
            except Exception as exc:
                print(f"  Failed to fetch {symbol}: {exc}")
                candles = []
            all_data[symbol] = candles
            print(f"  Got {len(candles)} candles")

        btc_candles = []
        if include_btc:
            print("Fetching BTC-PERP...")
            try:
                btc_candles = await fetch_symbol_history(
                    provider,
                    "BTC-PERP",
                    timeframe,
                    start_time,
                    end_time,
                    chunk_days,
                    sleep_seconds,
                )
                print(f"  Got {len(btc_candles)} candles")
            except Exception as exc:
                print(f"  Failed to fetch BTC-PERP: {exc}")

        return {
            "meta": {
                "provider": "Hyperliquid",
                "timeframe": timeframe,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
            "symbols": {s: [candle_to_dict(c) for c in candles] for s, candles in all_data.items()},
            "btc": [candle_to_dict(c) for c in btc_candles],
        }
    finally:
        await provider.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Hyperliquid historical candles")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["DOGE-PERP", "kSHIB-PERP", "kPEPE-PERP"],
        help="Symbols to fetch (e.g., DOGE-PERP kSHIB-PERP)",
    )
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (e.g., 5m)")
    parser.add_argument("--days", type=int, default=90, help="Number of days of history")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per API request")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between requests")
    parser.add_argument("--no-btc", action="store_true", help="Skip BTC-PERP for filters")
    parser.add_argument("--out", default="historical_data.json", help="Output JSON file")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = asyncio.run(
        fetch_historical_data(
            symbols=args.symbols,
            timeframe=args.timeframe,
            days=args.days,
            chunk_days=args.chunk_days,
            sleep_seconds=args.sleep,
            include_btc=not args.no_btc,
        )
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
