#!/usr/bin/env python3
"""
Test script to verify historical candle fetching is returning up-to-date data.
"""

import asyncio
from datetime import datetime, timedelta
from data.ingestion import HyperliquidProvider


async def test_candle_fetch():
    provider = HyperliquidProvider(testnet=False)
    await provider.connect()

    symbols = ["HYPE-PERP", "BTC-PERP", "ETH-PERP"]
    timeframe = "5m"
    num_candles = 100

    print(f"\n{'='*70}")
    print(f"Testing historical candle fetch at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"Requesting {num_candles} candles per symbol")
    print(f"{'='*70}\n")

    for symbol in symbols:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5 * num_candles)  # 5m candles

        print(f"{symbol}:")
        print(f"  Request range: {start_time.strftime('%H:%M:%S')} to {end_time.strftime('%H:%M:%S')} UTC")

        candles = await provider.get_historical_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        if candles:
            oldest = candles[0]
            newest = candles[-1]
            now = datetime.utcnow()

            # How old is the newest candle?
            newest_age = now - newest.timestamp
            oldest_age = now - oldest.timestamp

            print(f"  Candles returned: {len(candles)}")
            print(f"  Oldest candle: {oldest.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({oldest_age.total_seconds()/60:.1f} min ago)")
            print(f"  Newest candle: {newest.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ({newest_age.total_seconds()/60:.1f} min ago)")

            # Expected: newest candle should be within ~10 minutes (current candle is in progress)
            if newest_age.total_seconds() > 600:  # More than 10 minutes old
                print(f"  ⚠️  WARNING: Newest candle is {newest_age.total_seconds()/60:.1f} minutes old!")
            else:
                print(f"  ✓ Data appears current")

            # Show a few recent candle timestamps
            print(f"  Last 5 candle timestamps:")
            for c in candles[-5:]:
                age = (now - c.timestamp).total_seconds() / 60
                print(f"    {c.timestamp.strftime('%H:%M:%S')} ({age:.1f} min ago) - close: ${c.close:.4f}")
        else:
            print(f"  ❌ No candles returned!")

        print()

    await provider.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(test_candle_fetch())
