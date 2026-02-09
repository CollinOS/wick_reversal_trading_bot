"""
Order Execution Module
Handles order creation, submission, and management.
"""

import asyncio
import logging
import math
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
import uuid

from core.types import (
    Order, OrderType, OrderStatus, Side, Position, PositionStatus, Signal
)
from config.settings import StrategyConfig, ExecutionConfig

# Import official Hyperliquid SDK
try:
    from hyperliquid.info import Info
    from hyperliquid.exchange import Exchange
    from hyperliquid.utils import constants
    from eth_account import Account
    HYPERLIQUID_SDK_AVAILABLE = True
except ImportError:
    HYPERLIQUID_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


# Hyperliquid asset index mapping (mainnet)
# This maps symbol names to their numeric asset indices
# Updated list - fetch dynamically in production
HYPERLIQUID_ASSET_INDEX = {
    "BTC-PERP": 0,
    "ETH-PERP": 1,
    "ATOM-PERP": 2,
    "MATIC-PERP": 3,
    "DYDX-PERP": 4,
    "SOL-PERP": 5,
    "AVAX-PERP": 6,
    "BNB-PERP": 7,
    "APE-PERP": 8,
    "OP-PERP": 9,
    "LTC-PERP": 10,
    "ARB-PERP": 11,
    "DOGE-PERP": 12,
    "INJ-PERP": 13,
    "SUI-PERP": 14,
    "kPEPE-PERP": 15,
    "LINK-PERP": 16,
    "CRV-PERP": 17,
    "LDO-PERP": 18,
    "AAVE-PERP": 19,
    "MKR-PERP": 20,
    "GMX-PERP": 21,
    "FTM-PERP": 22,
    "TIA-PERP": 23,
    "BLUR-PERP": 24,
    "SEI-PERP": 25,
    "RUNE-PERP": 26,
    "ORDI-PERP": 27,
    "SATS-PERP": 28,
    "WLD-PERP": 29,
    "NEAR-PERP": 30,
    "TAO-PERP": 31,
    "DOT-PERP": 32,
    "XRP-PERP": 33,
    "MEME-PERP": 34,
    "PYTH-PERP": 35,
    "JTO-PERP": 36,
    "STX-PERP": 37,
    "WIF-PERP": 38,
    "kSHIB-PERP": 39,
    "JUP-PERP": 40,
    "STRK-PERP": 41,
    "DYM-PERP": 42,
    "MANTA-PERP": 43,
    "ALT-PERP": 44,
    "PIXEL-PERP": 45,
    "PENDLE-PERP": 46,
    "ONDO-PERP": 47,
    "ENA-PERP": 48,
    "W-PERP": 49,
    "ETHFI-PERP": 50,
    "ZRO-PERP": 51,
    "BOME-PERP": 52,
    "TON-PERP": 53,
    "TNSR-PERP": 54,
    "SAGA-PERP": 55,
    "REZ-PERP": 56,
    "IO-PERP": 57,
    "ZK-PERP": 58,
    "BRETT-PERP": 59,
    "BLAST-PERP": 60,
    "LISTA-PERP": 61,
    "NOT-PERP": 62,
    "RENDER-PERP": 63,
    "MEW-PERP": 64,
    "POPCAT-PERP": 65,
    "PEOPLE-PERP": 66,
    "TURBO-PERP": 67,
    "NEIRO-PERP": 68,
    "DOGS-PERP": 69,
    "POL-PERP": 70,
    "EIGEN-PERP": 71,
    "SCR-PERP": 72,
    "APT-PERP": 73,
    "kBONK-PERP": 74,
    "GOAT-PERP": 75,
    "GRASS-PERP": 76,
    "MOODENG-PERP": 77,
    "HYPE-PERP": 78,
    "PNUT-PERP": 79,
    "VIRTUAL-PERP": 80,
    "AI16Z-PERP": 81,
    "FARTCOIN-PERP": 82,
    "TRUMP-PERP": 83,
    "MELANIA-PERP": 84,
    "ANIME-PERP": 85,
    "VINE-PERP": 86,
}


@dataclass
class ExecutionResult:
    """Result of an order execution attempt."""
    success: bool
    order: Optional[Order] = None
    error: str = ""
    latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "order_id": self.order.id if self.order else None,
            "error": self.error,
            "latency_ms": self.latency_ms
        }


class ExecutionHandler(ABC):
    """Abstract base class for exchange execution handlers."""
    
    @abstractmethod
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit an order to the exchange."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get current status of an order."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        pass
    
    @abstractmethod
    async def close_position(self, symbol: str, side: Side) -> ExecutionResult:
        """Close entire position with market order."""
        pass


class HyperliquidExecutor(ExecutionHandler):
    """
    Execution handler for Hyperliquid DEX.

    Uses the official Hyperliquid Python SDK for order management.
    Requires wallet private key for signing transactions.
    """

    def __init__(
        self,
        private_key: str,
        testnet: bool = True,
        vault_address: Optional[str] = None
    ):
        if not HYPERLIQUID_SDK_AVAILABLE:
            raise ImportError(
                "hyperliquid-python-sdk is required for live trading. "
                "Install with: pip install hyperliquid-python-sdk"
            )

        # Normalize private key format
        if not private_key.startswith("0x"):
            private_key = "0x" + private_key

        self.private_key = private_key
        self.testnet = testnet

        # Derive wallet address from private key
        self.account = Account.from_key(private_key)
        self.wallet_address = self.account.address

        # Vault address (for sub-accounts, usually same as wallet)
        self.vault_address = vault_address

        # SDK instances (initialized in connect())
        self.info: Optional[Info] = None
        self.exchange: Optional[Exchange] = None

        # Set base URL based on network
        self.base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL

        # Asset metadata cache
        self._asset_meta: Dict[str, dict] = {}
        self._coin_to_asset: Dict[str, int] = {}

        # Execution config (set via set_execution_config)
        self._execution_config: Optional[ExecutionConfig] = None

        logger.info(f"Hyperliquid executor initialized for wallet: {self.wallet_address}")

    def set_execution_config(self, config: ExecutionConfig):
        """Set execution config for dynamic leverage settings."""
        self._execution_config = config

    async def connect(self, max_retries: int = 5):
        """Initialize SDK and fetch asset metadata with retry logic for rate limits."""
        for attempt in range(max_retries):
            try:
                # Initialize Info API (read-only, no auth needed)
                self.info = Info(self.base_url, skip_ws=True)

                # Initialize Exchange API (requires wallet for signing)
                self.exchange = Exchange(
                    self.account,
                    self.base_url,
                    vault_address=self.vault_address
                )

                # Fetch asset metadata
                await self._fetch_asset_metadata()

                logger.info(f"Hyperliquid executor connected ({'testnet' if self.testnet else 'mainnet'})")
                logger.info(f"Wallet address: {self.wallet_address}")
                return

            except Exception as e:
                error_str = str(e)
                # Check for rate limit error (429)
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
                    logger.warning(f"Rate limited by Hyperliquid API, waiting {wait_time}s before retry ({attempt + 1}/{max_retries})")
                    print(f"  Rate limited by API, waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    # Non-rate-limit error, raise immediately
                    raise

        raise Exception(f"Failed to connect to Hyperliquid after {max_retries} retries due to rate limiting")

    async def disconnect(self):
        """Cleanup resources."""
        # SDK doesn't require explicit disconnect
        pass

    async def _fetch_asset_metadata(self):
        """Fetch asset metadata from Hyperliquid API."""
        try:
            # Get metadata (this is sync in the SDK)
            meta = self.info.meta()

            # Build lookup maps
            if "universe" in meta:
                for idx, asset in enumerate(meta["universe"]):
                    coin = asset["name"]
                    symbol = f"{coin}-PERP"
                    self._coin_to_asset[coin] = idx
                    self._asset_meta[symbol] = {
                        "index": idx,
                        "coin": coin,
                        "szDecimals": asset.get("szDecimals", 4),
                    }
                logger.info(f"Loaded {len(self._asset_meta)} asset indices from exchange")

            # Also fetch asset contexts for tick sizes
            try:
                # meta_and_asset_ctxs returns both meta and asset contexts
                meta_and_ctxs = self.info.meta_and_asset_ctxs()
                if len(meta_and_ctxs) > 1:
                    asset_ctxs = meta_and_ctxs[1]
                    universe = meta_and_ctxs[0].get("universe", [])
                    tick_sizes_loaded = 0
                    for idx, ctx in enumerate(asset_ctxs):
                        if idx < len(universe):
                            coin = universe[idx]["name"]
                            symbol = f"{coin}-PERP"
                            if symbol in self._asset_meta:
                                # Store tick size (minimum price increment)
                                tick_size = float(ctx.get("tickSize", 0.0001))
                                self._asset_meta[symbol]["tickSize"] = tick_size
                                tick_sizes_loaded += 1
                    logger.info(f"Loaded tick sizes for {tick_sizes_loaded} assets")
            except Exception as e:
                logger.warning(f"Failed to fetch tick sizes: {e}")

        except Exception as e:
            logger.warning(f"Failed to fetch asset metadata: {e}")

    def _get_coin(self, symbol: str) -> str:
        """Convert symbol to coin name (e.g., 'TAO-PERP' -> 'TAO')."""
        return symbol.replace("-PERP", "")

    def _round_size(self, size: float, symbol: str) -> float:
        """Round size to appropriate decimals for the asset."""
        # Get size decimals from metadata if available
        if symbol in self._asset_meta:
            sz_decimals = self._asset_meta[symbol].get("szDecimals", 4)
        else:
            # Default based on size magnitude
            if size >= 1000:
                sz_decimals = 1
            elif size >= 100:
                sz_decimals = 2
            elif size >= 1:
                sz_decimals = 3
            else:
                sz_decimals = 4

        return round(size, sz_decimals)

    def _round_price(self, price: float, symbol: str) -> float:
        """Round price to tick size for the asset."""
        # Get tick size from metadata if available
        tick_size = 0.0001  # default
        if symbol in self._asset_meta:
            tick_size = self._asset_meta[symbol].get("tickSize", 0.0001)

        if tick_size > 0:
            # Round to nearest tick
            rounded = round(price / tick_size) * tick_size
        else:
            # Fallback: round based on magnitude
            if price >= 10000:
                rounded = round(price, 1)
            elif price >= 1000:
                rounded = round(price, 2)
            elif price >= 100:
                rounded = round(price, 3)
            elif price >= 10:
                rounded = round(price, 4)
            elif price >= 1:
                rounded = round(price, 5)
            else:
                rounded = round(price, 6)

        # Hyperliquid requires max 5 significant figures
        rounded = self._round_to_sig_figs(rounded, 5)
        return rounded

    def _round_to_sig_figs(self, value: float, sig_figs: int = 5) -> float:
        """Round value to specified number of significant figures."""
        if value == 0:
            return 0
        magnitude = math.floor(math.log10(abs(value)))
        factor = 10 ** (sig_figs - 1 - magnitude)
        return round(value * factor) / factor

    def calculate_leverage_from_confidence(self, dynamic_multiplier: float) -> int:
        """
        Calculate leverage based on trade confidence (dynamic multiplier).

        Maps the dynamic multiplier range to leverage range:
        - Low confidence (multiplier ~1.0) -> min_leverage (3x)
        - High confidence (multiplier ~2.5) -> max_leverage (5x)
        """
        if not self._execution_config or not self._execution_config.dynamic_leverage_enabled:
            return 3  # Default to conservative leverage

        min_lev = self._execution_config.min_leverage
        max_lev = self._execution_config.max_leverage

        # Dynamic multiplier typically ranges from 0.5 to 2.5
        # Map this to leverage range
        min_mult = 0.5
        max_mult = 2.5

        # Normalize multiplier to 0-1 range
        normalized = (dynamic_multiplier - min_mult) / (max_mult - min_mult)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to 0-1

        # Map to leverage range
        leverage = min_lev + normalized * (max_lev - min_lev)
        return int(round(leverage))

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol on Hyperliquid.

        Args:
            symbol: Trading pair (e.g., 'ZRO-PERP')
            leverage: Leverage to set (e.g., 3, 5)

        Returns:
            True if successful, False otherwise
        """
        try:
            name = self._get_coin(symbol)

            # Get max leverage for this asset to avoid exceeding it
            meta = self._asset_meta.get(symbol, {})
            max_leverage = meta.get("maxLeverage", 50)
            leverage = min(leverage, max_leverage)

            result = self.exchange.update_leverage(leverage, name, is_cross=True)
            logger.debug(f"Set leverage response for {symbol}: {result}")

            if result.get("status") == "ok":
                logger.info(f"Set {symbol} leverage to {leverage}x")
                return True
            else:
                error = result.get("response", str(result))
                logger.warning(f"Failed to set leverage for {symbol}: {error}")
                return False

        except Exception as e:
            logger.warning(f"Error setting leverage for {symbol}: {e}")
            return False

    async def submit_order(self, order: Order, dynamic_multiplier: float = None) -> ExecutionResult:
        """Submit order to Hyperliquid using official SDK.

        Args:
            order: The order to submit
            dynamic_multiplier: Optional confidence multiplier to set leverage (only for entries)
        """
        start_time = datetime.utcnow()

        try:
            # Set leverage based on confidence for entry orders (not reduce-only)
            if dynamic_multiplier is not None and self._execution_config:
                if self._execution_config.dynamic_leverage_enabled:
                    leverage = self.calculate_leverage_from_confidence(dynamic_multiplier)
                    await self.set_leverage(order.symbol, leverage)

            # SDK uses 'name' parameter (coin name without -PERP)
            name = self._get_coin(order.symbol)
            is_buy = order.side == Side.LONG

            # Round size to avoid precision errors
            rounded_size = self._round_size(order.quantity, order.symbol)

            # Determine order type and parameters
            if order.order_type == OrderType.MARKET:
                # Market order - use market_open from SDK
                result = self.exchange.market_open(
                    name=name,
                    is_buy=is_buy,
                    sz=rounded_size,
                    slippage=0.01  # 1% slippage tolerance
                )
            elif order.order_type == OrderType.LIMIT:
                # Round price too
                rounded_price = self._round_price(order.price, order.symbol)
                # Limit order
                result = self.exchange.order(
                    name=name,
                    is_buy=is_buy,
                    sz=rounded_size,
                    limit_px=rounded_price,
                    order_type={"limit": {"tif": "Gtc"}}
                )
            else:
                return ExecutionResult(
                    success=False,
                    order=order,
                    error=f"Unsupported order type: {order.order_type}",
                    latency_ms=0
                )

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            logger.debug(f"Order response: {result}")

            # Parse result
            if result.get("status") == "ok":
                response_data = result.get("response", {})
                if response_data.get("type") == "order":
                    statuses = response_data.get("data", {}).get("statuses", [])
                    if statuses:
                        status_info = statuses[0]
                        if "resting" in status_info:
                            order.exchange_order_id = str(status_info["resting"]["oid"])
                            order.status = OrderStatus.OPEN
                        elif "filled" in status_info:
                            order.exchange_order_id = str(status_info["filled"]["oid"])
                            order.status = OrderStatus.FILLED
                            order.filled_quantity = order.quantity
                            order.average_fill_price = float(status_info["filled"].get("avgPx", order.price or 0))
                            order.filled_at = datetime.utcnow()
                        elif "error" in status_info:
                            order.status = OrderStatus.REJECTED
                            return ExecutionResult(
                                success=False,
                                order=order,
                                error=status_info["error"],
                                latency_ms=latency
                            )

                order.updated_at = datetime.utcnow()
                logger.info(f"Order submitted: {order.symbol} {order.side.value} {order.quantity} @ {order.price or 'MARKET'} -> {order.exchange_order_id}")
                return ExecutionResult(success=True, order=order, latency_ms=latency)
            else:
                error = result.get("response", str(result))
                if isinstance(error, dict):
                    error = error.get("error", str(error))
                logger.error(f"Order failed: {error}")
                order.status = OrderStatus.REJECTED
                return ExecutionResult(success=False, order=order, error=str(error), latency_ms=latency)

        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.exception(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            return ExecutionResult(success=False, order=order, error=str(e), latency_ms=latency)
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order using official SDK."""
        try:
            name = self._get_coin(symbol)

            # Parse order ID
            try:
                oid = int(order_id)
            except ValueError:
                logger.error(f"Invalid order ID format: {order_id}")
                return False

            # Use SDK to cancel
            result = self.exchange.cancel(name=name, oid=oid)

            if result.get("status") == "ok":
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                error = result.get("response", str(result))
                logger.error(f"Cancel failed: {error}")
                return False

        except Exception as e:
            logger.exception(f"Cancel order error: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status from exchange using SDK."""
        try:
            # Get open orders
            open_orders = self.info.open_orders(self.wallet_address)

            # Find the order in open orders
            for open_order in open_orders:
                if str(open_order.get("oid")) == str(order_id):
                    order = Order(
                        id=order_id,
                        symbol=symbol,
                        exchange_order_id=str(open_order.get("oid"))
                    )
                    order.status = OrderStatus.OPEN
                    order.filled_quantity = float(open_order.get("filled", 0))
                    order.price = float(open_order.get("limitPx", 0)) or None
                    return order

            # If not in open orders, check fills history
            fills = self.info.user_fills(self.wallet_address)

            for fill in fills:
                if str(fill.get("oid")) == str(order_id):
                    order = Order(
                        id=order_id,
                        symbol=symbol,
                        exchange_order_id=str(fill.get("oid"))
                    )
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = float(fill.get("sz", 0))
                    order.average_fill_price = float(fill.get("px", 0))
                    return order

            # Order not found - may have been cancelled
            return None

        except Exception as e:
            logger.exception(f"Get order status error: {e}")
            return None

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position using SDK."""
        try:
            # Get user state from SDK
            user_state = self.info.user_state(self.wallet_address)
            coin = self._get_coin(symbol)

            # Check assetPositions
            asset_positions = user_state.get("assetPositions", [])
            for pos in asset_positions:
                position_data = pos.get("position", pos)
                if position_data.get("coin") == coin:
                    size = float(position_data.get("szi", 0))
                    if abs(size) < 1e-10:
                        return None

                    return {
                        "symbol": symbol,
                        "size": size,
                        "entry_price": float(position_data.get("entryPx", 0)),
                        "unrealized_pnl": float(position_data.get("unrealizedPnl", 0)),
                        "liquidation_price": float(position_data.get("liquidationPx") or 0),
                        "margin_used": float(position_data.get("marginUsed", 0)),
                    }
            return None

        except Exception as e:
            logger.exception(f"Get position error: {e}")
            return None

    async def get_account_balance(self) -> Optional[Dict]:
        """Get account balance and margin info using SDK."""
        try:
            # Get user state from SDK
            user_state = self.info.user_state(self.wallet_address)
            margin_summary = user_state.get("marginSummary", {})

            return {
                "account_value": float(margin_summary.get("accountValue", 0)),
                "total_margin_used": float(margin_summary.get("totalMarginUsed", 0)),
                "total_ntl_pos": float(margin_summary.get("totalNtlPos", 0)),
                "withdrawable": float(margin_summary.get("withdrawable", 0)),
            }

        except Exception as e:
            logger.exception(f"Get account balance error: {e}")
            return None
    
    async def close_position(self, symbol: str, side: Side) -> ExecutionResult:
        """Close position with market order."""
        position = await self.get_position(symbol)
        if not position or abs(position["size"]) < 1e-10:
            return ExecutionResult(success=False, error="No position to close")

        # Determine close side based on current position
        # If we're long (positive size), we need to sell (SHORT)
        # If we're short (negative size), we need to buy (LONG)
        if position["size"] > 0:
            close_side = Side.SHORT
        else:
            close_side = Side.LONG

        order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=abs(position["size"])
        )

        return await self.submit_order(order)

    async def verify_connection(self) -> bool:
        """Verify that we can connect and authenticate with the exchange."""
        try:
            # Check if we can fetch account state
            balance = await self.get_account_balance()
            if balance is None:
                logger.error("Failed to fetch account balance")
                return False

            logger.info(f"Connection verified. Account value: ${balance['account_value']:,.2f}")
            logger.info(f"Withdrawable: ${balance['withdrawable']:,.2f}")
            return True

        except Exception as e:
            logger.exception(f"Connection verification failed: {e}")
            return False

    async def get_all_positions(self) -> List[Dict]:
        """Get all open positions using SDK."""
        try:
            # Get user state from SDK
            user_state = self.info.user_state(self.wallet_address)

            positions = []
            for pos in user_state.get("assetPositions", []):
                position_data = pos.get("position", pos)
                size = float(position_data.get("szi", 0))
                if abs(size) > 1e-10:
                    positions.append({
                        "symbol": f"{position_data.get('coin')}-PERP",
                        "size": size,
                        "entry_price": float(position_data.get("entryPx", 0)),
                        "unrealized_pnl": float(position_data.get("unrealizedPnl", 0)),
                        "liquidation_price": float(position_data.get("liquidationPx") or 0),
                    })

            return positions

        except Exception as e:
            logger.exception(f"Get all positions error: {e}")
            return []

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders using SDK."""
        try:
            # Get all open orders from SDK
            orders = self.info.open_orders(self.wallet_address)

            cancelled = 0
            for order in orders:
                order_symbol = f"{order.get('coin')}-PERP"

                # Filter by symbol if specified
                if symbol and order_symbol != symbol:
                    continue

                oid = str(order.get("oid"))
                if await self.cancel_order(oid, order_symbol):
                    cancelled += 1

            logger.info(f"Cancelled {cancelled} orders")
            return cancelled

        except Exception as e:
            logger.exception(f"Cancel all orders error: {e}")
            return 0

    async def get_recent_fills(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent fills for a symbol to determine actual exit price and fees."""
        try:
            fills = self.info.user_fills(self.wallet_address)
            coin = self._get_coin(symbol)

            # Filter by symbol and get most recent
            symbol_fills = [
                {
                    "price": float(f.get("px", 0)),
                    "size": float(f.get("sz", 0)),
                    "side": f.get("side", ""),
                    "fee": float(f.get("fee", 0)),
                    "timestamp": f.get("time", 0),
                    "oid": f.get("oid"),
                    "closed_pnl": float(f.get("closedPnl", 0))
                }
                for f in fills
                if f.get("coin") == coin
            ]

            # Sort by timestamp descending and limit
            symbol_fills.sort(key=lambda x: x["timestamp"], reverse=True)
            return symbol_fills[:limit]

        except Exception as e:
            logger.warning(f"Failed to get recent fills for {symbol}: {e}")
            return []

    async def submit_stop_loss_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        trigger_price: float
    ) -> ExecutionResult:
        """
        Submit a stop-loss trigger order to Hyperliquid.

        This places an order on the exchange that will trigger when price hits the stop.
        The exchange monitors price in real-time, providing instant execution even if
        the bot is offline or slow.

        Args:
            symbol: Trading pair (e.g., 'TAO-PERP')
            side: Side of the CLOSING order (SHORT to close a long, LONG to close a short)
            quantity: Size to close
            trigger_price: Price at which to trigger the stop
        """
        start_time = datetime.utcnow()

        try:
            name = self._get_coin(symbol)
            is_buy = side == Side.LONG  # If closing a short, we buy
            rounded_size = self._round_size(quantity, symbol)
            rounded_trigger = self._round_price(trigger_price, symbol)

            # Log the tick size and rounding for debugging
            meta = self._asset_meta.get(symbol, {})
            tick_size = meta.get("tickSize", "unknown")
            if symbol not in self._asset_meta:
                logger.warning(f"No metadata for {symbol}, using default tick size")

            # Check if we should use limit orders for exits (reduces slippage)
            use_limit = getattr(self._execution_config, 'use_limit_orders_for_exits', False) if self._execution_config else False
            buffer_pct = getattr(self._execution_config, 'stop_limit_buffer_pct', 0.005) if self._execution_config else 0.005

            if use_limit:
                # Calculate limit price with buffer to ensure fill
                # For buying (closing short): limit above trigger
                # For selling (closing long): limit below trigger
                if is_buy:
                    limit_price = rounded_trigger * (1 + buffer_pct)
                else:
                    limit_price = rounded_trigger * (1 - buffer_pct)
                limit_price = self._round_price(limit_price, symbol)
                logger.info(f"Stop loss {symbol}: trigger={rounded_trigger:.6f}, limit={limit_price:.6f} (LIMIT ORDER), is_buy={is_buy}")
            else:
                limit_price = rounded_trigger
                logger.info(f"Stop loss {symbol}: trigger={rounded_trigger:.6f} (MARKET ORDER), is_buy={is_buy}")

            # Use trigger order type for stop loss
            # tpsl: "sl" indicates this is a stop loss order
            result = self.exchange.order(
                name=name,
                is_buy=is_buy,
                sz=rounded_size,
                limit_px=limit_price,
                order_type={
                    "trigger": {
                        "triggerPx": rounded_trigger,
                        "isMarket": not use_limit,  # Use limit order if configured
                        "tpsl": "sl"  # Mark as stop loss
                    }
                },
                reduce_only=True  # Stop loss should only reduce position
            )

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(f"Stop loss order response: {result}")

            if result.get("status") == "ok":
                response_data = result.get("response", {})
                if response_data.get("type") == "order":
                    statuses = response_data.get("data", {}).get("statuses", [])
                    if statuses:
                        status_info = statuses[0]
                        if "resting" in status_info:
                            oid = status_info["resting"]["oid"]
                            logger.info(f"Stop loss placed: {symbol} @ ${rounded_trigger:.2f} (oid: {oid})")

                            # Create order object for tracking
                            order = Order(
                                symbol=symbol,
                                side=side,
                                order_type=OrderType.STOP_MARKET,
                                quantity=rounded_size,
                                stop_price=rounded_trigger
                            )
                            order.exchange_order_id = str(oid)
                            order.status = OrderStatus.OPEN

                            return ExecutionResult(success=True, order=order, latency_ms=latency)
                        elif "error" in status_info:
                            return ExecutionResult(
                                success=False,
                                error=f"Stop loss rejected: {status_info['error']}",
                                latency_ms=latency
                            )
                        else:
                            # Unknown status - log it for debugging
                            logger.warning(f"Stop loss unknown status: {status_info}")
                            return ExecutionResult(
                                success=False,
                                error=f"Unknown order status: {status_info}",
                                latency_ms=latency
                            )
                    else:
                        # No statuses returned
                        logger.warning(f"Stop loss order returned no statuses: {response_data}")
                        return ExecutionResult(
                            success=False,
                            error=f"No order status returned: {response_data}",
                            latency_ms=latency
                        )
                else:
                    # Unexpected response type
                    logger.warning(f"Stop loss unexpected response type: {response_data}")
                    return ExecutionResult(
                        success=False,
                        error=f"Unexpected response type: {response_data.get('type')}",
                        latency_ms=latency
                    )
            else:
                error = result.get("response", str(result))
                if isinstance(error, dict):
                    error = error.get("error", str(error))
                return ExecutionResult(success=False, error=str(error), latency_ms=latency)

        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.exception(f"Stop loss order error: {e}")
            return ExecutionResult(success=False, error=str(e), latency_ms=latency)

    async def submit_take_profit_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        trigger_price: float
    ) -> ExecutionResult:
        """
        Submit a take-profit trigger order to Hyperliquid.

        Args:
            symbol: Trading pair (e.g., 'TAO-PERP')
            side: Side of the CLOSING order (SHORT to close a long, LONG to close a short)
            quantity: Size to close
            trigger_price: Price at which to trigger the take profit
        """
        start_time = datetime.utcnow()

        try:
            name = self._get_coin(symbol)
            is_buy = side == Side.LONG
            rounded_size = self._round_size(quantity, symbol)
            rounded_trigger = self._round_price(trigger_price, symbol)

            # Log the tick size and rounding for debugging
            meta = self._asset_meta.get(symbol, {})
            tick_size = meta.get("tickSize", "unknown")
            if symbol not in self._asset_meta:
                logger.warning(f"No metadata for {symbol}, using default tick size")

            # Check if we should use limit orders for exits (reduces slippage)
            use_limit = getattr(self._execution_config, 'use_limit_orders_for_exits', False) if self._execution_config else False

            if use_limit:
                # For take profits, use trigger price as limit (no buffer needed - we want exact price or better)
                logger.info(f"Take profit {symbol}: trigger={rounded_trigger:.6f} (LIMIT ORDER), is_buy={is_buy}")
            else:
                logger.info(f"Take profit {symbol}: trigger={rounded_trigger:.6f} (MARKET ORDER), is_buy={is_buy}")

            # Use trigger order type for take profit
            result = self.exchange.order(
                name=name,
                is_buy=is_buy,
                sz=rounded_size,
                limit_px=rounded_trigger,
                order_type={
                    "trigger": {
                        "triggerPx": rounded_trigger,
                        "isMarket": not use_limit,  # Use limit order if configured
                        "tpsl": "tp"  # Mark as take profit
                    }
                },
                reduce_only=True
            )

            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.debug(f"Take profit order response: {result}")

            if result.get("status") == "ok":
                response_data = result.get("response", {})
                if response_data.get("type") == "order":
                    statuses = response_data.get("data", {}).get("statuses", [])
                    if statuses:
                        status_info = statuses[0]
                        if "resting" in status_info:
                            oid = status_info["resting"]["oid"]
                            logger.info(f"Take profit placed: {symbol} @ ${rounded_trigger:.2f} (oid: {oid})")

                            order = Order(
                                symbol=symbol,
                                side=side,
                                order_type=OrderType.LIMIT,
                                quantity=rounded_size,
                                price=rounded_trigger
                            )
                            order.exchange_order_id = str(oid)
                            order.status = OrderStatus.OPEN

                            return ExecutionResult(success=True, order=order, latency_ms=latency)
                        elif "error" in status_info:
                            return ExecutionResult(
                                success=False,
                                error=f"Take profit rejected: {status_info['error']}",
                                latency_ms=latency
                            )
                        else:
                            # Unknown status - log it for debugging
                            logger.warning(f"Take profit unknown status: {status_info}")
                            return ExecutionResult(
                                success=False,
                                error=f"Unknown order status: {status_info}",
                                latency_ms=latency
                            )
                    else:
                        # No statuses returned
                        logger.warning(f"Take profit order returned no statuses: {response_data}")
                        return ExecutionResult(
                            success=False,
                            error=f"No order status returned: {response_data}",
                            latency_ms=latency
                        )
                else:
                    # Unexpected response type
                    logger.warning(f"Take profit unexpected response type: {response_data}")
                    return ExecutionResult(
                        success=False,
                        error=f"Unexpected response type: {response_data.get('type')}",
                        latency_ms=latency
                    )
            else:
                error = result.get("response", str(result))
                if isinstance(error, dict):
                    error = error.get("error", str(error))
                return ExecutionResult(success=False, error=str(error), latency_ms=latency)

        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.exception(f"Take profit order error: {e}")
            return ExecutionResult(success=False, error=str(e), latency_ms=latency)


class SimulatedExecutor(ExecutionHandler):
    """
    Simulated execution for backtesting and paper trading.
    Models slippage, partial fills, and latency.
    """
    
    def __init__(self, config: ExecutionConfig, get_price_func: Callable[[str], float]):
        self.config = config
        self.get_price = get_price_func
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict] = {}
        self.order_counter = 0
    
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Simulate order submission."""
        start_time = datetime.utcnow()
        
        # Simulate latency
        await asyncio.sleep(0.01)  # 10ms simulated latency
        
        self.order_counter += 1
        order.exchange_order_id = f"SIM-{self.order_counter}"
        
        current_price = self.get_price(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            # Immediate fill with slippage
            slippage = current_price * self.config.expected_slippage_pct
            if order.side == Side.LONG:
                fill_price = current_price + slippage
            else:
                fill_price = current_price - slippage
            
            order.average_fill_price = fill_price
            order.filled_quantity = order.quantity
            order.slippage = slippage
            order.commission = fill_price * order.quantity * 0.0006  # 0.06% fee
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.utcnow()
            
            # Update simulated position
            self._update_position(order)
        
        elif order.order_type == OrderType.LIMIT:
            # Check if limit order would fill immediately
            if order.side == Side.LONG and order.price >= current_price:
                order.average_fill_price = order.price
                order.filled_quantity = order.quantity
                order.commission = order.price * order.quantity * 0.0002  # 0.02% maker
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self._update_position(order)
            elif order.side == Side.SHORT and order.price <= current_price:
                order.average_fill_price = order.price
                order.filled_quantity = order.quantity
                order.commission = order.price * order.quantity * 0.0002
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.utcnow()
                self._update_position(order)
            else:
                order.status = OrderStatus.OPEN
                self.orders[order.id] = order
        
        elif order.order_type == OrderType.STOP_MARKET:
            order.status = OrderStatus.OPEN
            self.orders[order.id] = order
        
        order.updated_at = datetime.utcnow()
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ExecutionResult(success=True, order=order, latency_ms=latency)
    
    def _update_position(self, order: Order):
        """Update simulated position after fill."""
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {"size": 0, "entry_price": 0, "cost_basis": 0}
        
        pos = self.positions[symbol]
        fill_value = order.filled_quantity * order.average_fill_price
        
        if order.side == Side.LONG:
            new_size = pos["size"] + order.filled_quantity
            if new_size > 0:
                pos["entry_price"] = (pos["cost_basis"] + fill_value) / new_size
                pos["cost_basis"] += fill_value
            pos["size"] = new_size
        else:
            new_size = pos["size"] - order.filled_quantity
            if abs(new_size) > 0.0001:
                pos["entry_price"] = (pos["cost_basis"] - fill_value) / abs(new_size)
                pos["cost_basis"] -= fill_value
            else:
                pos["entry_price"] = 0
                pos["cost_basis"] = 0
            pos["size"] = new_size
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a simulated order."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            del self.orders[order_id]
            return True
        return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get status of simulated order."""
        return self.orders.get(order_id)
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get simulated position."""
        if symbol in self.positions and self.positions[symbol]["size"] != 0:
            return {
                "symbol": symbol,
                "size": self.positions[symbol]["size"],
                "entry_price": self.positions[symbol]["entry_price"]
            }
        return None
    
    async def close_position(self, symbol: str, side: Side) -> ExecutionResult:
        """Close simulated position."""
        pos = await self.get_position(symbol)
        if not pos:
            return ExecutionResult(success=False, error="No position")
        
        close_side = Side.SHORT if side == Side.LONG else Side.LONG
        order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=abs(pos["size"])
        )
        
        return await self.submit_order(order)
    
    def check_stop_orders(self, symbol: str, current_price: float):
        """Check and trigger stop orders (called on each price update)."""
        triggered = []
        for order_id, order in list(self.orders.items()):
            if order.symbol != symbol or order.order_type != OrderType.STOP_MARKET:
                continue
            
            if order.side == Side.LONG and current_price >= order.stop_price:
                triggered.append(order)
            elif order.side == Side.SHORT and current_price <= order.stop_price:
                triggered.append(order)
        
        return triggered


class OrderManager:
    """
    Manages order lifecycle and execution.
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        executor: ExecutionHandler
    ):
        self.config = config
        self.executor = executor
        
        # Order tracking
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.cancelled_orders: Dict[str, Order] = {}
        
        # Execution statistics
        self.total_orders = 0
        self.successful_fills = 0
        self.failed_orders = 0
        self.total_slippage = 0.0
        self.total_latency = 0.0
    
    async def create_entry_order(
        self,
        signal: Signal,
        position_size: float
    ) -> Order:
        """Create entry order from signal."""
        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            quantity=position_size,
            signal_id=signal.id,
            created_at=datetime.utcnow()
        )
        
        if self.config.entry.use_limit_orders and signal.suggested_entry:
            order.order_type = OrderType.LIMIT
            # Offset limit price slightly toward market for better fill
            offset = signal.atr * self.config.entry.limit_order_offset_atr
            if signal.side == Side.LONG:
                order.price = signal.suggested_entry + offset
            else:
                order.price = signal.suggested_entry - offset
        else:
            order.order_type = OrderType.MARKET
        
        return order
    
    async def create_stop_order(
        self,
        position: Position
    ) -> Order:
        """Create stop loss order for position."""
        close_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        
        order = Order(
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.STOP_MARKET,
            quantity=position.quantity,
            stop_price=position.stop_loss,
            position_id=position.id,
            created_at=datetime.utcnow()
        )
        
        return order
    
    async def create_take_profit_order(
        self,
        position: Position
    ) -> Order:
        """Create take profit order for position."""
        close_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        
        order = Order(
            symbol=position.symbol,
            side=close_side,
            order_type=OrderType.LIMIT,
            quantity=position.quantity,
            price=position.take_profit,
            position_id=position.id,
            created_at=datetime.utcnow()
        )
        
        return order
    
    async def submit_order(self, order: Order, dynamic_multiplier: float = None) -> ExecutionResult:
        """Submit order through executor.

        Args:
            order: The order to submit
            dynamic_multiplier: Optional confidence multiplier for setting leverage (entries only)
        """
        self.total_orders += 1
        self.pending_orders[order.id] = order

        result = await self.executor.submit_order(order, dynamic_multiplier=dynamic_multiplier)
        
        if result.success:
            self.total_latency += result.latency_ms
            
            if order.status == OrderStatus.FILLED:
                self.successful_fills += 1
                self.total_slippage += order.slippage or 0
                self.filled_orders[order.id] = order
                del self.pending_orders[order.id]
        else:
            self.failed_orders += 1
            self.cancelled_orders[order.id] = order
            if order.id in self.pending_orders:
                del self.pending_orders[order.id]
        
        return result
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self.pending_orders:
            return False
        
        order = self.pending_orders[order_id]
        success = await self.executor.cancel_order(order.exchange_order_id, order.symbol)
        
        if success:
            order.status = OrderStatus.CANCELLED
            self.cancelled_orders[order_id] = order
            del self.pending_orders[order_id]
        
        return success
    
    async def cancel_all_orders(self, symbol: Optional[str] = None):
        """Cancel all open orders, optionally for specific symbol."""
        for order_id, order in list(self.pending_orders.items()):
            if symbol is None or order.symbol == symbol:
                await self.cancel_order(order_id)
    
    async def update_order_statuses(self):
        """Update status of all pending orders."""
        for order_id, order in list(self.pending_orders.items()):
            updated = await self.executor.get_order_status(
                order.exchange_order_id,
                order.symbol
            )
            
            if updated:
                order.status = updated.status
                order.filled_quantity = updated.filled_quantity
                order.average_fill_price = updated.average_fill_price
                order.updated_at = datetime.utcnow()
                
                if order.status == OrderStatus.FILLED:
                    self.successful_fills += 1
                    self.filled_orders[order_id] = order
                    del self.pending_orders[order_id]
                elif order.status in [OrderStatus.CANCELLED, OrderStatus.EXPIRED, OrderStatus.REJECTED]:
                    self.cancelled_orders[order_id] = order
                    del self.pending_orders[order_id]
    
    def get_execution_stats(self) -> dict:
        """Get execution statistics."""
        return {
            "total_orders": self.total_orders,
            "successful_fills": self.successful_fills,
            "failed_orders": self.failed_orders,
            "fill_rate": self.successful_fills / self.total_orders if self.total_orders > 0 else 0,
            "avg_slippage": self.total_slippage / self.successful_fills if self.successful_fills > 0 else 0,
            "avg_latency_ms": self.total_latency / self.total_orders if self.total_orders > 0 else 0,
            "pending_orders": len(self.pending_orders)
        }
