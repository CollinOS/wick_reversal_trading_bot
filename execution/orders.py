"""
Order Execution Module
Handles order creation, submission, and management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
import uuid

from core.types import (
    Order, OrderType, OrderStatus, Side, Position, PositionStatus, Signal
)
from config.settings import StrategyConfig, ExecutionConfig


logger = logging.getLogger(__name__)


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
    
    Uses Hyperliquid's REST API for order management.
    Requires wallet private key for signing transactions.
    """
    
    def __init__(
        self,
        private_key: str,
        testnet: bool = True,
        vault_address: Optional[str] = None
    ):
        self.private_key = private_key
        self.testnet = testnet
        self.vault_address = vault_address
        self.base_url = (
            "https://api.hyperliquid-testnet.xyz" if testnet
            else "https://api.hyperliquid.xyz"
        )
        self._session = None
    
    async def connect(self):
        """Initialize session and verify connection."""
        import aiohttp
        self._session = aiohttp.ClientSession()
        logger.info(f"Hyperliquid executor connected ({'testnet' if self.testnet else 'mainnet'})")
    
    async def disconnect(self):
        if self._session:
            await self._session.close()
    
    def _sign_order(self, order_data: dict) -> dict:
        """Sign order with private key (simplified - actual impl needs eth-account)."""
        # In production, use eth-account library to sign
        # This is a placeholder showing the structure
        return {
            "action": order_data,
            "nonce": int(datetime.utcnow().timestamp() * 1000),
            "signature": "0x..."  # Would be actual signature
        }
    
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order to Hyperliquid."""
        start_time = datetime.utcnow()
        
        try:
            # Prepare order payload
            is_buy = order.side == Side.LONG
            
            order_data = {
                "type": "order",
                "orders": [{
                    "a": 0,  # Asset index (would need mapping)
                    "b": is_buy,
                    "p": str(order.price) if order.price else "0",
                    "s": str(order.quantity),
                    "r": order.order_type == OrderType.LIMIT,
                    "t": {
                        "limit": {"tif": "Gtc"} if order.order_type == OrderType.LIMIT
                        else {"trigger": {"triggerPx": str(order.stop_price), "isMarket": True}}
                    }
                }],
                "grouping": "na"
            }
            
            signed = self._sign_order(order_data)
            
            async with self._session.post(
                f"{self.base_url}/exchange",
                json=signed
            ) as response:
                data = await response.json()
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            if data.get("status") == "ok":
                order.status = OrderStatus.OPEN
                order.exchange_order_id = data.get("response", {}).get("data", {}).get("statuses", [{}])[0].get("oid")
                order.updated_at = datetime.utcnow()
                
                logger.info(f"Order submitted: {order.id} -> {order.exchange_order_id}")
                return ExecutionResult(success=True, order=order, latency_ms=latency)
            else:
                error = data.get("response", {}).get("error", "Unknown error")
                logger.error(f"Order failed: {error}")
                order.status = OrderStatus.REJECTED
                return ExecutionResult(success=False, order=order, error=error, latency_ms=latency)
        
        except Exception as e:
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.exception(f"Order submission error: {e}")
            order.status = OrderStatus.REJECTED
            return ExecutionResult(success=False, order=order, error=str(e), latency_ms=latency)
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an open order."""
        try:
            cancel_data = {
                "type": "cancel",
                "cancels": [{"a": 0, "o": int(order_id)}]  # Simplified
            }
            
            signed = self._sign_order(cancel_data)
            
            async with self._session.post(
                f"{self.base_url}/exchange",
                json=signed
            ) as response:
                data = await response.json()
            
            return data.get("status") == "ok"
        except Exception as e:
            logger.exception(f"Cancel order error: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Optional[Order]:
        """Get order status from exchange."""
        try:
            payload = {
                "type": "orderStatus",
                "user": self.vault_address or "0x...",  # Would need actual address
                "oid": int(order_id)
            }
            
            async with self._session.post(
                f"{self.base_url}/info",
                json=payload
            ) as response:
                data = await response.json()
            
            if data:
                order = Order(
                    id=order_id,
                    symbol=symbol,
                    exchange_order_id=order_id
                )
                # Map status
                status_map = {
                    "open": OrderStatus.OPEN,
                    "filled": OrderStatus.FILLED,
                    "canceled": OrderStatus.CANCELLED,
                    "triggered": OrderStatus.OPEN
                }
                order.status = status_map.get(data.get("status"), OrderStatus.PENDING)
                order.filled_quantity = float(data.get("filled", 0))
                order.average_fill_price = float(data.get("avgPrice", 0)) or None
                return order
            return None
        except Exception as e:
            logger.exception(f"Get order status error: {e}")
            return None
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position."""
        try:
            payload = {
                "type": "clearinghouseState",
                "user": self.vault_address or "0x..."
            }
            
            async with self._session.post(
                f"{self.base_url}/info",
                json=payload
            ) as response:
                data = await response.json()
            
            coin = symbol.replace("-PERP", "")
            for pos in data.get("assetPositions", []):
                if pos.get("coin") == coin:
                    return {
                        "symbol": symbol,
                        "size": float(pos.get("szi", 0)),
                        "entry_price": float(pos.get("entryPx", 0)),
                        "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                        "liquidation_price": float(pos.get("liquidationPx", 0))
                    }
            return None
        except Exception as e:
            logger.exception(f"Get position error: {e}")
            return None
    
    async def close_position(self, symbol: str, side: Side) -> ExecutionResult:
        """Close position with market order."""
        position = await self.get_position(symbol)
        if not position or position["size"] == 0:
            return ExecutionResult(success=False, error="No position to close")
        
        close_side = Side.SHORT if side == Side.LONG else Side.LONG
        order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=abs(position["size"])
        )
        
        return await self.submit_order(order)


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
    
    async def submit_order(self, order: Order) -> ExecutionResult:
        """Submit order through executor."""
        self.total_orders += 1
        self.pending_orders[order.id] = order
        
        result = await self.executor.submit_order(order)
        
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
