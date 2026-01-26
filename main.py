"""
Main Strategy Orchestrator
Coordinates all components for live trading and backtesting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from config.settings import StrategyConfig, DEFAULT_CONFIG
from core.types import (
    Candle, MarketData, Signal, Position, Order,
    Side, PositionStatus, SignalType, OrderType
)
from data.ingestion import (
    DataProvider, DataAggregator, HyperliquidProvider, 
    BybitProvider, SimulatedDataProvider, create_data_provider
)
from signals.detection import SignalGenerator
from risk.management import RiskManager
from execution.orders import (
    ExecutionHandler, OrderManager, HyperliquidExecutor,
    SimulatedExecutor, ExecutionResult
)
from utils.logger import StructuredLogger, TradeJournal, PerformanceMonitor


logger = logging.getLogger(__name__)


class WickReversalStrategy:
    """
    Main strategy class orchestrating all components.
    """
    
    def __init__(
        self,
        config: StrategyConfig = DEFAULT_CONFIG,
        data_provider: Optional[DataProvider] = None,
        executor: Optional[ExecutionHandler] = None
    ):
        self.config = config
        
        # Initialize components
        self.data_aggregator = DataAggregator(config)
        self.signal_generator = SignalGenerator(config)
        self.risk_manager = RiskManager(config)
        
        # Data provider
        self.data_provider = data_provider
        
        # Execution
        self.executor = executor
        self.order_manager: Optional[OrderManager] = None
        
        # Logging and monitoring
        self.logger = StructuredLogger(
            name="wick_reversal",
            log_level=config.log_level
        )
        self.trade_journal = TradeJournal()
        self.performance_monitor = PerformanceMonitor()
        
        # State
        self.is_running = False
        self.candle_count = 0
        self.btc_price_history: List[float] = []
        
        # Position tracking
        self.active_positions: Dict[str, Position] = {}
        self.pending_entries: Dict[str, Signal] = {}
    
    async def initialize(self, initial_capital: float = 10000.0):
        """Initialize the strategy for trading."""
        self.logger.log_system_event("strategy_init", {
            "config": self.config.strategy_name,
            "initial_capital": initial_capital,
            "symbols": [s.symbol for s in self.config.symbols]
        })
        
        self.risk_manager.initialize(initial_capital)
        
        if self.data_provider:
            await self.data_provider.connect()
        
        if self.executor:
            self.order_manager = OrderManager(self.config, self.executor)
        
        self.is_running = True
        self.logger.log_system_event("strategy_ready")
    
    async def shutdown(self):
        """Gracefully shutdown the strategy."""
        self.logger.log_system_event("strategy_shutdown_initiated")
        self.is_running = False
        
        if self.order_manager:
            await self.order_manager.cancel_all_orders()
        
        if self.data_provider:
            await self.data_provider.disconnect()
        
        self.logger.log_system_event("strategy_shutdown_complete")
    
    async def process_candle(
        self,
        symbol: str,
        candle: Candle,
        orderbook: Optional[Dict] = None
    ):
        """Process a new candle for a symbol."""
        start_time = datetime.utcnow()
        
        try:
            # Get BTC price for correlation filter
            btc_price = self.btc_price_history[-1] if self.btc_price_history else None
            
            # Update BTC price if this is BTC
            if "BTC" in symbol.upper():
                self.btc_price_history.append(candle.close)
                if len(self.btc_price_history) > 100:
                    self.btc_price_history = self.btc_price_history[-100:]
                return
            
            # Build market data
            market_data = self.data_aggregator.get_market_data(symbol, candle, orderbook)
            self.candle_count += 1
            
            # Update existing positions
            await self._update_positions(symbol, candle)
            
            # Check exits
            await self._check_position_exits(symbol, candle)
            
            # Generate signal
            signal = self.signal_generator.generate_signal(
                symbol, market_data, self.candle_count, btc_price
            )
            
            if signal.is_valid:
                self.logger.log_signal(signal)
                self.performance_monitor.increment_counter("signals_generated")
                await self._process_signal(signal, candle)
            elif signal.filter_result.value != "passed":
                self.logger.log_filter_rejection(
                    symbol, signal.filter_result.value, signal.filter_details
                )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.performance_monitor.record_latency("signal_processing_time", processing_time)
        
        except Exception as e:
            self.logger.log_error("candle_processing", str(e), {"symbol": symbol})
            self.performance_monitor.increment_counter("errors")
    
    async def _update_positions(self, symbol: str, candle: Candle):
        """Update unrealized P&L and trailing stops."""
        if symbol not in self.active_positions:
            return
        
        position = self.active_positions[symbol]
        if not position.is_open:
            return
        
        position.update_unrealized_pnl(candle.close)
        position.current_candle_count = self.candle_count
        
        atr = self.data_aggregator.calculate_atr(symbol)
        if atr > 0:
            position.update_trailing_stop(candle.close, atr, self.config.exit)
    
    async def _check_position_exits(self, symbol: str, candle: Candle):
        """Check and execute position exits."""
        if symbol not in self.active_positions:
            return

        position = self.active_positions[symbol]
        if not position.is_open:
            return

        exit_triggered = False
        exit_reason = ""
        exit_price = candle.close

        if self.risk_manager.check_stop_loss(position, candle.close):
            exit_triggered = True
            effective_stop = (
                position.trailing_stop_price
                if position.trailing_stop_active
                else position.stop_loss
            )
            exit_reason = "trailing_stop" if position.trailing_stop_active else "stop_loss"
            exit_price = effective_stop

        elif self.risk_manager.check_take_profit(position, candle.close):
            exit_triggered = True
            exit_reason = "take_profit"
            exit_price = position.take_profit

        elif self.risk_manager.check_time_exit(position):
            exit_triggered = True
            exit_reason = "time_exit"

        if exit_triggered:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _process_signal(self, signal: Signal, candle: Candle):
        """Process a valid trading signal."""
        if signal.symbol in self.active_positions:
            if self.active_positions[signal.symbol].is_open:
                return
        
        symbol_config = None
        for sc in self.config.symbols:
            if sc.symbol == signal.symbol:
                symbol_config = {
                    'risk_multiplier': sc.risk_multiplier,
                    'min_position_size': sc.min_position_size,
                    'max_position_usd': sc.max_position_usd
                }
                break
        
        assessment = self.risk_manager.assess_trade(signal, candle.close, symbol_config)
        
        if not assessment.approved:
            self.logger.log_risk_event("trade_rejected", {
                "symbol": signal.symbol,
                "reason": assessment.reason
            })
            return
        
        await self._enter_position(signal, assessment.position_size, candle)
    
    async def _enter_position(
        self,
        signal: Signal,
        position_size: float,
        candle: Candle
    ):
        """Enter a new position."""
        entry_price = signal.suggested_entry or candle.close
        
        # Create position
        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            status=PositionStatus.OPEN,
            quantity=position_size,
            entry_price=entry_price,
            stop_loss=signal.suggested_stop,
            take_profit=signal.suggested_target,
            opened_at=candle.timestamp,
            entry_candle_count=self.candle_count,
            current_candle_count=self.candle_count,
            signal_id=signal.id
        )
        
        # Calculate risk/reward
        if signal.side == Side.LONG:
            risk = entry_price - position.stop_loss
            reward = position.take_profit - entry_price
        else:
            risk = position.stop_loss - entry_price
            reward = entry_price - position.take_profit
        
        position.risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Execute entry order if we have an order manager
        if self.order_manager:
            order = await self.order_manager.create_entry_order(signal, position_size)
            result = await self.order_manager.submit_order(order)
            
            self.logger.log_order(order, "submitted")
            
            if result.success and order.is_filled:
                position.entry_price = order.average_fill_price
                position.total_commission = order.commission
                self.logger.log_fill(order, result.latency_ms)
                self.performance_monitor.record_latency("execution_latency", result.latency_ms)
            else:
                self.logger.log_error("entry_failed", result.error)
                return
        
        # Store position
        self.active_positions[signal.symbol] = position
        self.risk_manager.register_position(position)
        
        self.logger.log_position_open(position)
        self.performance_monitor.increment_counter("trades_executed")
        
        # Record cooldown
        self.signal_generator.record_signal(signal.symbol, self.candle_count)
    
    async def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str
    ):
        """Close an existing position."""
        # Execute exit order if we have order manager
        if self.order_manager:
            close_side = Side.SHORT if position.side == Side.LONG else Side.LONG
            order = Order(
                symbol=position.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                position_id=position.id
            )
            result = await self.order_manager.submit_order(order)
            
            if result.success and order.is_filled:
                exit_price = order.average_fill_price
                position.total_commission += order.commission
                self.logger.log_fill(order, result.latency_ms)
        
        # Update position
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.utcnow()
        
        # Calculate P&L
        if position.side == Side.LONG:
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity
        
        position.realized_pnl = gross_pnl
        net_pnl = gross_pnl - position.total_commission
        
        # Update risk manager
        self.risk_manager.close_position(position, exit_price, datetime.utcnow())
        
        self.logger.log_position_close(position, reason)
        
        # Record to journal
        self.trade_journal.record_trade(
            position,
            market_context={"exit_reason": reason, "net_pnl": net_pnl}
        )
    
    async def run_live(self, symbols: List[str]):
        """Run strategy in live trading mode."""
        if not self.data_provider:
            raise ValueError("No data provider configured for live trading")
        
        self.logger.log_system_event("live_trading_started", {"symbols": symbols})
        
        try:
            # Subscribe to candle streams for each symbol
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self._stream_symbol(symbol)
                )
                tasks.append(task)
            
            # Also stream BTC for correlation filter
            btc_task = asyncio.create_task(
                self._stream_btc()
            )
            tasks.append(btc_task)
            
            await asyncio.gather(*tasks)
        
        except asyncio.CancelledError:
            self.logger.log_system_event("live_trading_cancelled")
        finally:
            await self.shutdown()
    
    async def _stream_symbol(self, symbol: str):
        """Stream candles for a symbol."""
        async for candle in self.data_provider.subscribe_candles(
            symbol, self.config.timeframe.value
        ):
            if not self.is_running:
                break
            
            # Get orderbook snapshot
            orderbook = await self.data_provider.get_orderbook_snapshot(symbol)
            
            await self.process_candle(symbol, candle, orderbook)
    
    async def _stream_btc(self):
        """Stream BTC price for correlation filter."""
        async for candle in self.data_provider.subscribe_candles(
            "BTC-PERP", self.config.timeframe.value
        ):
            if not self.is_running:
                break
            self.btc_price_history.append(candle.close)
            if len(self.btc_price_history) > 100:
                self.btc_price_history = self.btc_price_history[-100:]
    
    def get_status(self) -> dict:
        """Get current strategy status."""
        return {
            "is_running": self.is_running,
            "candle_count": self.candle_count,
            "active_positions": len([p for p in self.active_positions.values() if p.is_open]),
            "portfolio": self.risk_manager.get_portfolio_summary(
                {s: p.entry_price for s, p in self.active_positions.items()}
            ),
            "performance": self.performance_monitor.get_summary()
        }
