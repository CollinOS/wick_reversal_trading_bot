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

        # Positions to ignore (pre-existing manual trades)
        self.ignored_positions: set = set()

        # Trade callbacks for external logging
        self.on_trade_entry: Optional[callable] = None
        self.on_trade_exit: Optional[callable] = None
    
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

            # Calculate wick percentages for signal evaluation
            upper_wick_pct = (candle.upper_wick / candle.close * 100) if candle.close > 0 else 0
            lower_wick_pct = (candle.lower_wick / candle.close * 100) if candle.close > 0 else 0
            # Verbose candle logging disabled - uncomment to debug:
            # logger.info(f"📈 {symbol} candle #{self.candle_count}: "
            #            f"O={candle.open:.4f} H={candle.high:.4f} L={candle.low:.4f} C={candle.close:.4f} | "
            #            f"Upper wick: {upper_wick_pct:.2f}% Lower wick: {lower_wick_pct:.2f}%")

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
                logger.info(f"🚫 {symbol}: Filter rejected - {signal.filter_result.value}: {signal.filter_details}")
            elif signal.signal_type.value == "no_signal":
                # Log why no signal was generated
                min_wick_pct = self.config.signal.min_wick_pct * 100
                if upper_wick_pct >= min_wick_pct or lower_wick_pct >= min_wick_pct:
                    # Wick was big enough but didn't pass other criteria
                    logger.info(f"⚠️ {symbol}: Large wick detected but signal criteria not met")
                    if signal.criteria_met:
                        logger.info(f"   Criteria status: {signal.criteria_met}")
            
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
        """Check and execute position exits.

        Logic matches backtest/engine.py exactly for consistent results.
        Uses candle high/low to check if price touched stop/target levels.
        """
        if symbol not in self.active_positions:
            return

        position = self.active_positions[symbol]
        if not position.is_open:
            return

        exit_price = None
        exit_reason = None

        # Check stop loss (use low/high depending on side) - matches backtest
        if position.side == Side.LONG:
            # Stop hit if low <= stop price
            effective_stop = position.trailing_stop_price if position.trailing_stop_active else position.stop_loss
            if candle.low <= effective_stop:
                exit_price = effective_stop
                exit_reason = "trailing_stop" if position.trailing_stop_active else "stop_loss"
        else:
            # Stop hit if high >= stop price
            effective_stop = position.trailing_stop_price if position.trailing_stop_active else position.stop_loss
            if candle.high >= effective_stop:
                exit_price = effective_stop
                exit_reason = "trailing_stop" if position.trailing_stop_active else "stop_loss"

        # Check take profit - matches backtest
        if exit_price is None:
            if position.side == Side.LONG:
                if candle.high >= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"
            else:
                if candle.low <= position.take_profit:
                    exit_price = position.take_profit
                    exit_reason = "take_profit"

        # Check time-based exit - matches backtest
        if exit_price is None and position.candles_held >= self.config.exit.max_hold_candles:
            exit_price = candle.close
            exit_reason = "time_exit"

        # Execute exit if triggered
        if exit_price is not None:
            await self._close_position(position, exit_price, exit_reason)
    
    async def _process_signal(self, signal: Signal, candle: Candle):
        """Process a valid trading signal."""
        # Skip if symbol has a pre-existing position we're ignoring
        if signal.symbol in self.ignored_positions:
            logger.debug(f"Skipping signal for {signal.symbol}: pre-existing position ignored")
            return

        if signal.symbol in self.active_positions:
            if self.active_positions[signal.symbol].is_open:
                return

        symbol_config = None
        for sc in self.config.symbols:
            if sc.symbol == signal.symbol:
                symbol_config = {
                    'risk_multiplier': sc.risk_multiplier,
                    'min_position_size': sc.min_position_size,
                    'base_position_usd': sc.base_position_usd,
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

        await self._enter_position(signal, assessment, candle)
    
    async def _enter_position(
        self,
        signal: Signal,
        assessment,  # RiskAssessment object
        candle: Candle
    ):
        """Enter a new position."""
        position_size = assessment.position_size
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
            result = await self.order_manager.submit_order(
                order, dynamic_multiplier=assessment.dynamic_multiplier
            )

            self.logger.log_order(order, "submitted")

            if result.success and order.is_filled:
                position.entry_price = order.average_fill_price
                position.total_commission = order.commission
                self.logger.log_fill(order, result.latency_ms)
                self.performance_monitor.record_latency("execution_latency", result.latency_ms)

                # Validate take profit is actually profitable after fill
                # If fill price is worse than expected, recalculate TP to ensure minimum profit
                min_profit_pct = 0.005  # 0.5% minimum
                if signal.side == Side.LONG:
                    min_take_profit = position.entry_price * (1 + min_profit_pct)
                    if position.take_profit < min_take_profit:
                        logger.warning(
                            f"Take profit ${position.take_profit:.4f} below entry ${position.entry_price:.4f}, "
                            f"adjusting to ${min_take_profit:.4f}"
                        )
                        position.take_profit = min_take_profit
                else:  # SHORT
                    max_take_profit = position.entry_price * (1 - min_profit_pct)
                    if position.take_profit > max_take_profit:
                        logger.warning(
                            f"Take profit ${position.take_profit:.4f} above entry ${position.entry_price:.4f}, "
                            f"adjusting to ${max_take_profit:.4f}"
                        )
                        position.take_profit = max_take_profit

                # Recalculate risk/reward with actual fill price
                if signal.side == Side.LONG:
                    risk = position.entry_price - position.stop_loss
                    reward = position.take_profit - position.entry_price
                else:
                    risk = position.stop_loss - position.entry_price
                    reward = position.entry_price - position.take_profit
                position.risk_reward_ratio = reward / risk if risk > 0 else 0

            else:
                self.logger.log_error("entry_failed", result.error)
                return

        # Store position
        self.active_positions[signal.symbol] = position
        self.risk_manager.register_position(position)

        self.logger.log_position_open(position)
        self.performance_monitor.increment_counter("trades_executed")

        # Call trade entry callback if set
        if self.on_trade_entry:
            try:
                # Get leverage breakdown if available
                leverage_breakdown = getattr(assessment, 'leverage_breakdown', {})

                self.on_trade_entry(
                    symbol=signal.symbol,
                    side=signal.side.value,
                    quantity=position_size,
                    price=position.entry_price,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                    leverage=assessment.leverage_used,
                    risk_amount=assessment.risk_amount,
                    criteria_met=signal.criteria_met,
                    signal_strength=signal.strength,
                    leverage_breakdown=leverage_breakdown,
                    base_position_size=assessment.base_position_size,
                    dynamic_multiplier=assessment.dynamic_multiplier
                )
            except Exception as e:
                logger.warning(f"Trade entry callback error: {e}")

        # Place exchange-based stop loss and take profit orders for real-time protection
        await self._place_exchange_exit_orders(position)

        # Record cooldown
        self.signal_generator.record_signal(signal.symbol, self.candle_count)

    async def _place_exchange_exit_orders(self, position: Position):
        """
        Place stop loss and take profit orders on the exchange for real-time protection.

        This ensures exits execute even if the bot is offline or slow to react.
        The exchange monitors price continuously and triggers orders instantly.

        If partial_tp_enabled, places two TP orders:
        - Partial TP at closer target (50% of position)
        - Full TP at original target (remaining 50%)
        """
        if not self.executor:
            print(f"  WARNING: No executor - SL/TP orders NOT placed for {position.symbol}")
            return

        # Check if executor supports trigger orders (HyperliquidExecutor does)
        if not hasattr(self.executor, 'submit_stop_loss_order'):
            print(f"  WARNING: Executor doesn't support exchange-based stops for {position.symbol}")
            return

        close_side = Side.SHORT if position.side == Side.LONG else Side.LONG

        # Store original quantity for partial TP tracking
        position.original_quantity = position.quantity

        # Place stop loss order on exchange (full position size)
        try:
            sl_result = await self.executor.submit_stop_loss_order(
                symbol=position.symbol,
                side=close_side,
                quantity=position.quantity,
                trigger_price=position.stop_loss
            )
            if sl_result.success and sl_result.order:
                position.stop_order_id = sl_result.order.exchange_order_id
                print(f"  Stop Loss: ${position.stop_loss:.4f} (order #{sl_result.order.exchange_order_id})")
            else:
                print(f"  WARNING: Stop loss order FAILED: {sl_result.error}")
                logger.error(f"Failed to place exchange stop loss for {position.symbol}: {sl_result.error}")
        except Exception as e:
            print(f"  WARNING: Stop loss order ERROR: {e}")
            logger.exception(f"Error placing exchange stop loss: {e}")

        # Check if partial take profit is enabled
        partial_tp_enabled = self.config.exit.partial_tp_enabled if hasattr(self.config.exit, 'partial_tp_enabled') else False

        if partial_tp_enabled:
            # Calculate partial TP price (closer target)
            partial_tp_mult = self.config.exit.partial_tp_atr_multiplier
            partial_tp_pct = self.config.exit.partial_tp_percent
            atr = self.data_aggregator.calculate_atr(position.symbol)

            if atr > 0:
                if position.side == Side.LONG:
                    partial_tp_price = position.entry_price + (atr * partial_tp_mult)
                else:
                    partial_tp_price = position.entry_price - (atr * partial_tp_mult)

                partial_quantity = position.quantity * partial_tp_pct
                remaining_quantity = position.quantity - partial_quantity

                position.partial_tp_price = partial_tp_price
                position.partial_tp_quantity = partial_quantity

                # Place partial take profit order
                try:
                    partial_result = await self.executor.submit_take_profit_order(
                        symbol=position.symbol,
                        side=close_side,
                        quantity=partial_quantity,
                        trigger_price=partial_tp_price
                    )
                    if partial_result.success and partial_result.order:
                        position.partial_tp_order_id = partial_result.order.exchange_order_id
                        print(f"  Partial TP (50%): ${partial_tp_price:.4f} (order #{partial_result.order.exchange_order_id})")
                    else:
                        print(f"  WARNING: Partial TP order FAILED: {partial_result.error}")
                except Exception as e:
                    print(f"  WARNING: Partial TP order ERROR: {e}")
                    logger.exception(f"Error placing partial take profit: {e}")

                # Place full take profit order for remaining quantity
                try:
                    tp_result = await self.executor.submit_take_profit_order(
                        symbol=position.symbol,
                        side=close_side,
                        quantity=remaining_quantity,
                        trigger_price=position.take_profit
                    )
                    if tp_result.success and tp_result.order:
                        position.take_profit_order_id = tp_result.order.exchange_order_id
                        print(f"  Full TP (50%): ${position.take_profit:.4f} (order #{tp_result.order.exchange_order_id})")
                    else:
                        print(f"  WARNING: Full TP order FAILED: {tp_result.error}")
                except Exception as e:
                    print(f"  WARNING: Full TP order ERROR: {e}")
                    logger.exception(f"Error placing full take profit: {e}")
            else:
                # No ATR available, fall back to single TP
                partial_tp_enabled = False

        if not partial_tp_enabled:
            # Place single take profit order for full position
            try:
                tp_result = await self.executor.submit_take_profit_order(
                    symbol=position.symbol,
                    side=close_side,
                    quantity=position.quantity,
                    trigger_price=position.take_profit
                )
                if tp_result.success and tp_result.order:
                    position.take_profit_order_id = tp_result.order.exchange_order_id
                    print(f"  Take Profit: ${position.take_profit:.4f} (order #{tp_result.order.exchange_order_id})")
                else:
                    print(f"  WARNING: Take profit order FAILED: {tp_result.error}")
                    logger.error(f"Failed to place exchange take profit for {position.symbol}: {tp_result.error}")
            except Exception as e:
                print(f"  WARNING: Take profit order ERROR: {e}")
                logger.exception(f"Error placing exchange take profit: {e}")

    async def _cancel_exchange_exit_orders(self, position: Position):
        """Cancel any exchange-based stop loss and take profit orders for a position."""
        if not self.executor or not hasattr(self.executor, 'cancel_order'):
            return

        # Cancel stop loss order if it exists
        if position.stop_order_id:
            try:
                await self.executor.cancel_order(position.stop_order_id, position.symbol)
                logger.debug(f"Cancelled exchange stop loss order: {position.stop_order_id}")
            except Exception as e:
                logger.warning(f"Error cancelling stop loss order: {e}")

        # Cancel partial take profit order if it exists
        if hasattr(position, 'partial_tp_order_id') and position.partial_tp_order_id:
            try:
                await self.executor.cancel_order(position.partial_tp_order_id, position.symbol)
                logger.debug(f"Cancelled exchange partial TP order: {position.partial_tp_order_id}")
            except Exception as e:
                logger.warning(f"Error cancelling partial TP order: {e}")

        # Cancel take profit order if it exists
        if position.take_profit_order_id:
            try:
                await self.executor.cancel_order(position.take_profit_order_id, position.symbol)
                logger.debug(f"Cancelled exchange take profit order: {position.take_profit_order_id}")
            except Exception as e:
                logger.warning(f"Error cancelling take profit order: {e}")

    async def _close_position(
        self,
        position: Position,
        exit_price: float,
        reason: str
    ):
        """Close an existing position."""
        actual_exit_price = exit_price
        total_fees = 0.0
        actual_pnl = None
        actual_reason = reason  # Will be updated based on exchange data

        # Execute exit order if we have an executor
        if self.executor:
            # First check if position actually exists on exchange
            # (it may have been closed by exchange stop loss/take profit already)
            exchange_position = await self.executor.get_position(position.symbol)
            position_already_closed = not exchange_position or abs(exchange_position.get("size", 0)) < 1e-10

            if not position_already_closed:
                # Position still exists - cancel TP/SL orders and close manually
                await self._cancel_exchange_exit_orders(position)
                result = await self.executor.close_position(position.symbol, position.side)

                if result.success and result.order and result.order.is_filled:
                    actual_exit_price = result.order.average_fill_price
                    position.total_commission += result.order.commission or 0
                    self.logger.log_fill(result.order, result.latency_ms)
            else:
                # Position was already closed by exchange (TP/SL triggered)
                # Cancel remaining orders to clean up
                await self._cancel_exchange_exit_orders(position)

            # ALWAYS fetch actual fill data to get real price, fees, and determine true exit reason
            if hasattr(self.executor, 'get_recent_fills'):
                try:
                    fills = await self.executor.get_recent_fills(position.symbol, limit=10)
                    if fills:
                        logger.debug(f"Found {len(fills)} recent fills for {position.symbol}")
                        # Get the most recent closing fill
                        for fill in fills:
                            # Check if this fill is closing our position (opposite side)
                            is_closing_fill = (
                                (position.side == Side.LONG and fill["side"] == "A") or  # Sell to close long
                                (position.side == Side.SHORT and fill["side"] == "B")    # Buy to close short
                            )
                            if is_closing_fill:
                                actual_exit_price = fill["price"]
                                total_fees = abs(fill["fee"])
                                if fill["closed_pnl"] != 0:
                                    actual_pnl = fill["closed_pnl"]

                                # Determine actual exit reason based on fill price
                                # Compare to TP/SL levels to figure out what really triggered
                                if position.side == Side.LONG:
                                    if actual_exit_price >= position.take_profit * 0.998:  # Within 0.2% of TP
                                        actual_reason = "take_profit"
                                    elif actual_exit_price <= position.stop_loss * 1.002:  # Within 0.2% of SL
                                        actual_reason = "stop_loss"
                                    else:
                                        actual_reason = reason  # Keep original (time_exit, etc.)
                                else:  # SHORT
                                    if actual_exit_price <= position.take_profit * 1.002:  # Within 0.2% of TP
                                        actual_reason = "take_profit"
                                    elif actual_exit_price >= position.stop_loss * 0.998:  # Within 0.2% of SL
                                        actual_reason = "stop_loss"
                                    else:
                                        actual_reason = reason

                                logger.info(f"Exit fill {position.symbol}: price=${actual_exit_price:.4f}, "
                                          f"pnl=${actual_pnl if actual_pnl else 'N/A'}, reason={actual_reason} "
                                          f"(TP=${position.take_profit:.4f}, SL=${position.stop_loss:.4f})")
                                break
                        else:
                            logger.warning(f"No closing fill found for {position.symbol} - using estimated data")
                    else:
                        logger.warning(f"No fills returned for {position.symbol}")
                except Exception as e:
                    logger.warning(f"Could not fetch actual fill data for {position.symbol}: {e}")

        # Update position
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.utcnow()

        # Use actual P&L from exchange if available, otherwise calculate
        if actual_pnl is not None:
            net_pnl = actual_pnl  # Exchange P&L already includes fees
            gross_pnl = net_pnl + total_fees
        else:
            # Calculate P&L (fallback if fill data unavailable)
            if position.side == Side.LONG:
                gross_pnl = (actual_exit_price - position.entry_price) * position.quantity
            else:
                gross_pnl = (position.entry_price - actual_exit_price) * position.quantity

            position.total_commission += total_fees
            net_pnl = gross_pnl - position.total_commission

        position.realized_pnl = gross_pnl

        # Update risk manager
        self.risk_manager.close_position(position, actual_exit_price, datetime.utcnow())

        self.logger.log_position_close(position, actual_reason)

        # Call trade exit callback if set
        if self.on_trade_exit:
            try:
                self.on_trade_exit(
                    symbol=position.symbol,
                    side=position.side.value,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    exit_price=actual_exit_price,
                    pnl=net_pnl,
                    reason=actual_reason
                )
            except Exception as e:
                logger.warning(f"Trade exit callback error: {e}")

        # Record to journal
        self.trade_journal.record_trade(
            position,
            market_context={"exit_reason": actual_reason, "net_pnl": net_pnl, "fees": total_fees}
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
