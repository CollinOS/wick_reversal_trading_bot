"""
Signal Detection Module
Identifies exaggerated wick patterns for mean-reversion entries.
"""

import logging
from typing import Optional, List, Tuple
from datetime import datetime

from core.types import (
    Candle, MarketData, Signal, SignalType, FilterResult, Side
)
from config.settings import StrategyConfig, SignalConfig, FilterConfig


logger = logging.getLogger(__name__)


class WickAnalyzer:
    """
    Analyzes candlestick wicks to detect exaggerated price dislocations.
    """

    # Criteria weights based on backtest win rates
    # Higher weight = criterion is more predictive of winning trades
    CRITERIA_WEIGHTS = {
        'rejection': 1.0,    # Base criterion (always required)
        'wtb_ratio': 0.95,   # 67.4% win rate
        'atr_ratio': 1.3,    # 81% win rate - most predictive
        'vwap_dist': 1.0,    # 68.7% win rate
        'volume': 0.8,       # High volume at wick = capitulation confirmation
    }

    # Volume ratio thresholds for strength bonus
    VOLUME_BONUS_MIN = 1.5   # Minimum volume ratio for any bonus
    VOLUME_BONUS_MAX = 3.0   # Volume ratio for maximum bonus

    def __init__(self, config: SignalConfig):
        self.config = config
    
    def check_wick_to_body_ratio(
        self,
        candle: Candle,
        wick_type: str
    ) -> Tuple[bool, float]:
        """
        Check if wick-to-body ratio exceeds threshold.
        
        Returns:
            (condition_met, ratio_value)
        """
        ratio = candle.wick_to_body_ratio(wick_type)
        threshold = self.config.wick_to_body_ratio
        
        return ratio >= threshold, ratio
    
    def check_wick_atr_ratio(
        self,
        candle: Candle,
        atr: float,
        wick_type: str
    ) -> Tuple[bool, float]:
        """
        Check if wick size exceeds ATR threshold.
        
        Returns:
            (condition_met, wick_atr_ratio)
        """
        if atr <= 0:
            return False, 0.0
        
        wick = candle.upper_wick if wick_type == "upper" else candle.lower_wick
        ratio = wick / atr
        threshold = self.config.wick_atr_multiplier
        
        return ratio >= threshold, ratio
    
    def check_vwap_distance(
        self,
        candle: Candle,
        vwap: float,
        atr: float,
        wick_type: str
    ) -> Tuple[bool, float]:
        """
        Check if wick extreme is sufficiently far from VWAP.
        
        Returns:
            (condition_met, distance_in_atr)
        """
        if atr <= 0 or vwap <= 0:
            return False, 0.0
        
        if wick_type == "upper":
            distance = candle.high - vwap
        else:
            distance = vwap - candle.low
        
        distance_atr = distance / atr
        threshold = self.config.vwap_distance_atr
        
        return distance_atr >= threshold, distance_atr
    
    def check_minimum_wick_size(
        self,
        candle: Candle,
        wick_type: str
    ) -> Tuple[bool, float]:
        """
        Check if wick meets minimum size requirement.
        
        Returns:
            (condition_met, wick_pct)
        """
        wick = candle.upper_wick if wick_type == "upper" else candle.lower_wick
        wick_pct = wick / candle.close if candle.close > 0 else 0
        threshold = self.config.min_wick_pct
        
        return wick_pct >= threshold, wick_pct
    
    def check_rejection_confirmation(
        self,
        candle: Candle,
        wick_type: str
    ) -> Tuple[bool, float]:
        """
        Check if close confirms rejection (price closed away from wick extreme).
        
        For upper wicks: close should be well below the high
        For lower wicks: close should be well above the low
        
        Returns:
            (condition_met, rejection_strength)
        """
        if candle.range == 0:
            return False, 0.0
        
        if wick_type == "upper":
            # How far close is from high (as % of wick)
            if candle.upper_wick == 0:
                return False, 0.0
            rejection = (candle.high - candle.close) / candle.upper_wick
        else:
            # How far close is from low (as % of wick)
            if candle.lower_wick == 0:
                return False, 0.0
            rejection = (candle.close - candle.low) / candle.lower_wick
        
        threshold = self.config.rejection_threshold_pct
        return rejection >= threshold, rejection
    
    def analyze_wick(
        self,
        candle: Candle,
        atr: float,
        vwap: float,
        volume_ratio: float = 1.0
    ) -> Tuple[SignalType, float, List[str]]:
        """
        Perform full wick analysis on a candle.

        Args:
            candle: The candle to analyze
            atr: Current ATR value
            vwap: Current VWAP value
            volume_ratio: Current volume / baseline volume (for strength boost)

        Returns:
            (signal_type, strength, criteria_met)
        """
        criteria_met = []

        # Analyze upper wick for short signal
        upper_results = self._analyze_single_wick(
            candle, atr, vwap, "upper", volume_ratio
        )

        # Analyze lower wick for long signal
        lower_results = self._analyze_single_wick(
            candle, atr, vwap, "lower", volume_ratio
        )

        # Determine which signal is stronger (if any)
        if upper_results[0] and lower_results[0]:
            # Both wicks qualify - take the stronger one
            if upper_results[1] >= lower_results[1]:
                return SignalType.UPPER_WICK_SHORT, upper_results[1], upper_results[2]
            else:
                return SignalType.LOWER_WICK_LONG, lower_results[1], lower_results[2]
        elif upper_results[0]:
            return SignalType.UPPER_WICK_SHORT, upper_results[1], upper_results[2]
        elif lower_results[0]:
            return SignalType.LOWER_WICK_LONG, lower_results[1], lower_results[2]

        return SignalType.NO_SIGNAL, 0.0, []
    
    def _analyze_single_wick(
        self,
        candle: Candle,
        atr: float,
        vwap: float,
        wick_type: str,
        volume_ratio: float = 1.0
    ) -> Tuple[bool, float, List[str]]:
        """
        Analyze a single wick (upper or lower).

        Args:
            candle: The candle to analyze
            atr: Current ATR value
            vwap: Current VWAP value
            wick_type: "upper" or "lower"
            volume_ratio: Current volume / baseline volume

        Returns:
            (signal_valid, strength, criteria_list)
        """
        criteria_met = []
        strength_components = []

        # Check minimum wick size first (required)
        min_size_met, min_size_val = self.check_minimum_wick_size(candle, wick_type)
        if not min_size_met:
            return False, 0.0, []

        # Check rejection confirmation (required)
        rejection_met, rejection_val = self.check_rejection_confirmation(candle, wick_type)
        if not rejection_met:
            return False, 0.0, []

        criteria_met.append(f"rejection:{rejection_val:.2f}")
        # Apply weight to rejection component
        strength_components.append(rejection_val * self.CRITERIA_WEIGHTS['rejection'])

        # Check optional conditions
        conditions_met = 0
        total_conditions = 3

        # 1. Wick-to-body ratio
        wtb_met, wtb_val = self.check_wick_to_body_ratio(candle, wick_type)
        if wtb_met:
            conditions_met += 1
            criteria_met.append(f"wtb_ratio:{wtb_val:.2f}")
            raw_component = min(wtb_val / self.config.wick_to_body_ratio, 2.0)
            strength_components.append(raw_component * self.CRITERIA_WEIGHTS['wtb_ratio'])

        # 2. Wick ATR ratio
        atr_met, atr_val = self.check_wick_atr_ratio(candle, atr, wick_type)
        if atr_met:
            conditions_met += 1
            criteria_met.append(f"atr_ratio:{atr_val:.2f}")
            raw_component = min(atr_val / self.config.wick_atr_multiplier, 2.0)
            strength_components.append(raw_component * self.CRITERIA_WEIGHTS['atr_ratio'])

        # 3. VWAP distance
        vwap_met, vwap_val = self.check_vwap_distance(candle, vwap, atr, wick_type)
        if vwap_met:
            conditions_met += 1
            criteria_met.append(f"vwap_dist:{vwap_val:.2f}")
            raw_component = min(vwap_val / self.config.vwap_distance_atr, 2.0)
            strength_components.append(raw_component * self.CRITERIA_WEIGHTS['vwap_dist'])

        # 4. Volume confirmation (adds to strength, not a required condition)
        # High volume at wick extremes indicates capitulation/reversal confirmation
        if volume_ratio >= self.VOLUME_BONUS_MIN:
            # Scale volume contribution: 1.5x -> 0.5, 3.0x -> 1.0
            volume_scale = min(
                (volume_ratio - self.VOLUME_BONUS_MIN) /
                (self.VOLUME_BONUS_MAX - self.VOLUME_BONUS_MIN),
                1.0
            )
            # Add volume as a strength component (0.5 to 1.0 range before weight)
            volume_component = (0.5 + volume_scale * 0.5) * self.CRITERIA_WEIGHTS['volume']
            strength_components.append(volume_component)
            criteria_met.append(f"vol_confirm:{volume_ratio:.1f}x")

        # Determine if signal is valid
        if self.config.require_all_conditions:
            signal_valid = conditions_met == total_conditions
        else:
            signal_valid = conditions_met >= 1  # At least one condition met

        # Calculate strength (0-1 scale)
        # Weighted average without artificial /2 compression
        # Components typically range 0.5-2.0 after weighting, normalize to 0-1
        if signal_valid and strength_components:
            raw_strength = sum(strength_components) / len(strength_components)
            strength = min(raw_strength / 1.5, 1.0)
        else:
            strength = 0.0

        return signal_valid, strength, criteria_met


class MarketFilter:
    """
    Filters out adverse market conditions where the strategy shouldn't trade.
    """
    
    def __init__(self, config: FilterConfig):
        self.config = config
        self.btc_prices: List[float] = []
    
    def update_btc_price(self, price: float):
        """Update BTC price history for correlation filter."""
        self.btc_prices.append(price)
        if len(self.btc_prices) > self.config.btc_lookback_candles + 1:
            self.btc_prices = self.btc_prices[-self.config.btc_lookback_candles - 1:]
    
    def check_volume_spike(self, market_data: MarketData) -> Tuple[bool, str]:
        """
        Check for abnormal volume spikes.

        Volume spikes are ALLOWED when the candle has a significant wick,
        as high volume at wick extremes indicates capitulation/reversal confirmation.
        """
        if market_data.volume_ratio <= self.config.volume_spike_multiplier:
            return True, ""

        # Check if candle has significant wick (indicates potential capitulation)
        # If so, allow the volume spike as it confirms the wick rejection
        candle = market_data.candle
        if candle.close > 0:
            upper_wick_pct = candle.upper_wick / candle.close
            lower_wick_pct = candle.lower_wick / candle.close
            # Allow volume spike if wick is >= 0.4% (below the 0.6% signal threshold
            # but significant enough to indicate potential capitulation)
            if upper_wick_pct >= 0.004 or lower_wick_pct >= 0.004:
                return True, ""  # Allow - volume spike at wick extreme is good

        return False, f"Volume spike: {market_data.volume_ratio:.1f}x baseline"
    
    def check_atr_expansion(self, market_data: MarketData) -> Tuple[bool, str]:
        """Check for abnormal volatility expansion."""
        if market_data.atr_ratio > self.config.atr_expansion_multiplier:
            return False, f"ATR expansion: {market_data.atr_ratio:.1f}x baseline"
        return True, ""
    
    def check_btc_move(self) -> Tuple[bool, str]:
        """Check for large BTC moves (market-wide risk-off)."""
        if len(self.btc_prices) < self.config.btc_lookback_candles + 1:
            return True, ""  # Not enough data
        
        start_price = self.btc_prices[-self.config.btc_lookback_candles - 1]
        current_price = self.btc_prices[-1]
        
        if start_price > 0:
            move_pct = abs(current_price - start_price) / start_price
            if move_pct > self.config.btc_move_threshold_pct:
                return False, f"BTC move: {move_pct*100:.1f}%"
        
        return True, ""
    
    def check_minimum_volume(self, market_data: MarketData) -> Tuple[bool, str]:
        """Check if volume meets minimum threshold."""
        # Estimate USD volume (volume * close price)
        usd_volume = market_data.candle.volume * market_data.candle.close
        
        if usd_volume < self.config.min_volume_usd:
            return False, f"Low volume: ${usd_volume:,.0f}"
        return True, ""
    
    def check_spread(self, market_data: MarketData) -> Tuple[bool, str]:
        """Check if spread is within acceptable range."""
        if market_data.spread is None:
            return True, ""  # No spread data available
        
        if market_data.spread > self.config.max_spread_pct:
            return False, f"High spread: {market_data.spread*100:.3f}%"
        return True, ""
    
    def check_orderbook_depth(self, market_data: MarketData) -> Tuple[bool, str]:
        """Check if order book has sufficient depth."""
        if market_data.bid_depth_usd is None or market_data.ask_depth_usd is None:
            return True, ""  # No orderbook data
        
        min_depth = min(market_data.bid_depth_usd, market_data.ask_depth_usd)
        
        if min_depth < self.config.min_orderbook_depth_usd:
            return False, f"Thin orderbook: ${min_depth:,.0f}"
        return True, ""
    
    def check_time_filter(self, timestamp: datetime) -> Tuple[bool, str]:
        """Check if current time is within allowed trading hours."""
        hour = timestamp.hour
        day = timestamp.weekday()
        
        if hour in self.config.avoid_hours:
            return False, f"Avoid hour: {hour}"
        
        if day in self.config.avoid_days:
            return False, f"Avoid day: {day}"
        
        return True, ""
    
    def run_all_filters(
        self,
        market_data: MarketData,
        btc_price: Optional[float] = None
    ) -> Tuple[FilterResult, str]:
        """
        Run all market filters.
        
        Returns:
            (FilterResult, detail_string)
        """
        # Update BTC price if provided
        if btc_price is not None:
            self.update_btc_price(btc_price)
        
        # Run each filter
        checks = [
            (self.check_volume_spike, FilterResult.VOLUME_SPIKE),
            (self.check_atr_expansion, FilterResult.ATR_EXPANSION),
            (self.check_minimum_volume, FilterResult.LOW_VOLUME),
            (self.check_spread, FilterResult.HIGH_SPREAD),
            (self.check_orderbook_depth, FilterResult.THIN_ORDERBOOK),
        ]
        
        for check_func, fail_result in checks:
            passed, detail = check_func(market_data)
            if not passed:
                return fail_result, detail
        
        # BTC move check
        passed, detail = self.check_btc_move()
        if not passed:
            return FilterResult.BTC_MOVE, detail
        
        # Time filter
        passed, detail = self.check_time_filter(market_data.timestamp)
        if not passed:
            return FilterResult.TIME_FILTER, detail
        
        return FilterResult.PASSED, ""


class SignalGenerator:
    """
    Main signal generation engine.
    Combines wick analysis with market filters to produce trading signals.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.wick_analyzer = WickAnalyzer(config.signal)
        self.market_filter = MarketFilter(config.filters)
        
        # Cooldown tracking per symbol
        self.cooldown_counters: dict = {}
        self.last_signal_candle: dict = {}
    
    def is_in_cooldown(self, symbol: str, current_candle_idx: int) -> bool:
        """Check if symbol is in cooldown period."""
        if symbol not in self.last_signal_candle:
            return False
        
        candles_since = current_candle_idx - self.last_signal_candle[symbol]
        return candles_since < self.config.risk.cooldown_candles
    
    def record_signal(self, symbol: str, candle_idx: int):
        """Record that a signal was generated (for cooldown tracking)."""
        self.last_signal_candle[symbol] = candle_idx
    
    def calculate_entry_levels(
        self,
        signal_type: SignalType,
        candle: Candle,
        atr: float,
        vwap: float
    ) -> Tuple[float, float, float]:
        """
        Calculate suggested entry, stop loss, and take profit levels.
        
        Returns:
            (entry_price, stop_loss, take_profit)
        """
        if signal_type == SignalType.UPPER_WICK_SHORT:
            # Short entry after upper wick
            if self.config.entry.mode.value == "candle_close":
                entry = candle.close
            else:
                # Retrace toward wick midpoint
                wick_mid = candle.high - (candle.upper_wick * self.config.entry.retrace_pct)
                entry = wick_mid
            
            # Stop loss above the high
            stop_buffer = atr * self.config.exit.stop_loss_buffer_atr
            stop_loss = candle.high + stop_buffer
            
            # Ensure stop loss doesn't exceed max
            max_stop = entry * (1 + self.config.exit.max_stop_loss_pct)
            stop_loss = min(stop_loss, max_stop)
            
            # Take profit based on configured target
            if self.config.exit.primary_target.value == "vwap":
                take_profit = vwap
            elif self.config.exit.primary_target.value == "candle_open":
                take_profit = candle.open
            elif self.config.exit.primary_target.value == "wick_midpoint":
                # For shorts: target the midpoint of the candle body (not wick)
                # Price rejected from high, expect reversion toward body midpoint
                take_profit = candle.body_midpoint
            else:  # ATR-based
                take_profit = entry - (atr * self.config.exit.atr_target_multiplier)
            
            # Ensure take profit is below entry for shorts
            take_profit = min(take_profit, entry * 0.995)
            
        else:  # LOWER_WICK_LONG
            # Long entry after lower wick
            if self.config.entry.mode.value == "candle_close":
                entry = candle.close
            else:
                # Retrace toward wick midpoint
                wick_mid = candle.low + (candle.lower_wick * self.config.entry.retrace_pct)
                entry = wick_mid
            
            # Stop loss below the low
            stop_buffer = atr * self.config.exit.stop_loss_buffer_atr
            stop_loss = candle.low - stop_buffer
            
            # Ensure stop loss doesn't exceed max
            max_stop = entry * (1 - self.config.exit.max_stop_loss_pct)
            stop_loss = max(stop_loss, max_stop)
            
            # Take profit based on configured target
            if self.config.exit.primary_target.value == "vwap":
                take_profit = vwap
            elif self.config.exit.primary_target.value == "candle_open":
                take_profit = candle.open
            elif self.config.exit.primary_target.value == "wick_midpoint":
                # For longs: target the midpoint of the candle body (not wick)
                # Price rejected from low, expect reversion toward body midpoint
                take_profit = candle.body_midpoint
            else:  # ATR-based
                take_profit = entry + (atr * self.config.exit.atr_target_multiplier)
            
            # Ensure take profit is above entry for longs
            take_profit = max(take_profit, entry * 1.005)
        
        return entry, stop_loss, take_profit
    
    def generate_signal(
        self,
        symbol: str,
        market_data: MarketData,
        candle_idx: int = 0,
        btc_price: Optional[float] = None
    ) -> Signal:
        """
        Generate a trading signal from market data.
        
        Args:
            symbol: Trading pair symbol
            market_data: Aggregated market data
            candle_idx: Current candle index (for cooldown tracking)
            btc_price: Current BTC price (for correlation filter)
        
        Returns:
            Signal object (may be NO_SIGNAL if no valid setup)
        """
        signal = Signal(
            timestamp=market_data.timestamp,
            symbol=symbol,
            atr=market_data.atr,
            vwap=market_data.vwap
        )
        
        # Check cooldown first
        if self.is_in_cooldown(symbol, candle_idx):
            signal.signal_type = SignalType.NO_SIGNAL
            signal.filter_result = FilterResult.COOLDOWN
            signal.filter_details = "In cooldown period"
            return signal
        
        # Run market filters
        filter_result, filter_detail = self.market_filter.run_all_filters(
            market_data, btc_price
        )
        
        if filter_result != FilterResult.PASSED:
            signal.signal_type = SignalType.NO_SIGNAL
            signal.filter_result = filter_result
            signal.filter_details = filter_detail
            return signal
        
        # Analyze wick pattern (pass volume_ratio for strength calculation)
        signal_type, strength, criteria = self.wick_analyzer.analyze_wick(
            market_data.candle,
            market_data.atr,
            market_data.vwap,
            market_data.volume_ratio
        )
        
        signal.signal_type = signal_type
        signal.strength = strength
        signal.criteria_met = criteria
        signal.trigger_candle = market_data.candle
        
        # Calculate entry/exit levels if signal is valid
        if signal_type != SignalType.NO_SIGNAL and strength > 0:
            entry, stop, target = self.calculate_entry_levels(
                signal_type,
                market_data.candle,
                market_data.atr,
                market_data.vwap
            )
            
            signal.suggested_entry = entry
            signal.suggested_stop = stop
            signal.suggested_target = target
            
            # Log signal generation
            logger.info(
                f"Signal generated: {symbol} {signal_type.value} "
                f"strength={strength:.2f} entry={entry:.6f} "
                f"stop={stop:.6f} target={target:.6f} "
                f"criteria={criteria}"
            )
        
        return signal


def calculate_signal_quality(signal: Signal) -> float:
    """
    Calculate overall signal quality score for ranking/filtering.
    
    Returns score from 0-100.
    """
    if not signal.is_valid:
        return 0.0
    
    score = signal.strength * 50  # Base score from strength
    
    # Bonus for multiple criteria met
    score += min(len(signal.criteria_met) * 10, 30)
    
    # Bonus for favorable risk/reward
    if signal.suggested_entry and signal.suggested_stop and signal.suggested_target:
        if signal.side == Side.LONG:
            risk = signal.suggested_entry - signal.suggested_stop
            reward = signal.suggested_target - signal.suggested_entry
        else:
            risk = signal.suggested_stop - signal.suggested_entry
            reward = signal.suggested_entry - signal.suggested_target
        
        if risk > 0:
            rr_ratio = reward / risk
            score += min(rr_ratio * 5, 20)
    
    return min(score, 100)
