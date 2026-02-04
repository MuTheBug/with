"""
Small Account Trading Strategy
Optimized for rapid account growth with strict risk management

Strategy Components:
1. Momentum Scalping - Trade with strong momentum
2. Mean Reversion - Fade extreme moves
3. Breakout Trading - Catch breakouts with confirmation
4. Multi-timeframe analysis for better entries
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from indicators import (
    calculate_all_indicators,
    detect_support_resistance,
    SignalResult,
    ema, rsi, atr, macd, bollinger_bands
)
from config import strategy_config


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOW = "trend_follow"


@dataclass
class TradeSignal:
    """Complete trade signal with entry/exit levels"""
    signal_type: SignalType
    strategy: StrategyType
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_2: float  # Second TP for partial close
    position_size_pct: float
    confidence: float  # 0-1 confidence score
    reason: str
    timestamp: pd.Timestamp = None


class SmallAccountStrategy:
    """
    Multi-strategy trading system for small accounts
    Designed to compound $4.5 quickly with strict risk management
    """

    def __init__(self, config: Any = None):
        self.config = config or strategy_config

        # Strategy weights (adjusted based on market conditions)
        self.strategy_weights = {
            StrategyType.MOMENTUM: 0.35,
            StrategyType.MEAN_REVERSION: 0.25,
            StrategyType.BREAKOUT: 0.25,
            StrategyType.TREND_FOLLOW: 0.15
        }

        # Track recent signals to avoid overtrading
        self.recent_signals: List[TradeSignal] = []
        self.max_signals_per_hour = 3

        # Performance tracking for adaptive weights
        self.strategy_performance: Dict[StrategyType, List[float]] = {
            s: [] for s in StrategyType
        }

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Main analysis function - returns trade signal if conditions met

        Args:
            df: DataFrame with OHLCV data (must have sufficient history)

        Returns:
            TradeSignal if conditions met, None otherwise
        """
        if len(df) < 200:
            return None

        # Calculate all indicators
        df = calculate_all_indicators(df)

        # Get current values
        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check if we're in a tradeable state
        if not self._is_tradeable(df):
            return None

        # Collect signals from all strategies
        signals = []

        # 1. Momentum Strategy
        momentum_signal = self._momentum_strategy(df, current, prev)
        if momentum_signal:
            signals.append(momentum_signal)

        # 2. Mean Reversion Strategy
        mean_rev_signal = self._mean_reversion_strategy(df, current, prev)
        if mean_rev_signal:
            signals.append(mean_rev_signal)

        # 3. Breakout Strategy
        breakout_signal = self._breakout_strategy(df, current, prev)
        if breakout_signal:
            signals.append(breakout_signal)

        # 4. Trend Following Strategy
        trend_signal = self._trend_follow_strategy(df, current, prev)
        if trend_signal:
            signals.append(trend_signal)

        # Select best signal based on confidence and strategy weights
        if not signals:
            return None

        best_signal = self._select_best_signal(signals)

        # Validate signal passes all filters
        if self._validate_signal(best_signal, df):
            return best_signal

        return None

    def _is_tradeable(self, df: pd.DataFrame) -> bool:
        """Check if market conditions are suitable for trading"""
        current = df.iloc[-1]

        # Check minimum volatility (need some movement)
        if current['atr_pct'] < 0.5:  # Less than 0.5% ATR
            return False

        # Check maximum volatility (too risky)
        if current['atr_pct'] > 5.0:  # More than 5% ATR
            return False

        # Check volume
        if current['volume_ratio'] < 0.5:  # Very low volume
            return False

        return True

    def _momentum_strategy(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        prev: pd.Series
    ) -> Optional[TradeSignal]:
        """
        Momentum Scalping Strategy
        - Trade in direction of strong momentum
        - Quick entries and exits
        - Best for trending markets
        """
        signal_type = SignalType.NEUTRAL
        confidence = 0.0
        reasons = []

        # EMA alignment for trend
        ema_bullish = (current['ema_8'] > current['ema_21'] > current['ema_50'])
        ema_bearish = (current['ema_8'] < current['ema_21'] < current['ema_50'])

        # MACD momentum
        macd_bullish = (current['macd'] > current['macd_signal'] and
                       current['macd_hist'] > prev['macd_hist'])
        macd_bearish = (current['macd'] < current['macd_signal'] and
                       current['macd_hist'] < prev['macd_hist'])

        # RSI momentum (not overbought/oversold)
        rsi_ok_long = 40 < current['rsi'] < 70
        rsi_ok_short = 30 < current['rsi'] < 60

        # Volume confirmation
        volume_surge = current['volume_ratio'] > self.config.min_volume_ratio

        # Strong momentum candle
        candle_bullish = (current['close'] > current['open'] and
                        current['body_size'] > current['atr'] * 0.5)
        candle_bearish = (current['close'] < current['open'] and
                        current['body_size'] > current['atr'] * 0.5)

        # LONG Signal
        if ema_bullish and macd_bullish and rsi_ok_long and volume_surge and candle_bullish:
            signal_type = SignalType.LONG
            confidence = 0.7

            reasons.append("EMA bullish alignment")
            if current['adx'] > 25:
                confidence += 0.1
                reasons.append("Strong trend (ADX)")
            if current['supertrend_dir'] == 1:
                confidence += 0.1
                reasons.append("Supertrend bullish")

        # SHORT Signal
        elif ema_bearish and macd_bearish and rsi_ok_short and volume_surge and candle_bearish:
            signal_type = SignalType.SHORT
            confidence = 0.7

            reasons.append("EMA bearish alignment")
            if current['adx'] > 25:
                confidence += 0.1
                reasons.append("Strong trend (ADX)")
            if current['supertrend_dir'] == -1:
                confidence += 0.1
                reasons.append("Supertrend bearish")

        if signal_type == SignalType.NEUTRAL:
            return None

        # Calculate entry/exit levels
        entry_price = current['close']
        atr_val = current['atr']

        if signal_type == SignalType.LONG:
            stop_loss = entry_price - (atr_val * self.config.atr_multiplier_sl)
            take_profit = entry_price + (atr_val * self.config.atr_multiplier_tp)
            take_profit_2 = entry_price + (atr_val * self.config.atr_multiplier_tp * 1.5)
        else:
            stop_loss = entry_price + (atr_val * self.config.atr_multiplier_sl)
            take_profit = entry_price - (atr_val * self.config.atr_multiplier_tp)
            take_profit_2 = entry_price - (atr_val * self.config.atr_multiplier_tp * 1.5)

        return TradeSignal(
            signal_type=signal_type,
            strategy=StrategyType.MOMENTUM,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            position_size_pct=self.config.position_size_pct,
            confidence=min(confidence, 1.0),
            reason=" | ".join(reasons),
            timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else None
        )

    def _mean_reversion_strategy(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        prev: pd.Series
    ) -> Optional[TradeSignal]:
        """
        Mean Reversion Strategy
        - Fade extreme moves
        - Trade bounces from Bollinger Bands
        - Best for ranging markets
        """
        signal_type = SignalType.NEUTRAL
        confidence = 0.0
        reasons = []

        # Bollinger Band position
        bb_oversold = current['bb_position'] < 0.1  # Near lower band
        bb_overbought = current['bb_position'] > 0.9  # Near upper band

        # RSI extreme
        rsi_oversold = current['rsi'] < self.config.rsi_oversold
        rsi_overbought = current['rsi'] > self.config.rsi_overbought

        # Stochastic extreme
        stoch_oversold = current['stoch_k'] < 20
        stoch_overbought = current['stoch_k'] > 80

        # Williams %R extreme
        williams_oversold = current['williams_r'] < -80
        williams_overbought = current['williams_r'] > -20

        # CCI extreme
        cci_oversold = current['cci'] < -100
        cci_overbought = current['cci'] > 100

        # Reversal candle pattern
        bullish_reversal = (prev['close'] < prev['open'] and
                          current['close'] > current['open'] and
                          current['close'] > prev['close'])
        bearish_reversal = (prev['close'] > prev['open'] and
                          current['close'] < current['open'] and
                          current['close'] < prev['close'])

        # Count oversold/overbought confirmations
        oversold_count = sum([bb_oversold, rsi_oversold, stoch_oversold,
                            williams_oversold, cci_oversold])
        overbought_count = sum([bb_overbought, rsi_overbought, stoch_overbought,
                               williams_overbought, cci_overbought])

        # LONG Signal (oversold bounce)
        if oversold_count >= 3 and bullish_reversal:
            signal_type = SignalType.LONG
            confidence = 0.5 + (oversold_count * 0.1)
            reasons.append(f"Oversold ({oversold_count} confirmations)")
            reasons.append("Bullish reversal candle")

            # Not in strong downtrend
            if current['adx'] < 30:
                confidence += 0.1
                reasons.append("Weak trend (mean reversion friendly)")

        # SHORT Signal (overbought fade)
        elif overbought_count >= 3 and bearish_reversal:
            signal_type = SignalType.SHORT
            confidence = 0.5 + (overbought_count * 0.1)
            reasons.append(f"Overbought ({overbought_count} confirmations)")
            reasons.append("Bearish reversal candle")

            if current['adx'] < 30:
                confidence += 0.1
                reasons.append("Weak trend (mean reversion friendly)")

        if signal_type == SignalType.NEUTRAL:
            return None

        # Tighter stops for mean reversion
        entry_price = current['close']
        atr_val = current['atr']

        if signal_type == SignalType.LONG:
            stop_loss = min(current['low'], entry_price - atr_val)
            take_profit = current['bb_middle']  # Target middle band
            take_profit_2 = current['ema_21']
        else:
            stop_loss = max(current['high'], entry_price + atr_val)
            take_profit = current['bb_middle']
            take_profit_2 = current['ema_21']

        return TradeSignal(
            signal_type=signal_type,
            strategy=StrategyType.MEAN_REVERSION,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            position_size_pct=self.config.position_size_pct * 0.8,  # Smaller size
            confidence=min(confidence, 1.0),
            reason=" | ".join(reasons),
            timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else None
        )

    def _breakout_strategy(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        prev: pd.Series
    ) -> Optional[TradeSignal]:
        """
        Breakout Strategy
        - Trade breakouts from consolidation
        - Volume confirmation required
        - Best for volatile markets
        """
        signal_type = SignalType.NEUTRAL
        confidence = 0.0
        reasons = []

        # Detect support/resistance
        supports, resistances = detect_support_resistance(df, lookback=50)

        if not supports or not resistances:
            return None

        current_price = current['close']
        nearest_support = min(supports, key=lambda x: abs(x - current_price))
        nearest_resistance = min(resistances, key=lambda x: abs(x - current_price))

        # Bollinger Band squeeze (consolidation)
        bb_squeeze = current['bb_width'] < df['bb_width'].rolling(20).mean().iloc[-1] * 0.8

        # Strong volume on breakout
        volume_breakout = current['volume_ratio'] > 1.5

        # Price breakout above resistance
        if (current_price > nearest_resistance and
            prev['close'] <= nearest_resistance and
            volume_breakout):

            signal_type = SignalType.LONG
            confidence = 0.65
            reasons.append(f"Breakout above resistance {nearest_resistance:.4f}")

            if bb_squeeze:
                confidence += 0.15
                reasons.append("Bollinger squeeze breakout")

            if current['adx'] > 20:
                confidence += 0.1
                reasons.append("ADX confirms momentum")

        # Price breakout below support
        elif (current_price < nearest_support and
              prev['close'] >= nearest_support and
              volume_breakout):

            signal_type = SignalType.SHORT
            confidence = 0.65
            reasons.append(f"Breakdown below support {nearest_support:.4f}")

            if bb_squeeze:
                confidence += 0.15
                reasons.append("Bollinger squeeze breakdown")

            if current['adx'] > 20:
                confidence += 0.1
                reasons.append("ADX confirms momentum")

        if signal_type == SignalType.NEUTRAL:
            return None

        entry_price = current['close']
        atr_val = current['atr']

        if signal_type == SignalType.LONG:
            stop_loss = nearest_resistance - (atr_val * 0.5)  # Just below breakout
            take_profit = entry_price + (atr_val * 3)  # Larger target for breakouts
            take_profit_2 = entry_price + (atr_val * 5)
        else:
            stop_loss = nearest_support + (atr_val * 0.5)
            take_profit = entry_price - (atr_val * 3)
            take_profit_2 = entry_price - (atr_val * 5)

        return TradeSignal(
            signal_type=signal_type,
            strategy=StrategyType.BREAKOUT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            position_size_pct=self.config.position_size_pct * 0.7,  # Smaller for breakouts
            confidence=min(confidence, 1.0),
            reason=" | ".join(reasons),
            timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else None
        )

    def _trend_follow_strategy(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        prev: pd.Series
    ) -> Optional[TradeSignal]:
        """
        Trend Following Strategy
        - Trade pullbacks in strong trends
        - Use EMA as dynamic support/resistance
        - Best for trending markets
        """
        signal_type = SignalType.NEUTRAL
        confidence = 0.0
        reasons = []

        # Strong trend detection
        strong_uptrend = (current['adx'] > 25 and
                        current['plus_di'] > current['minus_di'] and
                        current['ema_21'] > current['ema_50'])

        strong_downtrend = (current['adx'] > 25 and
                          current['minus_di'] > current['plus_di'] and
                          current['ema_21'] < current['ema_50'])

        # Pullback to EMA
        pullback_to_ema21 = abs(current['close'] - current['ema_21']) / current['atr'] < 0.5

        # RSI not extreme
        rsi_ok = 35 < current['rsi'] < 65

        # Supertrend confirmation
        supertrend_long = current['supertrend_dir'] == 1
        supertrend_short = current['supertrend_dir'] == -1

        # LONG: Pullback in uptrend
        if strong_uptrend and pullback_to_ema21 and rsi_ok and supertrend_long:
            # Bullish candle at EMA
            if current['close'] > current['open']:
                signal_type = SignalType.LONG
                confidence = 0.7
                reasons.append("Strong uptrend (ADX > 25)")
                reasons.append("Pullback to EMA21")
                reasons.append("Bullish candle confirmation")

                if current['macd'] > current['macd_signal']:
                    confidence += 0.1
                    reasons.append("MACD bullish")

        # SHORT: Pullback in downtrend
        elif strong_downtrend and pullback_to_ema21 and rsi_ok and supertrend_short:
            if current['close'] < current['open']:
                signal_type = SignalType.SHORT
                confidence = 0.7
                reasons.append("Strong downtrend (ADX > 25)")
                reasons.append("Pullback to EMA21")
                reasons.append("Bearish candle confirmation")

                if current['macd'] < current['macd_signal']:
                    confidence += 0.1
                    reasons.append("MACD bearish")

        if signal_type == SignalType.NEUTRAL:
            return None

        entry_price = current['close']
        atr_val = current['atr']

        if signal_type == SignalType.LONG:
            stop_loss = current['ema_21'] - (atr_val * 1.2)
            take_profit = entry_price + (atr_val * 2)
            take_profit_2 = entry_price + (atr_val * 4)
        else:
            stop_loss = current['ema_21'] + (atr_val * 1.2)
            take_profit = entry_price - (atr_val * 2)
            take_profit_2 = entry_price - (atr_val * 4)

        return TradeSignal(
            signal_type=signal_type,
            strategy=StrategyType.TREND_FOLLOW,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_2=take_profit_2,
            position_size_pct=self.config.position_size_pct,
            confidence=min(confidence, 1.0),
            reason=" | ".join(reasons),
            timestamp=df.index[-1] if isinstance(df.index[-1], pd.Timestamp) else None
        )

    def _select_best_signal(self, signals: List[TradeSignal]) -> TradeSignal:
        """Select the best signal based on confidence and strategy performance"""

        if len(signals) == 1:
            return signals[0]

        # Score each signal
        scored_signals = []
        for signal in signals:
            # Base score is confidence
            score = signal.confidence

            # Apply strategy weight
            weight = self.strategy_weights.get(signal.strategy, 0.25)
            score *= (1 + weight)

            # Apply historical performance adjustment
            perf = self.strategy_performance.get(signal.strategy, [])
            if len(perf) >= 5:
                win_rate = sum(1 for p in perf[-10:] if p > 0) / len(perf[-10:])
                score *= (0.5 + win_rate)

            scored_signals.append((signal, score))

        # Return highest scored signal
        scored_signals.sort(key=lambda x: x[1], reverse=True)
        return scored_signals[0][0]

    def _validate_signal(self, signal: TradeSignal, df: pd.DataFrame) -> bool:
        """Final validation before returning signal"""

        # Risk/reward check - require at least 2:1 R:R for small accounts
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0 or reward / risk < 2.0:  # Minimum 2:1 R:R
            return False

        # Stop loss sanity check - max 3% stop distance
        stop_pct = risk / entry
        if stop_pct > 0.03:
            return False

        # Confidence threshold - higher bar for small accounts
        if signal.confidence < 0.70:
            return False

        # Check we're not overtrading
        if len(self.recent_signals) >= self.max_signals_per_hour:
            oldest = self.recent_signals[0]
            if oldest.timestamp:
                current_time = df.index[-1]
                if hasattr(current_time, 'hour'):
                    if oldest.timestamp.hour == current_time.hour:
                        return False

        return True

    def update_performance(self, strategy: StrategyType, pnl: float):
        """Update strategy performance tracking"""
        self.strategy_performance[strategy].append(pnl)

        # Keep only recent history
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-100:]

        # Adaptive weight adjustment
        self._adjust_weights()

    def _adjust_weights(self):
        """Adjust strategy weights based on recent performance"""
        total_trades = 0
        strategy_scores = {}

        for strategy, perfs in self.strategy_performance.items():
            if len(perfs) >= 5:
                recent = perfs[-20:]
                win_rate = sum(1 for p in recent if p > 0) / len(recent)
                avg_pnl = np.mean(recent)
                score = win_rate * (1 + avg_pnl / 100)
                strategy_scores[strategy] = score
                total_trades += len(recent)

        if strategy_scores and total_trades >= 20:
            total_score = sum(strategy_scores.values())
            if total_score > 0:
                for strategy, score in strategy_scores.items():
                    new_weight = score / total_score
                    # Smooth adjustment
                    current = self.strategy_weights[strategy]
                    self.strategy_weights[strategy] = current * 0.7 + new_weight * 0.3


# Global strategy instance
strategy = SmallAccountStrategy()
