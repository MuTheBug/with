"""
Simple High-Probability Trading Strategy v2
Optimized for small accounts - focuses on quality over quantity

Key principles:
1. Trade only with the trend
2. Enter on pullbacks, not breakouts
3. Use tight stops with good R:R
4. Fewer trades = higher quality
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class StrategyType(Enum):
    TREND_PULLBACK = "trend_pullback"


@dataclass
class TradeSignal:
    """Trade signal with entry/exit levels"""
    signal_type: SignalType
    strategy: StrategyType
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_2: float
    position_size_pct: float
    confidence: float
    reason: str
    timestamp: pd.Timestamp = None


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index - measures trend strength"""
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = atr(high, low, close, 1)
    atr_smooth = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    return dx.rolling(window=period).mean()


class SimpleStrategy:
    """
    Simple Trend-Pullback Strategy

    Logic:
    1. Identify trend using EMA 50 and EMA 200
    2. Wait for pullback to EMA 20
    3. Enter when RSI shows reversal
    4. Use ATR for stop loss
    """

    def __init__(self):
        # EMA periods
        self.fast_ema = 20
        self.medium_ema = 50
        self.slow_ema = 200

        # RSI
        self.rsi_period = 14
        self.rsi_oversold = 35
        self.rsi_overbought = 65

        # ATR
        self.atr_period = 14

        # Risk parameters
        self.stop_atr_mult = 1.5
        self.tp_atr_mult = 3.0  # 2:1 R:R

        # Trend strength
        self.min_adx = 20

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        df = df.copy()

        # EMAs
        df['ema_20'] = ema(df['close'], self.fast_ema)
        df['ema_50'] = ema(df['close'], self.medium_ema)
        df['ema_200'] = ema(df['close'], self.slow_ema)

        # RSI
        df['rsi'] = rsi(df['close'], self.rsi_period)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], self.atr_period)

        # ADX for trend strength
        df['adx'] = adx(df['high'], df['low'], df['close'], self.atr_period)

        # Trend direction
        df['uptrend'] = (df['ema_50'] > df['ema_200']) & (df['close'] > df['ema_50'])
        df['downtrend'] = (df['ema_50'] < df['ema_200']) & (df['close'] < df['ema_50'])

        # Pullback detection
        df['near_ema20'] = abs(df['close'] - df['ema_20']) / df['atr'] < 1.0

        # Candle patterns
        df['bullish_candle'] = df['close'] > df['open']
        df['bearish_candle'] = df['close'] < df['open']

        # Price momentum
        df['momentum'] = df['close'].pct_change(5) * 100

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Analyze data and generate trading signal

        Returns signal only when high-probability setup detected
        """
        if len(df) < 250:
            return None

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Get current and previous bars
        current = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]

        # Check for valid data
        if pd.isna(current['ema_200']) or pd.isna(current['adx']):
            return None

        signal = None

        # ===== LONG SETUP =====
        # Conditions:
        # 1. Strong uptrend (EMA 50 > EMA 200, price > EMA 50)
        # 2. Pullback to EMA 20 area
        # 3. RSI was oversold and turning up
        # 4. Bullish candle confirmation
        # 5. ADX > 20 (trending market)

        if (current['uptrend'] and
            current['adx'] > self.min_adx and
            current['near_ema20'] and
            prev['rsi'] < 40 and current['rsi'] > prev['rsi'] and
            current['bullish_candle'] and
            current['close'] > current['ema_20']):

            entry_price = current['close']
            atr_val = current['atr']

            # Stop below recent swing low or EMA 20
            stop_loss = min(
                df['low'].iloc[-5:].min(),
                current['ema_20'] - atr_val * 0.5
            )

            # Ensure stop isn't too far (max 2% from entry)
            max_stop_distance = entry_price * 0.02
            if entry_price - stop_loss > max_stop_distance:
                stop_loss = entry_price - max_stop_distance

            # Take profit based on ATR
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2.5)  # 2.5:1 R:R
            take_profit_2 = entry_price + (risk * 4)  # 4:1 R:R

            confidence = 0.70
            reasons = ["Uptrend confirmed", "Pullback to EMA20", "RSI reversal"]

            # Boost confidence for stronger setups
            if current['adx'] > 30:
                confidence += 0.1
                reasons.append("Strong trend (ADX>30)")

            if current['momentum'] > 0:
                confidence += 0.05
                reasons.append("Positive momentum")

            signal = TradeSignal(
                signal_type=SignalType.LONG,
                strategy=StrategyType.TREND_PULLBACK,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=min(confidence, 1.0),
                reason=" | ".join(reasons),
                timestamp=df.index[-1] if hasattr(df.index[-1], 'strftime') else None
            )

        # ===== SHORT SETUP =====
        # Mirror conditions for downtrend

        elif (current['downtrend'] and
              current['adx'] > self.min_adx and
              current['near_ema20'] and
              prev['rsi'] > 60 and current['rsi'] < prev['rsi'] and
              current['bearish_candle'] and
              current['close'] < current['ema_20']):

            entry_price = current['close']
            atr_val = current['atr']

            # Stop above recent swing high or EMA 20
            stop_loss = max(
                df['high'].iloc[-5:].max(),
                current['ema_20'] + atr_val * 0.5
            )

            # Ensure stop isn't too far
            max_stop_distance = entry_price * 0.02
            if stop_loss - entry_price > max_stop_distance:
                stop_loss = entry_price + max_stop_distance

            # Take profit
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2.5)
            take_profit_2 = entry_price - (risk * 4)

            confidence = 0.70
            reasons = ["Downtrend confirmed", "Pullback to EMA20", "RSI reversal"]

            if current['adx'] > 30:
                confidence += 0.1
                reasons.append("Strong trend (ADX>30)")

            if current['momentum'] < 0:
                confidence += 0.05
                reasons.append("Negative momentum")

            signal = TradeSignal(
                signal_type=SignalType.SHORT,
                strategy=StrategyType.TREND_PULLBACK,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=min(confidence, 1.0),
                reason=" | ".join(reasons),
                timestamp=df.index[-1] if hasattr(df.index[-1], 'strftime') else None
            )

        # Validate signal
        if signal and not self._validate_signal(signal):
            return None

        return signal

    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate signal quality"""

        # Check R:R ratio
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        rr_ratio = reward / risk
        if rr_ratio < 2.0:
            return False

        # Check stop distance (max 2.5% for small accounts)
        stop_pct = risk / entry
        if stop_pct > 0.025:
            return False

        # Confidence threshold
        if signal.confidence < 0.70:
            return False

        return True


# For backward compatibility
SmallAccountStrategy = SimpleStrategy
