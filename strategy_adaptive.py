"""
Adaptive Trend Strategy for Small Accounts
Designed to work in BOTH bull and bear markets

Key insight from backtesting:
- The historical data is from a BEAR market (SOL -55%, XRP -47%, BTC -23%)
- Any long-biased strategy will fail in bear markets
- Need to follow the trend, not fight it

Strategy:
1. Identify the dominant trend (using longer EMAs)
2. Only trade IN THE DIRECTION of the trend
3. Enter on pullbacks within the trend
4. Use tight stops for protection
"""

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class StrategyType(Enum):
    ADAPTIVE_TREND = "adaptive_trend"


@dataclass
class TradeSignal:
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
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(span=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=period, adjust=False).mean()
    return 100 - (100 / (1 + gain / (loss + 0.0001)))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


class AdaptiveTrendStrategy:
    """
    Adaptive Trend-Following Strategy

    Core principle: Follow the trend, don't fight it
    - In downtrends: Only SHORT
    - In uptrends: Only LONG
    - Enter on pullbacks for better entries
    """

    def __init__(self):
        # Trend detection
        self.trend_fast = 20  # Fast EMA
        self.trend_slow = 50  # Slow EMA
        self.trend_filter = 100  # Trend filter EMA

        # Entry
        self.rsi_period = 14
        self.pullback_rsi_long = 40  # RSI below this for long entry
        self.pullback_rsi_short = 60  # RSI above this for short entry

        # Risk
        self.atr_period = 14
        self.stop_atr_mult = 1.5
        self.target_atr_mult = 2.5

        # Track signals
        self.last_signal_bar = -100  # Prevent overtrading

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs for trend
        df['ema_20'] = ema(df['close'], self.trend_fast)
        df['ema_50'] = ema(df['close'], self.trend_slow)
        df['ema_100'] = ema(df['close'], self.trend_filter)

        # SMA for confirmation
        df['sma_200'] = sma(df['close'], 200)

        # RSI
        df['rsi'] = rsi(df['close'], self.rsi_period)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], self.atr_period)
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # Trend identification
        # Strong downtrend: All EMAs bearish aligned
        df['strong_downtrend'] = (
            (df['ema_20'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &
            (df['close'] < df['ema_50'])
        )

        # Strong uptrend: All EMAs bullish aligned
        df['strong_uptrend'] = (
            (df['ema_20'] > df['ema_50']) &
            (df['ema_50'] > df['ema_100']) &
            (df['close'] > df['ema_50'])
        )

        # Weak/ranging market
        df['ranging'] = ~(df['strong_downtrend'] | df['strong_uptrend'])

        # Price position relative to EMAs
        df['above_ema20'] = df['close'] > df['ema_20']
        df['below_ema20'] = df['close'] < df['ema_20']

        # Momentum
        df['momentum_5'] = df['close'].pct_change(5) * 100
        df['momentum_20'] = df['close'].pct_change(20) * 100

        # Candle patterns
        df['bullish_candle'] = (df['close'] > df['open']) & ((df['close'] - df['open']) > df['atr'] * 0.3)
        df['bearish_candle'] = (df['close'] < df['open']) & ((df['open'] - df['close']) > df['atr'] * 0.3)

        # Pullback detection
        df['pullback_up'] = df['rsi'] < self.pullback_rsi_long  # Oversold in uptrend
        df['pullback_down'] = df['rsi'] > self.pullback_rsi_short  # Overbought in downtrend

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 250:
            return None

        df = self.calculate_indicators(df)

        c = df.iloc[-1]
        p1 = df.iloc[-2]
        p2 = df.iloc[-3]

        if pd.isna(c['atr']) or c['atr'] == 0 or pd.isna(c['ema_100']):
            return None

        current_bar = len(df)
        entry = c['close']
        atr_val = c['atr']

        signal = None

        # ===== SHORT SIGNALS (for downtrend) =====
        # This is the KEY for bear markets
        if c['strong_downtrend']:
            # Look for pullback to sell (price rallied up a bit)
            pullback_conditions = (
                c['pullback_down'] or  # RSI overbought
                c['above_ema20'] or  # Price above EMA20
                (p1['rsi'] > 55 and c['rsi'] < p1['rsi'])  # RSI turning down
            )

            # Confirmation
            confirmation = (
                c['bearish_candle'] or  # Bearish candle
                (c['rsi'] < p1['rsi'] and p1['rsi'] > 50)  # RSI turning from overbought
            )

            if pullback_conditions and confirmation:
                stop_loss = entry + atr_val * self.stop_atr_mult
                take_profit = entry - atr_val * self.target_atr_mult
                take_profit_2 = entry - atr_val * 4

                # Ensure stop isn't too wide
                stop_pct = (stop_loss - entry) / entry
                if stop_pct > 0.025:  # Max 2.5% stop
                    stop_loss = entry * 1.025

                confidence = 0.72
                reasons = ["Strong downtrend"]

                if c['momentum_20'] < -5:
                    confidence += 0.05
                    reasons.append("Strong bearish momentum")

                if c['above_ema20'] and c['bearish_candle']:
                    confidence += 0.05
                    reasons.append("Rejection from EMA20")

                signal = TradeSignal(
                    signal_type=SignalType.SHORT,
                    strategy=StrategyType.ADAPTIVE_TREND,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    take_profit_2=take_profit_2,
                    position_size_pct=0.5,
                    confidence=min(confidence, 0.90),
                    reason=" | ".join(reasons)
                )

        # ===== LONG SIGNALS (for uptrend) =====
        elif c['strong_uptrend']:
            # Look for pullback to buy
            pullback_conditions = (
                c['pullback_up'] or  # RSI oversold
                c['below_ema20'] or  # Price below EMA20
                (p1['rsi'] < 45 and c['rsi'] > p1['rsi'])  # RSI turning up
            )

            confirmation = (
                c['bullish_candle'] or
                (c['rsi'] > p1['rsi'] and p1['rsi'] < 50)
            )

            if pullback_conditions and confirmation:
                stop_loss = entry - atr_val * self.stop_atr_mult
                take_profit = entry + atr_val * self.target_atr_mult
                take_profit_2 = entry + atr_val * 4

                stop_pct = (entry - stop_loss) / entry
                if stop_pct > 0.025:
                    stop_loss = entry * 0.975

                confidence = 0.72
                reasons = ["Strong uptrend"]

                if c['momentum_20'] > 5:
                    confidence += 0.05
                    reasons.append("Strong bullish momentum")

                if c['below_ema20'] and c['bullish_candle']:
                    confidence += 0.05
                    reasons.append("Bounce from EMA20")

                signal = TradeSignal(
                    signal_type=SignalType.LONG,
                    strategy=StrategyType.ADAPTIVE_TREND,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    take_profit_2=take_profit_2,
                    position_size_pct=0.5,
                    confidence=min(confidence, 0.90),
                    reason=" | ".join(reasons)
                )

        # Validate signal
        if signal and not self._validate_signal(signal):
            return None

        return signal

    def _validate_signal(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        # Minimum R:R
        if reward / risk < 1.5:
            return False

        # Maximum stop distance
        stop_pct = risk / entry
        if stop_pct > 0.03:
            return False

        # Minimum confidence
        if signal.confidence < 0.70:
            return False

        return True


# Export as default
SimpleStrategy = AdaptiveTrendStrategy
