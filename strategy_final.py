"""
Final Optimized Strategy for Small Accounts
Ultra-simple trend following with strict rules

Key insight: The market dropped 55% (SOL), 47% (XRP), 23% (BTC)
A simple "follow the trend" strategy should profit in bear markets

Rules:
1. ONLY SHORT when price is below EMA 50 AND EMA 50 is below EMA 100
2. ONLY LONG when price is above EMA 50 AND EMA 50 is above EMA 100
3. Enter on pullbacks/bounces
4. Use very tight stops
5. FEWER trades = HIGHER quality
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
    TREND_FOLLOW = "trend_follow"


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


class FinalStrategy:
    """
    Ultra-simple trend following strategy
    - Follows the trend religiously
    - Very selective entries
    - Tight risk management
    """

    def __init__(self):
        # Minimum bars between signals to prevent overtrading
        self.min_bars_between = 24  # At least 24 hours between trades
        self.last_signal_idx = -100

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df['ema_20'] = ema(df['close'], 20)
        df['ema_50'] = ema(df['close'], 50)
        df['ema_100'] = ema(df['close'], 100)
        df['ema_200'] = ema(df['close'], 200)

        # RSI
        df['rsi'] = rsi(df['close'], 14)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)

        # Trend definition (VERY clear trends only)
        # Bearish: EMA 20 < 50 < 100 and price < EMA 50
        df['bearish_trend'] = (
            (df['ema_20'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &
            (df['close'] < df['ema_50'])
        )

        # Bullish: EMA 20 > 50 > 100 and price > EMA 50
        df['bullish_trend'] = (
            (df['ema_20'] > df['ema_50']) &
            (df['ema_50'] > df['ema_100']) &
            (df['close'] > df['ema_50'])
        )

        # Trend strength - how far is price from EMAs
        df['trend_strength'] = abs(df['close'] - df['ema_50']) / df['atr']

        # Recent price change
        df['pct_change_5'] = df['close'].pct_change(5) * 100
        df['pct_change_20'] = df['close'].pct_change(20) * 100

        # Candle analysis
        df['body'] = df['close'] - df['open']
        df['bullish_candle'] = df['close'] > df['open']
        df['bearish_candle'] = df['close'] < df['open']

        # Distance from EMA 20 (for pullback detection)
        df['dist_from_ema20'] = (df['close'] - df['ema_20']) / df['atr']

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 250:
            return None

        df = self.calculate_indicators(df)
        current_idx = len(df)

        # Check minimum bars between trades
        if current_idx - self.last_signal_idx < self.min_bars_between:
            return None

        c = df.iloc[-1]
        p1 = df.iloc[-2]

        if pd.isna(c['atr']) or c['atr'] == 0 or pd.isna(c['ema_100']):
            return None

        entry = c['close']
        atr_val = c['atr']

        signal = None

        # ===== SHORT SETUP =====
        # Very selective: Only in clear downtrends with pullback opportunity
        if c['bearish_trend']:
            # Pullback: Price bounced up toward EMA 20 or 50
            pullback_to_ema = (
                c['dist_from_ema20'] > -0.5 or  # Price near or above EMA 20
                c['rsi'] > 50  # RSI bounced up
            )

            # Confirmation: Price starting to turn back down
            turning_down = (
                c['bearish_candle'] and
                (c['rsi'] < p1['rsi'] or c['close'] < p1['close'])
            )

            # Strong trend (EMAs clearly separated)
            strong_trend = c['trend_strength'] > 1.5

            if pullback_to_ema and turning_down and strong_trend:
                # Tight stop above recent high
                recent_high = df['high'].iloc[-5:].max()
                stop_loss = min(recent_high + atr_val * 0.3, entry + atr_val * 1.5)

                # Target continuation of downtrend
                take_profit = entry - atr_val * 2.5
                take_profit_2 = entry - atr_val * 4

                # Ensure R:R is good
                risk = stop_loss - entry
                reward = entry - take_profit
                if reward >= risk * 1.5:
                    signal = TradeSignal(
                        signal_type=SignalType.SHORT,
                        strategy=StrategyType.TREND_FOLLOW,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        take_profit_2=take_profit_2,
                        position_size_pct=0.5,
                        confidence=0.75,
                        reason=f"Downtrend continuation | RSI {c['rsi']:.0f}"
                    )

        # ===== LONG SETUP =====
        elif c['bullish_trend']:
            # Pullback: Price dipped toward EMA 20 or 50
            pullback_to_ema = (
                c['dist_from_ema20'] < 0.5 or
                c['rsi'] < 50
            )

            # Confirmation: Price starting to turn back up
            turning_up = (
                c['bullish_candle'] and
                (c['rsi'] > p1['rsi'] or c['close'] > p1['close'])
            )

            strong_trend = c['trend_strength'] > 1.5

            if pullback_to_ema and turning_up and strong_trend:
                recent_low = df['low'].iloc[-5:].min()
                stop_loss = max(recent_low - atr_val * 0.3, entry - atr_val * 1.5)

                take_profit = entry + atr_val * 2.5
                take_profit_2 = entry + atr_val * 4

                risk = entry - stop_loss
                reward = take_profit - entry
                if reward >= risk * 1.5:
                    signal = TradeSignal(
                        signal_type=SignalType.LONG,
                        strategy=StrategyType.TREND_FOLLOW,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        take_profit_2=take_profit_2,
                        position_size_pct=0.5,
                        confidence=0.75,
                        reason=f"Uptrend continuation | RSI {c['rsi']:.0f}"
                    )

        if signal and self._validate(signal):
            self.last_signal_idx = current_idx
            return signal

        return None

    def _validate(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        # R:R at least 1.5:1
        if reward / risk < 1.5:
            return False

        # Max stop 2%
        stop_pct = risk / entry
        if stop_pct > 0.02:
            return False

        return True


# For import compatibility
SimpleStrategy = FinalStrategy
