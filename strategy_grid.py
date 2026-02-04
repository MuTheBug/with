"""
Grid/DCA Trading Strategy for Small Accounts
Works by averaging into positions at predefined price levels

Key advantages:
1. Doesn't try to predict direction
2. Profits from volatility
3. Works in ranging markets
4. Simple and mechanical
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0


class StrategyType(Enum):
    GRID = "grid"


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


class GridStrategy:
    """
    Grid Trading Strategy

    Logic:
    1. Calculate a price range based on recent volatility
    2. Set up grid levels within this range
    3. Buy when price drops to lower grid levels
    4. Sell when price rises to upper grid levels
    5. Profit from the oscillation
    """

    def __init__(self):
        self.grid_size = 0.015  # 1.5% between grid levels
        self.num_grids = 5  # Number of grid levels above and below
        self.last_trade_price = None
        self.last_trade_side = None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs for trend
        df['ema_20'] = ema(df['close'], 20)
        df['ema_50'] = ema(df['close'], 50)
        df['ema_100'] = ema(df['close'], 100)

        # RSI
        df['rsi'] = rsi(df['close'], 14)

        # ATR
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)
        df['atr_pct'] = df['atr'] / df['close']

        # Price range
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['range_mid'] = (df['high_20'] + df['low_20']) / 2

        # Position in range (0 = at low, 1 = at high)
        df['range_pos'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 0.0001)

        # Trend
        df['uptrend'] = df['ema_20'] > df['ema_50']
        df['strong_trend'] = abs(df['ema_20'] - df['ema_50']) / df['close'] > 0.01

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 100:
            return None

        df = self.calculate_indicators(df)
        c = df.iloc[-1]

        if pd.isna(c['atr']) or c['atr'] == 0:
            return None

        current_price = c['close']
        atr_val = c['atr']
        rsi_val = c['rsi']
        range_pos = c['range_pos']

        signal = None

        # Grid trading in ranging markets (avoid strong trends)
        if c['strong_trend']:
            # In strong trends, trade with trend only
            if c['uptrend'] and rsi_val < 40 and range_pos < 0.3:
                # Buy dips in uptrend
                entry = current_price
                stop_loss = entry - atr_val * 2
                take_profit = entry + atr_val * 2.5
                take_profit_2 = entry + atr_val * 4

                signal = TradeSignal(
                    signal_type=SignalType.LONG,
                    strategy=StrategyType.GRID,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    take_profit_2=take_profit_2,
                    position_size_pct=0.4,
                    confidence=0.72,
                    reason="Buy dip in uptrend"
                )
            elif not c['uptrend'] and rsi_val > 60 and range_pos > 0.7:
                # Sell rallies in downtrend
                entry = current_price
                stop_loss = entry + atr_val * 2
                take_profit = entry - atr_val * 2.5
                take_profit_2 = entry - atr_val * 4

                signal = TradeSignal(
                    signal_type=SignalType.SHORT,
                    strategy=StrategyType.GRID,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    take_profit_2=take_profit_2,
                    position_size_pct=0.4,
                    confidence=0.72,
                    reason="Sell rally in downtrend"
                )
        else:
            # Range trading - buy low, sell high
            if range_pos < 0.2 and rsi_val < 35:
                # Near bottom of range + oversold
                entry = current_price
                stop_loss = entry - atr_val * 1.5
                take_profit = c['range_mid']  # Target middle of range
                take_profit_2 = c['high_20'] * 0.98

                # Check R:R
                risk = entry - stop_loss
                reward = take_profit - entry
                if reward > risk * 1.5:
                    signal = TradeSignal(
                        signal_type=SignalType.LONG,
                        strategy=StrategyType.GRID,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        take_profit_2=take_profit_2,
                        position_size_pct=0.4,
                        confidence=0.75,
                        reason=f"Grid buy at range bottom (RSI {rsi_val:.0f})"
                    )

            elif range_pos > 0.8 and rsi_val > 65:
                # Near top of range + overbought
                entry = current_price
                stop_loss = entry + atr_val * 1.5
                take_profit = c['range_mid']
                take_profit_2 = c['low_20'] * 1.02

                risk = stop_loss - entry
                reward = entry - take_profit
                if reward > risk * 1.5:
                    signal = TradeSignal(
                        signal_type=SignalType.SHORT,
                        strategy=StrategyType.GRID,
                        entry_price=entry,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        take_profit_2=take_profit_2,
                        position_size_pct=0.4,
                        confidence=0.75,
                        reason=f"Grid sell at range top (RSI {rsi_val:.0f})"
                    )

        if signal and not self._validate(signal):
            return None

        # Update last trade tracking
        if signal:
            self.last_trade_price = current_price
            self.last_trade_side = signal.signal_type

        return signal

    def _validate(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        if reward / risk < 1.3:
            return False

        stop_pct = risk / entry
        if stop_pct > 0.03:  # Max 3% stop
            return False

        if signal.confidence < 0.70:
            return False

        return True


class MomentumBreakoutStrategy:
    """
    Strong Momentum Breakout Strategy

    Only trades when there's clear, strong momentum
    Fewer signals but higher quality
    """

    def __init__(self):
        self.momentum_threshold = 2.0  # 2% move
        self.volume_threshold = 1.5  # 1.5x average volume

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['ema_10'] = ema(df['close'], 10)
        df['ema_30'] = ema(df['close'], 30)
        df['ema_50'] = ema(df['close'], 50)

        df['rsi'] = rsi(df['close'], 14)
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)

        # Momentum
        df['mom_1h'] = df['close'].pct_change(1) * 100
        df['mom_4h'] = df['close'].pct_change(4) * 100
        df['mom_8h'] = df['close'].pct_change(8) * 100

        # Volume
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1)

        # Breakout detection
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['breakout_up'] = df['close'] > df['high_20'].shift()
        df['breakout_down'] = df['close'] < df['low_20'].shift()

        return df

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        if len(df) < 100:
            return None

        df = self.calculate_indicators(df)
        c = df.iloc[-1]
        p1 = df.iloc[-2]

        if pd.isna(c['atr']) or c['atr'] == 0:
            return None

        signal = None

        # Strong bullish momentum breakout
        bullish_momentum = (
            c['mom_4h'] > self.momentum_threshold and
            c['mom_8h'] > 0 and
            c['vol_ratio'] > self.volume_threshold and
            c['ema_10'] > c['ema_30'] > c['ema_50'] and
            30 < c['rsi'] < 70  # Not overbought
        )

        if bullish_momentum:
            entry = c['close']
            atr_val = c['atr']

            stop_loss = entry - atr_val * 2
            take_profit = entry + atr_val * 3
            take_profit_2 = entry + atr_val * 5

            signal = TradeSignal(
                signal_type=SignalType.LONG,
                strategy=StrategyType.GRID,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=0.75,
                reason=f"Momentum breakout up ({c['mom_4h']:.1f}% in 4h)"
            )

        # Strong bearish momentum breakdown
        bearish_momentum = (
            c['mom_4h'] < -self.momentum_threshold and
            c['mom_8h'] < 0 and
            c['vol_ratio'] > self.volume_threshold and
            c['ema_10'] < c['ema_30'] < c['ema_50'] and
            30 < c['rsi'] < 70
        )

        if bearish_momentum:
            entry = c['close']
            atr_val = c['atr']

            stop_loss = entry + atr_val * 2
            take_profit = entry - atr_val * 3
            take_profit_2 = entry - atr_val * 5

            signal = TradeSignal(
                signal_type=SignalType.SHORT,
                strategy=StrategyType.GRID,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_2=take_profit_2,
                position_size_pct=0.5,
                confidence=0.75,
                reason=f"Momentum breakdown ({c['mom_4h']:.1f}% in 4h)"
            )

        if signal and not self._validate(signal):
            return None

        return signal

    def _validate(self, signal: TradeSignal) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        tp = signal.take_profit

        risk = abs(entry - sl)
        reward = abs(tp - entry)

        if risk == 0:
            return False

        if reward / risk < 1.3:
            return False

        stop_pct = risk / entry
        if stop_pct > 0.035:
            return False

        if signal.confidence < 0.70:
            return False

        return True


class CombinedStrategy:
    """Combines Grid and Momentum strategies"""

    def __init__(self):
        self.grid = GridStrategy()
        self.momentum = MomentumBreakoutStrategy()

    def analyze(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        signals = []

        grid_signal = self.grid.analyze(df)
        if grid_signal:
            signals.append(grid_signal)

        momentum_signal = self.momentum.analyze(df)
        if momentum_signal:
            signals.append(momentum_signal)

        if not signals:
            return None

        return max(signals, key=lambda s: s.confidence)


SimpleStrategy = CombinedStrategy
