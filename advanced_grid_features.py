#!/usr/bin/env python3
"""
ADVANCED GRID TRADING FEATURES
==============================
Enhanced capabilities for the grid trading system including:
- Machine Learning signal generation
- Multi-timeframe analysis
- Adaptive grid strategies
- Smart order placement
- Anti-liquidation system
- Martingale/Anti-Martingale modes
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class GridMode(Enum):
    """Grid trading mode"""
    NEUTRAL = "NEUTRAL"  # Equal buy/sell grids
    LONG_BIAS = "LONG_BIAS"  # More buy grids
    SHORT_BIAS = "SHORT_BIAS"  # More sell grids
    SCALP = "SCALP"  # Tight grids for quick profits
    SWING = "SWING"  # Wide grids for larger moves


@dataclass
class MLSignal:
    """Machine learning signal output"""
    direction: int  # 1 = long, -1 = short, 0 = neutral
    confidence: float  # 0 to 1
    predicted_volatility: float
    suggested_leverage: float
    suggested_grid_spacing: float


class FeatureEngineering:
    """Generate features for ML models and signal generation"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive feature set"""
        features = pd.DataFrame(index=df.index)

        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Multiple timeframe returns
        for period in [1, 4, 12, 24, 48, 168]:  # 1h, 4h, 12h, 1d, 2d, 1w
            features[f'return_{period}h'] = df['close'].pct_change(period)

        # Volatility features
        features['volatility_24h'] = features['returns'].rolling(24).std()
        features['volatility_168h'] = features['returns'].rolling(168).std()
        features['vol_ratio'] = features['volatility_24h'] / features['volatility_168h']

        # Trend features
        for period in [10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'price_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']

        # Momentum features
        features['rsi_14'] = FeatureEngineering._calculate_rsi(df['close'], 14)
        features['rsi_7'] = FeatureEngineering._calculate_rsi(df['close'], 7)

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']

        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        features['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma20 + 2 * std20
        features['bb_lower'] = sma20 - 2 * std20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma20
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_pct'] = features['atr_14'] / df['close']

        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_sma'] = df['volume'].rolling(20).mean()
            features['volume_ratio'] = df['volume'] / features['volume_sma']
            features['volume_change'] = df['volume'].pct_change()
        else:
            features['volume_ratio'] = 1.0
            features['volume_change'] = 0.0

        # Price action features
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        features['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        features['candle_direction'] = np.where(df['close'] > df['open'], 1, -1)

        # Support/Resistance levels
        features['resistance_24h'] = df['high'].rolling(24).max()
        features['support_24h'] = df['low'].rolling(24).min()
        features['range_24h'] = features['resistance_24h'] - features['support_24h']
        features['price_in_range'] = (df['close'] - features['support_24h']) / features['range_24h']

        # Mean reversion features
        features['z_score_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        features['z_score_50'] = (df['close'] - df['close'].rolling(50).mean()) / df['close'].rolling(50).std()

        return features

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class MarketRegimeDetector:
    """Detect current market regime for adaptive strategy"""

    def __init__(self):
        self.lookback_periods = {
            'short': 24,   # 1 day
            'medium': 168,  # 1 week
            'long': 720    # 1 month
        }

    def detect_regime(self, df: pd.DataFrame, features: pd.DataFrame) -> MarketRegime:
        """Detect the current market regime"""
        if len(features) < 200:
            return MarketRegime.RANGING

        current = features.iloc[-1]

        # Volatility assessment
        vol_percentile = self._calculate_volatility_percentile(features)

        if vol_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        elif vol_percentile < 20:
            return MarketRegime.LOW_VOLATILITY

        # Trend assessment
        trend_strength = self._calculate_trend_strength(df, features)

        if trend_strength > 0.6:
            return MarketRegime.TRENDING_UP
        elif trend_strength < -0.6:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def _calculate_volatility_percentile(self, features: pd.DataFrame) -> float:
        """Calculate where current volatility sits in historical distribution"""
        current_vol = features['volatility_24h'].iloc[-1]
        historical_vol = features['volatility_24h'].dropna()

        if len(historical_vol) == 0:
            return 50.0

        percentile = (historical_vol < current_vol).sum() / len(historical_vol) * 100
        return percentile

    def _calculate_trend_strength(self, df: pd.DataFrame, features: pd.DataFrame) -> float:
        """Calculate trend strength from -1 (strong down) to 1 (strong up)"""
        scores = []

        # Price vs SMAs
        for period in [20, 50, 100, 200]:
            col = f'price_sma_{period}_ratio'
            if col in features.columns:
                ratio = features[col].iloc[-1]
                if pd.notna(ratio):
                    scores.append(np.clip((ratio - 1) * 10, -1, 1))

        # MACD
        if 'macd_hist' in features.columns:
            macd_hist = features['macd_hist'].iloc[-1]
            if pd.notna(macd_hist):
                scores.append(np.clip(macd_hist / df['close'].iloc[-1] * 100, -1, 1))

        # RSI
        if 'rsi_14' in features.columns:
            rsi = features['rsi_14'].iloc[-1]
            if pd.notna(rsi):
                scores.append((rsi - 50) / 50)

        if not scores:
            return 0.0

        return np.mean(scores)


class SignalGenerator:
    """Generate trading signals using multiple methods"""

    def __init__(self):
        self.regime_detector = MarketRegimeDetector()

    def generate_signal(self, df: pd.DataFrame, features: pd.DataFrame) -> MLSignal:
        """Generate comprehensive trading signal"""
        regime = self.regime_detector.detect_regime(df, features)
        current = features.iloc[-1]

        # Calculate direction and confidence
        direction, confidence = self._calculate_direction(features, regime)

        # Calculate suggested parameters
        predicted_vol = self._predict_volatility(features)
        suggested_leverage = self._calculate_suggested_leverage(confidence, predicted_vol, regime)
        suggested_spacing = self._calculate_suggested_spacing(predicted_vol, regime)

        return MLSignal(
            direction=direction,
            confidence=confidence,
            predicted_volatility=predicted_vol,
            suggested_leverage=suggested_leverage,
            suggested_grid_spacing=suggested_spacing
        )

    def _calculate_direction(self, features: pd.DataFrame, regime: MarketRegime) -> Tuple[int, float]:
        """Calculate trade direction and confidence"""
        signals = []

        current = features.iloc[-1]

        # RSI signal
        rsi = current.get('rsi_14', 50)
        if pd.notna(rsi):
            if rsi < 30:
                signals.append((1, 0.7))  # Oversold - bullish
            elif rsi > 70:
                signals.append((-1, 0.7))  # Overbought - bearish
            else:
                signals.append((0, 0.3))

        # BB position signal
        bb_pos = current.get('bb_position', 0.5)
        if pd.notna(bb_pos):
            if bb_pos < 0.1:
                signals.append((1, 0.6))  # Near lower band - bullish
            elif bb_pos > 0.9:
                signals.append((-1, 0.6))  # Near upper band - bearish
            else:
                signals.append((0, 0.2))

        # Z-score signal (mean reversion)
        z_score = current.get('z_score_20', 0)
        if pd.notna(z_score):
            if z_score < -2:
                signals.append((1, 0.8))  # Very oversold
            elif z_score > 2:
                signals.append((-1, 0.8))  # Very overbought
            elif z_score < -1:
                signals.append((1, 0.5))
            elif z_score > 1:
                signals.append((-1, 0.5))
            else:
                signals.append((0, 0.3))

        # MACD signal
        macd_hist = current.get('macd_hist', 0)
        if pd.notna(macd_hist):
            # Check for crossover
            prev_macd_hist = features['macd_hist'].iloc[-2] if len(features) > 1 else 0
            if macd_hist > 0 and prev_macd_hist <= 0:
                signals.append((1, 0.6))  # Bullish crossover
            elif macd_hist < 0 and prev_macd_hist >= 0:
                signals.append((-1, 0.6))  # Bearish crossover
            else:
                signals.append((0, 0.2))

        # Stochastic signal
        stoch_k = current.get('stoch_k', 50)
        stoch_d = current.get('stoch_d', 50)
        if pd.notna(stoch_k) and pd.notna(stoch_d):
            if stoch_k < 20 and stoch_d < 20:
                signals.append((1, 0.5))  # Oversold
            elif stoch_k > 80 and stoch_d > 80:
                signals.append((-1, 0.5))  # Overbought
            else:
                signals.append((0, 0.2))

        # Calculate weighted average
        if not signals:
            return 0, 0.5

        total_weight = sum(s[1] for s in signals)
        weighted_direction = sum(s[0] * s[1] for s in signals) / total_weight
        avg_confidence = total_weight / len(signals)

        # Final direction
        if weighted_direction > 0.3:
            direction = 1
        elif weighted_direction < -0.3:
            direction = -1
        else:
            direction = 0

        confidence = min(1.0, abs(weighted_direction) * avg_confidence)

        return direction, confidence

    def _predict_volatility(self, features: pd.DataFrame) -> float:
        """Predict near-term volatility"""
        if 'volatility_24h' not in features.columns:
            return 0.02  # Default 2%

        recent_vol = features['volatility_24h'].iloc[-24:].mean()
        current_vol = features['volatility_24h'].iloc[-1]

        # Simple prediction: weighted average of current and recent
        predicted = 0.6 * current_vol + 0.4 * recent_vol

        if pd.isna(predicted):
            return 0.02

        return predicted

    def _calculate_suggested_leverage(self, confidence: float, volatility: float,
                                     regime: MarketRegime) -> float:
        """Calculate suggested leverage based on conditions"""
        base_leverage = 20

        # Confidence adjustment
        confidence_mult = 0.5 + confidence * 0.5  # 0.5 to 1.0

        # Volatility adjustment (lower leverage for high volatility)
        if volatility > 0.05:  # >5% volatility
            vol_mult = 0.4
        elif volatility > 0.03:
            vol_mult = 0.6
        elif volatility > 0.02:
            vol_mult = 0.8
        else:
            vol_mult = 1.0

        # Regime adjustment
        regime_mult = {
            MarketRegime.HIGH_VOLATILITY: 0.5,
            MarketRegime.TRENDING_UP: 1.1,
            MarketRegime.TRENDING_DOWN: 1.1,
            MarketRegime.RANGING: 1.2,  # Grid works best in ranging
            MarketRegime.LOW_VOLATILITY: 0.8,
        }.get(regime, 1.0)

        leverage = base_leverage * confidence_mult * vol_mult * regime_mult

        return np.clip(leverage, 5, 75)

    def _calculate_suggested_spacing(self, volatility: float, regime: MarketRegime) -> float:
        """Calculate suggested grid spacing"""
        base_spacing = 0.3  # 0.3%

        # Volatility adjustment
        vol_mult = 1 + (volatility * 10)  # Higher spacing for higher volatility

        # Regime adjustment
        regime_mult = {
            MarketRegime.HIGH_VOLATILITY: 1.5,
            MarketRegime.TRENDING_UP: 1.2,
            MarketRegime.TRENDING_DOWN: 1.2,
            MarketRegime.RANGING: 0.8,  # Tighter grids in ranging
            MarketRegime.LOW_VOLATILITY: 0.7,
        }.get(regime, 1.0)

        spacing = base_spacing * vol_mult * regime_mult

        return np.clip(spacing, 0.1, 2.0)


class AdaptiveGridStrategy:
    """Adaptive grid strategy that adjusts to market conditions"""

    def __init__(self, initial_capital: float = 4.0):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.signal_generator = SignalGenerator()
        self.regime_detector = MarketRegimeDetector()

    def get_optimal_mode(self, regime: MarketRegime, signal: MLSignal) -> GridMode:
        """Determine optimal grid mode based on conditions"""
        if regime == MarketRegime.RANGING:
            if signal.confidence < 0.3:
                return GridMode.NEUTRAL
            elif signal.direction > 0:
                return GridMode.LONG_BIAS
            else:
                return GridMode.SHORT_BIAS

        elif regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # In trends, use wider grids (swing mode)
            return GridMode.SWING

        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Use neutral mode with wider spacing
            return GridMode.NEUTRAL

        elif regime == MarketRegime.LOW_VOLATILITY:
            # Use scalp mode for quick profits
            return GridMode.SCALP

        return GridMode.NEUTRAL

    def calculate_grid_parameters(self, mode: GridMode, signal: MLSignal,
                                 current_price: float) -> Dict:
        """Calculate grid parameters based on mode and signal"""
        params = {
            'num_buy_grids': 0,
            'num_sell_grids': 0,
            'grid_spacing': signal.suggested_grid_spacing,
            'leverage': signal.suggested_leverage,
            'position_size_pct': 2.5,
        }

        if mode == GridMode.NEUTRAL:
            params['num_buy_grids'] = 15
            params['num_sell_grids'] = 15

        elif mode == GridMode.LONG_BIAS:
            params['num_buy_grids'] = 20
            params['num_sell_grids'] = 10

        elif mode == GridMode.SHORT_BIAS:
            params['num_buy_grids'] = 10
            params['num_sell_grids'] = 20

        elif mode == GridMode.SCALP:
            params['num_buy_grids'] = 10
            params['num_sell_grids'] = 10
            params['grid_spacing'] = signal.suggested_grid_spacing * 0.5
            params['position_size_pct'] = 3.5  # Higher size for quick scalps

        elif mode == GridMode.SWING:
            params['num_buy_grids'] = 8
            params['num_sell_grids'] = 8
            params['grid_spacing'] = signal.suggested_grid_spacing * 1.5
            params['position_size_pct'] = 2.0

        return params


class AntiLiquidationSystem:
    """System to prevent liquidation through proactive risk management"""

    def __init__(self, max_leverage: float = 75):
        self.max_leverage = max_leverage
        self.warning_levels = [50, 70, 85, 95]  # % of margin used warnings

    def calculate_liquidation_price(self, entry_price: float, leverage: float,
                                   is_long: bool, maintenance_margin: float = 0.004) -> float:
        """Calculate liquidation price for a position"""
        if is_long:
            # Long liquidation = entry * (1 - 1/leverage + maintenance_margin)
            liq_price = entry_price * (1 - 1/leverage + maintenance_margin)
        else:
            # Short liquidation = entry * (1 + 1/leverage - maintenance_margin)
            liq_price = entry_price * (1 + 1/leverage - maintenance_margin)

        return liq_price

    def calculate_safe_leverage(self, volatility: float, max_expected_move: float = None) -> float:
        """Calculate safe leverage based on expected price movements"""
        if max_expected_move is None:
            # Estimate max move as 3x recent volatility
            max_expected_move = volatility * 3

        # Want liquidation price to be beyond expected move
        # Safe leverage = 1 / max_expected_move
        safe_leverage = 0.8 / max_expected_move if max_expected_move > 0 else 20

        return np.clip(safe_leverage, 5, self.max_leverage)

    def calculate_margin_usage(self, positions: List, capital: float,
                              current_price: float) -> float:
        """Calculate current margin usage percentage"""
        total_margin = 0
        for pos in positions:
            position_value = pos.size * current_price
            margin_required = position_value / pos.leverage
            total_margin += margin_required

        return (total_margin / capital * 100) if capital > 0 else 100

    def should_reduce_exposure(self, margin_usage: float, unrealized_pnl: float,
                              capital: float) -> Tuple[bool, str]:
        """Determine if exposure should be reduced"""
        # Check margin usage
        if margin_usage >= 95:
            return True, "CRITICAL: Margin usage at 95%+ - Reduce positions immediately"
        elif margin_usage >= 85:
            return True, "WARNING: Margin usage at 85%+ - Consider reducing positions"

        # Check unrealized loss
        loss_pct = abs(min(0, unrealized_pnl)) / capital * 100 if capital > 0 else 0
        if loss_pct >= 10:
            return True, f"WARNING: Unrealized loss at {loss_pct:.1f}% - Consider hedging"

        return False, "OK"


class MartingaleManager:
    """Manages Martingale and Anti-Martingale position sizing"""

    def __init__(self, mode: str = 'anti'):  # 'martingale', 'anti', or 'fixed'
        self.mode = mode
        self.multiplier = 1.5  # Size multiplier
        self.max_multiplier = 4.0  # Maximum multiplier from base
        self.consecutive_losses = 0
        self.consecutive_wins = 0

    def get_position_multiplier(self) -> float:
        """Get position size multiplier based on mode"""
        if self.mode == 'fixed':
            return 1.0

        elif self.mode == 'martingale':
            # Increase size after losses (dangerous but can recover quickly)
            mult = self.multiplier ** self.consecutive_losses
            return min(mult, self.max_multiplier)

        elif self.mode == 'anti':
            # Increase size after wins, decrease after losses (safer)
            if self.consecutive_wins > 0:
                mult = self.multiplier ** min(self.consecutive_wins, 2)
                return min(mult, self.max_multiplier)
            elif self.consecutive_losses > 0:
                mult = 1 / (self.multiplier ** min(self.consecutive_losses, 2))
                return max(mult, 0.25)
            return 1.0

        return 1.0

    def record_result(self, is_win: bool):
        """Record trade result"""
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0


class SmartOrderPlacer:
    """Intelligent order placement with slippage and fee optimization"""

    def __init__(self, maker_fee: float = 0.0002, taker_fee: float = 0.0004):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def calculate_breakeven_move(self, leverage: float) -> float:
        """Calculate minimum price move needed to break even after fees"""
        # Round trip fees = 2 * taker_fee (worst case)
        round_trip_fee = 2 * self.taker_fee
        # Breakeven move = fees / leverage
        return round_trip_fee / leverage

    def optimize_grid_spacing(self, base_spacing: float, leverage: float) -> float:
        """Ensure grid spacing is profitable after fees"""
        breakeven = self.calculate_breakeven_move(leverage)
        min_spacing = breakeven * 2  # At least 2x breakeven for meaningful profit

        return max(base_spacing, min_spacing * 100)  # Convert to percentage

    def calculate_expected_profit(self, spacing_pct: float, leverage: float,
                                 position_size: float, price: float) -> float:
        """Calculate expected profit per successful grid trade"""
        price_move = price * spacing_pct / 100
        gross_profit = price_move * position_size * leverage

        # Fees
        entry_fee = price * position_size * self.taker_fee
        exit_fee = (price + price_move) * position_size * self.taker_fee

        return gross_profit - entry_fee - exit_fee


class CapitalCompoundingEngine:
    """Advanced capital compounding strategies"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.compound_history = []
        self.withdrawal_history = []

    def calculate_compound_schedule(self, target_growth: float, period_days: int) -> Dict:
        """Calculate required daily return for target growth"""
        daily_return = (target_growth ** (1/period_days)) - 1

        return {
            'target_growth': target_growth,
            'period_days': period_days,
            'required_daily_return_pct': daily_return * 100,
            'required_per_trade_return_pct': daily_return * 100 / 10,  # Assuming ~10 trades/day
        }

    def should_compound(self, current_equity: float, last_compound_equity: float,
                       threshold_pct: float = 5.0) -> bool:
        """Determine if profits should be compounded"""
        growth = (current_equity - last_compound_equity) / last_compound_equity * 100
        return growth >= threshold_pct

    def calculate_withdrawal_safe_amount(self, current_equity: float,
                                        min_operating_capital: float = 4.0) -> float:
        """Calculate safe amount to withdraw while maintaining trading capacity"""
        # Keep at least min_operating_capital or 50% of equity, whichever is higher
        min_keep = max(min_operating_capital, current_equity * 0.5)
        safe_withdrawal = max(0, current_equity - min_keep)

        return safe_withdrawal

    def project_growth(self, daily_return_pct: float, days: int,
                      starting_capital: float = None) -> pd.DataFrame:
        """Project capital growth over time"""
        if starting_capital is None:
            starting_capital = self.capital

        data = []
        capital = starting_capital

        for day in range(days + 1):
            data.append({
                'day': day,
                'capital': capital,
                'growth_pct': (capital - starting_capital) / starting_capital * 100
            })
            capital *= (1 + daily_return_pct / 100)

        return pd.DataFrame(data)


def run_advanced_analysis(df: pd.DataFrame, symbol: str = "UNKNOWN") -> Dict:
    """Run comprehensive advanced analysis"""

    print(f"\n{'='*60}")
    print(f"  ADVANCED ANALYSIS: {symbol}")
    print('='*60)

    # Generate features
    feature_eng = FeatureEngineering()
    features = feature_eng.calculate_features(df)

    # Detect market regime
    regime_detector = MarketRegimeDetector()
    regime = regime_detector.detect_regime(df, features)
    print(f"\n📊 Current Market Regime: {regime.value}")

    # Generate signals
    signal_gen = SignalGenerator()
    signal = signal_gen.generate_signal(df, features)

    print(f"\n📈 ML Signal Analysis:")
    print(f"  Direction: {'LONG' if signal.direction > 0 else 'SHORT' if signal.direction < 0 else 'NEUTRAL'}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Predicted Volatility: {signal.predicted_volatility:.2%}")
    print(f"  Suggested Leverage: {signal.suggested_leverage:.1f}x")
    print(f"  Suggested Grid Spacing: {signal.suggested_grid_spacing:.2f}%")

    # Get adaptive strategy
    adaptive = AdaptiveGridStrategy(initial_capital=4.0)
    optimal_mode = adaptive.get_optimal_mode(regime, signal)
    grid_params = adaptive.calculate_grid_parameters(optimal_mode, signal, df['close'].iloc[-1])

    print(f"\n⚙️ Recommended Grid Configuration:")
    print(f"  Mode: {optimal_mode.value}")
    print(f"  Buy Grids: {grid_params['num_buy_grids']}")
    print(f"  Sell Grids: {grid_params['num_sell_grids']}")
    print(f"  Grid Spacing: {grid_params['grid_spacing']:.2f}%")
    print(f"  Leverage: {grid_params['leverage']:.1f}x")

    # Anti-liquidation analysis
    anti_liq = AntiLiquidationSystem()
    safe_leverage = anti_liq.calculate_safe_leverage(signal.predicted_volatility)

    print(f"\n🛡️ Risk Analysis:")
    print(f"  Safe Leverage (based on volatility): {safe_leverage:.1f}x")
    print(f"  Liquidation buffer: {100/grid_params['leverage']:.2f}% from entry")

    # Capital projection
    compounding = CapitalCompoundingEngine(4.0)
    projection = compounding.project_growth(daily_return_pct=2.0, days=30)

    print(f"\n💰 30-Day Capital Projection (at 2% daily):")
    print(f"  Starting: ${4.00:.2f}")
    print(f"  Day 7:    ${projection[projection['day']==7]['capital'].values[0]:.2f}")
    print(f"  Day 14:   ${projection[projection['day']==14]['capital'].values[0]:.2f}")
    print(f"  Day 30:   ${projection[projection['day']==30]['capital'].values[0]:.2f}")

    return {
        'regime': regime,
        'signal': signal,
        'optimal_mode': optimal_mode,
        'grid_params': grid_params,
        'safe_leverage': safe_leverage,
        'projection': projection
    }


if __name__ == "__main__":
    # Test with sample data
    print("Advanced Grid Features Module Loaded")
    print("Use run_advanced_analysis(df, symbol) with your price data")
