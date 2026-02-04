#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED GRID TRADING SYSTEM
===================================
Maximum profit extraction with minimal drawdown for small accounts ($4+)

Key Features:
- Aggressive compounding with profit protection
- Multi-layer grid system for maximum trade frequency
- Adaptive leverage scaling based on account growth
- Volatility regime switching
- Smart position sizing with Kelly Criterion
- Dynamic take-profit and stop-loss
- Trailing profit capture
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PositionType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class UltraGridConfig:
    """Ultra-optimized configuration for maximum returns"""
    # Starting capital
    initial_capital: float = 4.0

    # Grid structure - Multi-layer design
    inner_grids: int = 8  # Tight grids for scalping
    outer_grids: int = 6  # Wider grids for swings
    inner_spacing_pct: float = 0.15  # 0.15% for inner
    outer_spacing_pct: float = 0.4  # 0.4% for outer

    # Leverage - Aggressive but controlled
    min_leverage: float = 10
    base_leverage: float = 30
    max_leverage: float = 75
    leverage_scale_factor: float = 1.5  # Scale up as account grows

    # Position sizing
    inner_position_pct: float = 1.5  # % of capital per inner grid
    outer_position_pct: float = 2.0  # % of capital per outer grid
    max_total_exposure: float = 85  # Max % of capital exposed

    # Risk management
    max_drawdown: float = 12.0  # Hard stop at 12% drawdown
    soft_drawdown: float = 8.0  # Reduce exposure at 8%
    position_stop_loss: float = 5.0  # Individual position stop
    trailing_stop_activation: float = 2.0  # Activate trailing after 2% profit
    trailing_stop_distance: float = 1.0  # Trail by 1%

    # Take profit - Dynamic
    base_take_profit: float = 0.3  # Base TP at 0.3%
    extended_take_profit: float = 0.6  # Extended TP at 0.6%
    use_partial_tp: bool = True  # Take partial profits

    # Fees
    maker_fee: float = 0.0002
    taker_fee: float = 0.0004

    # Compounding
    compound_threshold: float = 3.0  # Compound after 3% gain
    profit_lock_threshold: float = 10.0  # Lock 50% profits after 10% gain

    # Market filters
    min_volatility: float = 0.005  # Don't trade if vol < 0.5%
    max_volatility: float = 0.08  # Reduce size if vol > 8%
    trend_filter_strength: float = 0.3  # Bias grids with trend

    # Grid management
    rebalance_threshold: float = 3.0  # Rebalance if price moves 3%
    max_one_sided_fills: int = 5  # Max fills on one side before hedge


@dataclass
class GridLevel:
    """Represents a single grid level"""
    price: float
    side: PositionType
    size: float
    layer: str  # 'inner' or 'outer'
    is_filled: bool = False
    fill_time: Optional[pd.Timestamp] = None
    fill_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trailing_stop: Optional[float] = None
    unrealized_pnl: float = 0.0


@dataclass
class ActivePosition:
    """Active trading position"""
    entry_price: float
    side: PositionType
    size: float
    leverage: float
    entry_time: pd.Timestamp
    take_profit: float
    stop_loss: float
    trailing_activated: bool = False
    trailing_stop: Optional[float] = None
    highest_profit: float = 0.0
    layer: str = 'inner'


class KellyCriterion:
    """Kelly Criterion for optimal position sizing"""

    @staticmethod
    def calculate_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal Kelly fraction"""
        if avg_loss == 0 or win_rate == 0:
            return 0.1  # Default conservative

        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Use half-Kelly for safety
        half_kelly = kelly / 2

        return np.clip(half_kelly, 0.05, 0.25)  # 5% to 25% of capital


class VolatilityAnalyzer:
    """Advanced volatility analysis"""

    @staticmethod
    def calculate_regime_volatility(df: pd.DataFrame) -> Dict:
        """Analyze volatility regime"""
        returns = df['close'].pct_change()

        current_vol = returns.iloc[-24:].std() if len(returns) >= 24 else returns.std()
        short_vol = returns.iloc[-6:].std() if len(returns) >= 6 else returns.std()
        long_vol = returns.iloc[-168:].std() if len(returns) >= 168 else returns.std()

        # Volatility trend
        vol_expanding = short_vol > current_vol > long_vol
        vol_contracting = short_vol < current_vol < long_vol

        # ATR-based
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.iloc[-14:].mean()
        atr_pct = atr / close.iloc[-1]

        return {
            'current_vol': current_vol,
            'short_vol': short_vol,
            'long_vol': long_vol,
            'atr': atr,
            'atr_pct': atr_pct,
            'vol_expanding': vol_expanding,
            'vol_contracting': vol_contracting,
            'regime': 'high' if current_vol > 0.03 else 'low' if current_vol < 0.01 else 'normal'
        }


class TrendAnalyzer:
    """Trend detection for grid bias"""

    @staticmethod
    def detect_trend(df: pd.DataFrame) -> Dict:
        """Detect market trend"""
        close = df['close']

        # Multiple EMAs
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_55 = close.ewm(span=55).mean()

        current_price = close.iloc[-1]

        # Trend scores
        trend_score = 0
        if current_price > ema_8.iloc[-1]:
            trend_score += 1
        if current_price > ema_21.iloc[-1]:
            trend_score += 1
        if current_price > ema_55.iloc[-1]:
            trend_score += 1
        if ema_8.iloc[-1] > ema_21.iloc[-1]:
            trend_score += 1
        if ema_21.iloc[-1] > ema_55.iloc[-1]:
            trend_score += 1

        # Normalize to -1 to 1
        trend_strength = (trend_score - 2.5) / 2.5

        # Price momentum
        momentum_24h = (current_price - close.iloc[-24]) / close.iloc[-24] if len(close) >= 24 else 0
        momentum_168h = (current_price - close.iloc[-168]) / close.iloc[-168] if len(close) >= 168 else 0

        return {
            'trend_strength': trend_strength,
            'momentum_24h': momentum_24h,
            'momentum_168h': momentum_168h,
            'direction': 'up' if trend_strength > 0.2 else 'down' if trend_strength < -0.2 else 'neutral',
            'ema_8': ema_8.iloc[-1],
            'ema_21': ema_21.iloc[-1],
            'ema_55': ema_55.iloc[-1]
        }


class UltraGridTrader:
    """Ultra-optimized grid trading engine"""

    def __init__(self, config: UltraGridConfig):
        self.config = config
        self.capital = config.initial_capital
        self.initial_capital = config.initial_capital
        self.locked_profit = 0.0  # Protected profits

        # Grid state
        self.grid_levels: List[GridLevel] = []
        self.active_positions: List[ActivePosition] = []

        # Tracking
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.trade_log: List[Dict] = []
        self.timestamps: List[pd.Timestamp] = []

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Kelly tracking
        self.recent_wins: List[float] = []
        self.recent_losses: List[float] = []

        # Analyzers
        self.vol_analyzer = VolatilityAnalyzer()
        self.trend_analyzer = TrendAnalyzer()

    def calculate_dynamic_leverage(self, vol_regime: Dict, trend: Dict) -> float:
        """Calculate optimal leverage based on conditions"""
        base = self.config.base_leverage

        # Account size scaling - more aggressive as account grows
        account_growth = self.capital / self.config.initial_capital
        if account_growth > 5:
            size_mult = 0.8  # Reduce leverage as account grows large
        elif account_growth > 2:
            size_mult = 1.0
        else:
            size_mult = self.config.leverage_scale_factor  # More aggressive when small

        # Volatility adjustment
        if vol_regime['regime'] == 'high':
            vol_mult = 0.5
        elif vol_regime['regime'] == 'low':
            vol_mult = 1.2
        else:
            vol_mult = 1.0

        # Trend adjustment - higher leverage in ranging (grid sweet spot)
        if abs(trend['trend_strength']) < 0.2:  # Ranging market
            trend_mult = 1.2
        else:
            trend_mult = 0.9

        # Drawdown adjustment
        if self.current_drawdown > self.config.soft_drawdown:
            dd_mult = 0.5
        elif self.current_drawdown > self.config.soft_drawdown / 2:
            dd_mult = 0.7
        else:
            dd_mult = 1.0

        leverage = base * size_mult * vol_mult * trend_mult * dd_mult
        return np.clip(leverage, self.config.min_leverage, self.config.max_leverage)

    def calculate_position_size(self, layer: str, price: float,
                               leverage: float, kelly_fraction: float) -> float:
        """Calculate position size using Kelly and config"""
        if layer == 'inner':
            base_pct = self.config.inner_position_pct
        else:
            base_pct = self.config.outer_position_pct

        # Apply Kelly adjustment
        kelly_adjusted = base_pct * (kelly_fraction / 0.1)  # Normalize around 10%
        kelly_adjusted = np.clip(kelly_adjusted, base_pct * 0.5, base_pct * 1.5)

        # Calculate size
        position_value = (self.capital * kelly_adjusted / 100) * leverage
        size = position_value / price

        return max(0.0001, size)

    def setup_grid(self, center_price: float, vol_regime: Dict,
                  trend: Dict, leverage: float, timestamp: pd.Timestamp):
        """Setup multi-layer grid"""
        self.grid_levels = []

        # Calculate Kelly fraction from recent performance
        win_rate = len(self.recent_wins) / max(1, len(self.recent_wins) + len(self.recent_losses))
        avg_win = np.mean(self.recent_wins) if self.recent_wins else 0.003
        avg_loss = np.mean(self.recent_losses) if self.recent_losses else 0.002
        kelly = KellyCriterion.calculate_kelly_fraction(win_rate, avg_win, avg_loss)

        # Adjust spacing based on volatility
        vol_mult = 1.0 + vol_regime['atr_pct'] * 5
        inner_spacing = self.config.inner_spacing_pct * vol_mult
        outer_spacing = self.config.outer_spacing_pct * vol_mult

        # Trend bias for grid distribution
        trend_bias = trend['trend_strength'] * self.config.trend_filter_strength

        # Calculate grid counts with trend bias
        inner_buy_grids = int(self.config.inner_grids * (1 + trend_bias))
        inner_sell_grids = int(self.config.inner_grids * (1 - trend_bias))
        outer_buy_grids = int(self.config.outer_grids * (1 + trend_bias))
        outer_sell_grids = int(self.config.outer_grids * (1 - trend_bias))

        # Ensure minimum grids
        inner_buy_grids = max(3, inner_buy_grids)
        inner_sell_grids = max(3, inner_sell_grids)
        outer_buy_grids = max(2, outer_buy_grids)
        outer_sell_grids = max(2, outer_sell_grids)

        # Create inner grid levels (tight spacing for scalping)
        for i in range(1, inner_buy_grids + 1):
            price = center_price * (1 - inner_spacing * i / 100)
            size = self.calculate_position_size('inner', price, leverage, kelly)
            tp = price * (1 + self.config.base_take_profit / 100)
            sl = price * (1 - self.config.position_stop_loss / 100)

            self.grid_levels.append(GridLevel(
                price=price,
                side=PositionType.LONG,
                size=size,
                layer='inner',
                take_profit=tp,
                stop_loss=sl
            ))

        for i in range(1, inner_sell_grids + 1):
            price = center_price * (1 + inner_spacing * i / 100)
            size = self.calculate_position_size('inner', price, leverage, kelly)
            tp = price * (1 - self.config.base_take_profit / 100)
            sl = price * (1 + self.config.position_stop_loss / 100)

            self.grid_levels.append(GridLevel(
                price=price,
                side=PositionType.SHORT,
                size=size,
                layer='inner',
                take_profit=tp,
                stop_loss=sl
            ))

        # Create outer grid levels (wider spacing for swing trades)
        for i in range(1, outer_buy_grids + 1):
            price = center_price * (1 - (inner_spacing * self.config.inner_grids + outer_spacing * i) / 100)
            size = self.calculate_position_size('outer', price, leverage, kelly)
            tp = price * (1 + self.config.extended_take_profit / 100)
            sl = price * (1 - self.config.position_stop_loss * 1.5 / 100)

            self.grid_levels.append(GridLevel(
                price=price,
                side=PositionType.LONG,
                size=size,
                layer='outer',
                take_profit=tp,
                stop_loss=sl
            ))

        for i in range(1, outer_sell_grids + 1):
            price = center_price * (1 + (inner_spacing * self.config.inner_grids + outer_spacing * i) / 100)
            size = self.calculate_position_size('outer', price, leverage, kelly)
            tp = price * (1 - self.config.extended_take_profit / 100)
            sl = price * (1 + self.config.position_stop_loss * 1.5 / 100)

            self.grid_levels.append(GridLevel(
                price=price,
                side=PositionType.SHORT,
                size=size,
                layer='outer',
                take_profit=tp,
                stop_loss=sl
            ))

        self.grid_center = center_price

    def check_grid_fills(self, high: float, low: float, timestamp: pd.Timestamp,
                        leverage: float):
        """Check for grid level fills"""
        for level in self.grid_levels:
            if level.is_filled:
                continue

            filled = False
            fill_price = level.price

            if level.side == PositionType.LONG and low <= level.price:
                filled = True
            elif level.side == PositionType.SHORT and high >= level.price:
                filled = True

            if filled:
                level.is_filled = True
                level.fill_time = timestamp
                level.fill_price = fill_price

                # Create active position
                position = ActivePosition(
                    entry_price=fill_price,
                    side=level.side,
                    size=level.size,
                    leverage=leverage,
                    entry_time=timestamp,
                    take_profit=level.take_profit,
                    stop_loss=level.stop_loss,
                    layer=level.layer
                )
                self.active_positions.append(position)

                # Record fee
                fee = fill_price * level.size * self.config.taker_fee
                self.total_fees += fee
                self.capital -= fee

    def update_positions(self, high: float, low: float, close: float,
                        timestamp: pd.Timestamp):
        """Update active positions, check exits"""
        positions_to_close = []

        for i, pos in enumerate(self.active_positions):
            exit_triggered = False
            exit_price = close
            exit_reason = ""

            # Calculate current unrealized PnL
            if pos.side == PositionType.LONG:
                current_profit_pct = (close - pos.entry_price) / pos.entry_price * 100
                unrealized = (close - pos.entry_price) * pos.size * pos.leverage
            else:
                current_profit_pct = (pos.entry_price - close) / pos.entry_price * 100
                unrealized = (pos.entry_price - close) * pos.size * pos.leverage

            # Track highest profit for trailing stop
            if current_profit_pct > pos.highest_profit:
                pos.highest_profit = current_profit_pct

            # Activate trailing stop if threshold reached
            if not pos.trailing_activated and pos.highest_profit >= self.config.trailing_stop_activation:
                pos.trailing_activated = True
                if pos.side == PositionType.LONG:
                    pos.trailing_stop = close * (1 - self.config.trailing_stop_distance / 100)
                else:
                    pos.trailing_stop = close * (1 + self.config.trailing_stop_distance / 100)

            # Update trailing stop
            if pos.trailing_activated:
                if pos.side == PositionType.LONG:
                    new_trail = close * (1 - self.config.trailing_stop_distance / 100)
                    if new_trail > pos.trailing_stop:
                        pos.trailing_stop = new_trail
                else:
                    new_trail = close * (1 + self.config.trailing_stop_distance / 100)
                    if new_trail < pos.trailing_stop:
                        pos.trailing_stop = new_trail

            # Check exit conditions
            if pos.side == PositionType.LONG:
                # Take profit
                if high >= pos.take_profit:
                    exit_triggered = True
                    exit_price = pos.take_profit
                    exit_reason = "TP"
                # Stop loss
                elif low <= pos.stop_loss:
                    exit_triggered = True
                    exit_price = pos.stop_loss
                    exit_reason = "SL"
                # Trailing stop
                elif pos.trailing_activated and low <= pos.trailing_stop:
                    exit_triggered = True
                    exit_price = pos.trailing_stop
                    exit_reason = "TRAIL"

            else:  # SHORT
                # Take profit
                if low <= pos.take_profit:
                    exit_triggered = True
                    exit_price = pos.take_profit
                    exit_reason = "TP"
                # Stop loss
                elif high >= pos.stop_loss:
                    exit_triggered = True
                    exit_price = pos.stop_loss
                    exit_reason = "SL"
                # Trailing stop
                elif pos.trailing_activated and high >= pos.trailing_stop:
                    exit_triggered = True
                    exit_price = pos.trailing_stop
                    exit_reason = "TRAIL"

            if exit_triggered:
                # Calculate final PnL
                if pos.side == PositionType.LONG:
                    pnl = (exit_price - pos.entry_price) * pos.size * pos.leverage
                else:
                    pnl = (pos.entry_price - exit_price) * pos.size * pos.leverage

                # Exit fee
                exit_fee = exit_price * pos.size * self.config.taker_fee
                self.total_fees += exit_fee
                net_pnl = pnl - exit_fee

                # Update capital
                self.capital += net_pnl
                self.total_pnl += net_pnl
                self.total_trades += 1

                if net_pnl > 0:
                    self.winning_trades += 1
                    self.recent_wins.append(abs(net_pnl / self.capital))
                    if len(self.recent_wins) > 50:
                        self.recent_wins.pop(0)
                else:
                    self.recent_losses.append(abs(net_pnl / self.capital))
                    if len(self.recent_losses) > 50:
                        self.recent_losses.pop(0)

                # Log trade
                self.trade_log.append({
                    'entry_time': pos.entry_time,
                    'exit_time': timestamp,
                    'entry_price': pos.entry_price,
                    'exit_price': exit_price,
                    'side': pos.side.value,
                    'size': pos.size,
                    'leverage': pos.leverage,
                    'pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'layer': pos.layer
                })

                positions_to_close.append(i)

        # Remove closed positions
        for i in sorted(positions_to_close, reverse=True):
            self.active_positions.pop(i)

    def calculate_total_unrealized(self, current_price: float) -> float:
        """Calculate total unrealized PnL"""
        unrealized = 0.0
        for pos in self.active_positions:
            if pos.side == PositionType.LONG:
                unrealized += (current_price - pos.entry_price) * pos.size * pos.leverage
            else:
                unrealized += (pos.entry_price - current_price) * pos.size * pos.leverage
        return unrealized

    def update_equity_tracking(self, current_price: float, timestamp: pd.Timestamp) -> bool:
        """Update equity curve and check risk limits"""
        unrealized = self.calculate_total_unrealized(current_price)
        current_equity = self.capital + unrealized

        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Calculate drawdown
        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Record
        self.equity_curve.append(current_equity)
        self.drawdown_curve.append(self.current_drawdown)
        self.timestamps.append(timestamp)

        # Check if should stop
        if self.current_drawdown >= self.config.max_drawdown:
            return False

        return True

    def should_rebalance_grid(self, current_price: float) -> bool:
        """Check if grid needs rebalancing"""
        if not hasattr(self, 'grid_center'):
            return True

        price_change = abs(current_price - self.grid_center) / self.grid_center * 100
        if price_change >= self.config.rebalance_threshold:
            return True

        # Check one-sided fills
        filled_longs = sum(1 for l in self.grid_levels if l.is_filled and l.side == PositionType.LONG)
        filled_shorts = sum(1 for l in self.grid_levels if l.is_filled and l.side == PositionType.SHORT)

        if filled_longs >= self.config.max_one_sided_fills and filled_shorts == 0:
            return True
        if filled_shorts >= self.config.max_one_sided_fills and filled_longs == 0:
            return True

        return False

    def apply_compounding(self):
        """Apply profit compounding"""
        if not self.config.compound_threshold:
            return

        profit_pct = (self.capital - self.initial_capital) / self.initial_capital * 100

        # Compound threshold reached
        if profit_pct >= self.config.compound_threshold:
            self.initial_capital = self.capital

        # Lock profits if high gain
        if profit_pct >= self.config.profit_lock_threshold:
            new_profit = self.capital - self.initial_capital
            lock_amount = new_profit * 0.3  # Lock 30%
            self.locked_profit += lock_amount
            # Don't actually remove from capital, just track


class UltraBacktester:
    """Backtester for ultra-optimized grid system"""

    def __init__(self, config: UltraGridConfig):
        self.config = config
        self.trader: Optional[UltraGridTrader] = None
        self.results: Dict = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load OHLCV data"""
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower()

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run ultra-optimized backtest"""
        self.trader = UltraGridTrader(self.config)

        vol_analyzer = VolatilityAnalyzer()
        trend_analyzer = TrendAnalyzer()

        warmup = 200
        grid_initialized = False

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            lookback = df.iloc[max(0, i-200):i+1]

            current_price = row['close']
            high = row['high']
            low = row['low']

            # Analyze market conditions
            vol_regime = vol_analyzer.calculate_regime_volatility(lookback)
            trend = trend_analyzer.detect_trend(lookback)

            # Skip if volatility too low (no opportunity)
            if vol_regime['current_vol'] < self.config.min_volatility:
                continue

            # Calculate leverage
            leverage = self.trader.calculate_dynamic_leverage(vol_regime, trend)

            # Setup or rebalance grid
            if not grid_initialized or self.trader.should_rebalance_grid(current_price):
                # Close all positions before rebalancing
                for pos in self.trader.active_positions:
                    if pos.side == PositionType.LONG:
                        pnl = (current_price - pos.entry_price) * pos.size * pos.leverage
                    else:
                        pnl = (pos.entry_price - current_price) * pos.size * pos.leverage

                    fee = current_price * pos.size * self.config.taker_fee
                    self.trader.capital += pnl - fee
                    self.trader.total_fees += fee
                    self.trader.total_pnl += pnl - fee
                    self.trader.total_trades += 1
                    if pnl > 0:
                        self.trader.winning_trades += 1

                self.trader.active_positions = []
                self.trader.setup_grid(current_price, vol_regime, trend, leverage, timestamp)
                grid_initialized = True

            # Check for fills
            self.trader.check_grid_fills(high, low, timestamp, leverage)

            # Update positions
            self.trader.update_positions(high, low, current_price, timestamp)

            # Update equity tracking
            can_continue = self.trader.update_equity_tracking(current_price, timestamp)

            # Apply compounding
            self.trader.apply_compounding()

            if not can_continue:
                print(f"  Stopped at {timestamp}: Max drawdown reached ({self.trader.current_drawdown:.2f}%)")
                break

        # Close remaining positions
        final_price = df.iloc[-1]['close']
        for pos in self.trader.active_positions:
            if pos.side == PositionType.LONG:
                pnl = (final_price - pos.entry_price) * pos.size * pos.leverage
            else:
                pnl = (pos.entry_price - final_price) * pos.size * pos.leverage

            fee = final_price * pos.size * self.config.taker_fee
            self.trader.capital += pnl - fee
            self.trader.total_pnl += pnl - fee

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trader or len(self.trader.equity_curve) == 0:
            return {}

        equity = np.array(self.trader.equity_curve)
        drawdowns = np.array(self.trader.drawdown_curve)

        # Returns
        initial = self.config.initial_capital
        final = self.trader.capital
        total_return = (final - initial) / initial * 100

        # Trade stats
        total_trades = self.trader.total_trades
        winning = self.trader.winning_trades
        win_rate = winning / total_trades * 100 if total_trades > 0 else 0

        # Risk metrics
        returns = np.diff(equity) / equity[:-1]
        sharpe = np.sqrt(365 * 24) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        neg_returns = returns[returns < 0]
        sortino = np.sqrt(365 * 24) * np.mean(returns) / np.std(neg_returns) if len(neg_returns) > 0 else 0

        max_dd = np.max(drawdowns)
        calmar = total_return / max_dd if max_dd > 0 else 0

        # Profit metrics
        wins = [t['pnl'] for t in self.trader.trade_log if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in self.trader.trade_log if t['pnl'] < 0]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')

        # Layer analysis
        inner_trades = [t for t in self.trader.trade_log if t['layer'] == 'inner']
        outer_trades = [t for t in self.trader.trade_log if t['layer'] == 'outer']

        inner_pnl = sum(t['pnl'] for t in inner_trades)
        outer_pnl = sum(t['pnl'] for t in outer_trades)

        self.results = {
            'initial_capital': initial,
            'final_capital': final,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'calmar_ratio': calmar,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_fees': self.trader.total_fees,
            'net_pnl': self.trader.total_pnl,
            'inner_layer_pnl': inner_pnl,
            'outer_layer_pnl': outer_pnl,
            'inner_trades': len(inner_trades),
            'outer_trades': len(outer_trades),
            'locked_profit': self.trader.locked_profit,
            'equity_curve': equity.tolist(),
            'drawdown_curve': drawdowns.tolist()
        }

        return self.results

    def print_report(self):
        """Print formatted report"""
        if not self.results:
            return

        r = self.results

        print("\n" + "="*60)
        print("     ULTRA-OPTIMIZED GRID BACKTEST RESULTS")
        print("="*60)

        print("\n💰 CAPITAL PERFORMANCE")
        print("-"*40)
        print(f"  Initial Capital:    ${r['initial_capital']:.2f}")
        print(f"  Final Capital:      ${r['final_capital']:.2f}")
        print(f"  Total Return:       {r['total_return_pct']:.2f}%")
        print(f"  Net PnL:            ${r['net_pnl']:.4f}")
        print(f"  Locked Profit:      ${r['locked_profit']:.4f}")

        print("\n📊 TRADE STATISTICS")
        print("-"*40)
        print(f"  Total Trades:       {r['total_trades']}")
        print(f"  Win Rate:           {r['win_rate_pct']:.2f}%")
        print(f"  Profit Factor:      {r['profit_factor']:.2f}")
        print(f"  Avg Win:            ${r['avg_win']:.6f}")
        print(f"  Avg Loss:           ${r['avg_loss']:.6f}")

        print("\n🔲 LAYER BREAKDOWN")
        print("-"*40)
        print(f"  Inner Layer Trades: {r['inner_trades']}")
        print(f"  Inner Layer PnL:    ${r['inner_layer_pnl']:.4f}")
        print(f"  Outer Layer Trades: {r['outer_trades']}")
        print(f"  Outer Layer PnL:    ${r['outer_layer_pnl']:.4f}")

        print("\n⚠️ RISK METRICS")
        print("-"*40)
        print(f"  Max Drawdown:       {r['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:       {r['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {r['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:       {r['calmar_ratio']:.2f}")

        print("\n💸 FEES")
        print("-"*40)
        print(f"  Total Fees:         ${r['total_fees']:.6f}")

        print("="*60)


def optimize_ultra_config(df: pd.DataFrame, n_iterations: int = 50) -> UltraGridConfig:
    """Find optimal configuration"""
    print("\n🔧 Optimizing Ultra Grid Configuration...")

    best_score = -np.inf
    best_config = None

    for i in range(n_iterations):
        config = UltraGridConfig(
            initial_capital=4.0,
            inner_grids=np.random.choice([6, 8, 10, 12]),
            outer_grids=np.random.choice([4, 6, 8]),
            inner_spacing_pct=np.random.uniform(0.1, 0.25),
            outer_spacing_pct=np.random.uniform(0.3, 0.6),
            base_leverage=np.random.choice([20, 25, 30, 35, 40]),
            max_leverage=np.random.choice([50, 60, 75]),
            max_drawdown=np.random.choice([10, 12, 15]),
            soft_drawdown=np.random.choice([6, 8, 10]),
            position_stop_loss=np.random.uniform(3, 8),
            base_take_profit=np.random.uniform(0.2, 0.5),
            extended_take_profit=np.random.uniform(0.4, 0.8),
            compound_threshold=np.random.uniform(2, 5),
        )

        try:
            backtester = UltraBacktester(config)
            results = backtester.run_backtest(df.copy())

            if not results or results['total_trades'] < 20:
                continue

            # Score: prioritize return, penalize drawdown heavily
            score = (results['total_return_pct'] * 2
                    - results['max_drawdown_pct'] * 3
                    + results['win_rate_pct'] * 0.3
                    + min(results['sharpe_ratio'], 3) * 5
                    + min(results['profit_factor'], 3) * 3)

            if score > best_score:
                best_score = score
                best_config = config
                print(f"  [Iter {i+1}] New best: Return={results['total_return_pct']:.1f}%, DD={results['max_drawdown_pct']:.1f}%, WR={results['win_rate_pct']:.1f}%")

        except Exception as e:
            continue

    return best_config


def run_ultra_analysis():
    """Run complete ultra-optimized analysis"""
    print("\n" + "="*70)
    print("      ULTRA-OPTIMIZED GRID TRADING SYSTEM")
    print("      Designed for $4 Capital with Maximum Returns")
    print("="*70)

    data_files = [
        ('BTCUSDT_1h_1year.csv', 'BTC/USDT'),
        ('SOLUSDT_1h_1year.csv', 'SOL/USDT'),
        ('XRPUSDT_1h_1year.csv', 'XRP/USDT'),
    ]

    all_results = {}

    for file_name, symbol in data_files:
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {symbol}")
        print('='*60)

        try:
            # Load data
            backtester = UltraBacktester(UltraGridConfig())
            df = backtester.load_data(file_name)
            print(f"  Data: {len(df)} candles")

            # Optimize
            best_config = optimize_ultra_config(df, n_iterations=40)

            if best_config:
                print(f"\n📋 Optimal Configuration:")
                print(f"  Inner Grids: {best_config.inner_grids}, Outer: {best_config.outer_grids}")
                print(f"  Inner Spacing: {best_config.inner_spacing_pct:.2f}%, Outer: {best_config.outer_spacing_pct:.2f}%")
                print(f"  Leverage: {best_config.base_leverage}x (max {best_config.max_leverage}x)")
                print(f"  Max DD: {best_config.max_drawdown}%, Stop Loss: {best_config.position_stop_loss:.1f}%")

                # Final backtest
                final_bt = UltraBacktester(best_config)
                results = final_bt.run_backtest(df.copy())
                final_bt.print_report()

                all_results[symbol] = {
                    'config': best_config,
                    'results': results
                }

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("                    FINAL SUMMARY")
    print("="*70)
    print(f"{'Asset':<12} {'Return %':<12} {'Max DD %':<12} {'Win Rate':<12} {'Trades':<10}")
    print("-"*70)

    for symbol, data in all_results.items():
        r = data['results']
        print(f"{symbol:<12} {r['total_return_pct']:<12.1f} {r['max_drawdown_pct']:<12.1f} {r['win_rate_pct']:<12.1f} {r['total_trades']:<10}")

    return all_results


if __name__ == "__main__":
    results = run_ultra_analysis()
