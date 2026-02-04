#!/usr/bin/env python3
"""
FUTURES NEUTRAL GRID TRADING SYSTEM
====================================
A comprehensive grid trading system designed for small capital accounts ($4+)
with aggressive compounding, low drawdown, and optimized leverage.

Features:
- Neutral grid strategy (profits in both directions)
- Dynamic volatility-based grid spacing
- Adaptive leverage based on account size and market conditions
- Advanced risk management with strict drawdown limits
- Capital compounding for rapid growth
- Multi-asset support with correlation analysis
- Real-time position sizing optimization
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OrderSide(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass
class GridOrder:
    """Represents a single grid order"""
    id: int
    price: float
    side: OrderSide
    size: float
    status: OrderStatus = OrderStatus.PENDING
    entry_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: float = 0.0
    fees: float = 0.0


@dataclass
class Position:
    """Represents an open position"""
    side: OrderSide
    entry_price: float
    size: float
    entry_time: pd.Timestamp
    unrealized_pnl: float = 0.0
    leverage: float = 1.0


@dataclass
class TradeResult:
    """Result of a completed trade"""
    entry_price: float
    exit_price: float
    side: OrderSide
    size: float
    pnl: float
    fees: float
    duration: pd.Timedelta
    leverage: float


@dataclass
class GridConfig:
    """Configuration for grid trading"""
    # Grid parameters
    num_grids: int = 20  # Number of grid levels each side
    grid_spacing_pct: float = 0.3  # Base grid spacing percentage
    use_dynamic_spacing: bool = True  # Adjust spacing based on volatility

    # Capital and leverage
    initial_capital: float = 4.0  # Starting capital in USDT
    base_leverage: float = 20  # Base leverage (adjusted dynamically)
    max_leverage: float = 75  # Maximum allowed leverage
    min_leverage: float = 5  # Minimum leverage

    # Risk management
    max_drawdown_pct: float = 15.0  # Maximum drawdown before stop
    position_size_pct: float = 2.5  # Size per grid as % of capital
    max_total_exposure_pct: float = 80.0  # Max total position exposure
    stop_loss_pct: float = 8.0  # Individual position stop loss

    # Fees (Binance Futures typical)
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0004  # 0.04%

    # Compounding
    compound_profits: bool = True
    compound_threshold: float = 1.5  # Compound when profits exceed this %

    # Volatility settings
    volatility_lookback: int = 24  # Hours for volatility calculation
    volatility_multiplier: float = 1.5  # Multiply ATR for spacing

    # Trading filters
    min_volume_threshold: float = 0.0  # Minimum volume filter
    trend_filter_enabled: bool = True  # Use trend filter
    trend_ma_period: int = 50  # MA period for trend

    # Grid reset conditions
    reset_on_trend_change: bool = True
    reset_threshold_pct: float = 5.0  # Reset if price moves this much


class VolatilityCalculator:
    """Calculates various volatility metrics"""

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_bollinger_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
        """Calculate Bollinger Band width as volatility measure"""
        close = df['close']
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma * 100
        return width

    @staticmethod
    def calculate_historical_volatility(df: pd.DataFrame, period: int = 24) -> pd.Series:
        """Calculate historical volatility (annualized)"""
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(365 * 24)  # Annualized for hourly
        return volatility


class RiskManager:
    """Advanced risk management for grid trading"""

    def __init__(self, config: GridConfig):
        self.config = config
        self.peak_equity = config.initial_capital
        self.current_drawdown = 0.0
        self.max_recorded_drawdown = 0.0
        self.daily_pnl = 0.0
        self.consecutive_losses = 0

    def update_equity(self, current_equity: float) -> bool:
        """Update equity tracking and check risk limits"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.consecutive_losses = 0

        self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity * 100
        self.max_recorded_drawdown = max(self.max_recorded_drawdown, self.current_drawdown)

        # Check if we should stop trading
        if self.current_drawdown >= self.config.max_drawdown_pct:
            return False  # Stop trading
        return True

    def calculate_position_size(self, capital: float, price: float,
                                volatility: float, leverage: float) -> float:
        """Calculate optimal position size based on risk parameters"""
        base_size = (capital * self.config.position_size_pct / 100) * leverage / price

        # Adjust for volatility (reduce size in high volatility)
        vol_adjustment = 1.0
        if volatility > 0.5:  # High volatility
            vol_adjustment = 0.5
        elif volatility > 0.3:
            vol_adjustment = 0.7

        # Adjust for drawdown (reduce size as drawdown increases)
        dd_adjustment = 1.0 - (self.current_drawdown / self.config.max_drawdown_pct * 0.5)
        dd_adjustment = max(0.3, dd_adjustment)

        # Adjust for consecutive losses
        loss_adjustment = max(0.5, 1.0 - (self.consecutive_losses * 0.1))

        final_size = base_size * vol_adjustment * dd_adjustment * loss_adjustment
        return max(0.001, final_size)  # Minimum size

    def calculate_dynamic_leverage(self, capital: float, volatility: float,
                                   trend_strength: float) -> float:
        """Calculate optimal leverage based on conditions"""
        base_leverage = self.config.base_leverage

        # Small account boost (more aggressive for tiny accounts)
        if capital < 10:
            account_multiplier = 1.5
        elif capital < 50:
            account_multiplier = 1.2
        else:
            account_multiplier = 1.0

        # Volatility adjustment (lower leverage in high volatility)
        if volatility > 0.5:
            vol_multiplier = 0.5
        elif volatility > 0.3:
            vol_multiplier = 0.7
        else:
            vol_multiplier = 1.0

        # Trend adjustment (higher leverage with strong trends for grid)
        trend_multiplier = 1.0 + abs(trend_strength) * 0.3

        # Drawdown adjustment
        dd_multiplier = 1.0 - (self.current_drawdown / self.config.max_drawdown_pct * 0.5)
        dd_multiplier = max(0.4, dd_multiplier)

        leverage = base_leverage * account_multiplier * vol_multiplier * trend_multiplier * dd_multiplier
        return np.clip(leverage, self.config.min_leverage, self.config.max_leverage)

    def record_loss(self):
        """Record a losing trade"""
        self.consecutive_losses += 1

    def record_win(self):
        """Record a winning trade"""
        self.consecutive_losses = 0


class NeutralGridTrader:
    """
    Main Grid Trading Engine
    Implements a market-neutral grid strategy that profits from price oscillations
    """

    def __init__(self, config: GridConfig):
        self.config = config
        self.capital = config.initial_capital
        self.initial_capital = config.initial_capital
        self.risk_manager = RiskManager(config)
        self.volatility_calc = VolatilityCalculator()

        # Grid state
        self.grid_orders: List[GridOrder] = []
        self.positions: List[Position] = []
        self.completed_trades: List[TradeResult] = []

        # Tracking
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []

        # Grid center and boundaries
        self.grid_center: Optional[float] = None
        self.grid_upper: Optional[float] = None
        self.grid_lower: Optional[float] = None

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_fees = 0.0
        self.total_pnl = 0.0

    def calculate_grid_levels(self, center_price: float, atr: float,
                             volatility: float) -> Tuple[List[float], List[float]]:
        """Calculate grid price levels based on volatility"""
        if self.config.use_dynamic_spacing:
            # Dynamic spacing based on ATR
            spacing = (atr / center_price) * self.config.volatility_multiplier * 100
            spacing = np.clip(spacing, 0.1, 2.0)  # Min 0.1%, max 2%
        else:
            spacing = self.config.grid_spacing_pct

        buy_levels = []
        sell_levels = []

        for i in range(1, self.config.num_grids + 1):
            buy_price = center_price * (1 - spacing * i / 100)
            sell_price = center_price * (1 + spacing * i / 100)
            buy_levels.append(buy_price)
            sell_levels.append(sell_price)

        self.grid_center = center_price
        self.grid_lower = min(buy_levels)
        self.grid_upper = max(sell_levels)

        return buy_levels, sell_levels

    def setup_grid(self, center_price: float, atr: float, volatility: float,
                   leverage: float, timestamp: pd.Timestamp):
        """Initialize or reset the grid"""
        buy_levels, sell_levels = self.calculate_grid_levels(center_price, atr, volatility)

        self.grid_orders = []
        order_id = 0

        # Calculate position size for each grid level
        position_size = self.risk_manager.calculate_position_size(
            self.capital, center_price, volatility, leverage
        )

        # Create buy orders below price
        for price in buy_levels:
            order = GridOrder(
                id=order_id,
                price=price,
                side=OrderSide.LONG,
                size=position_size
            )
            self.grid_orders.append(order)
            order_id += 1

        # Create sell orders above price
        for price in sell_levels:
            order = GridOrder(
                id=order_id,
                price=price,
                side=OrderSide.SHORT,
                size=position_size
            )
            self.grid_orders.append(order)
            order_id += 1

    def check_order_fills(self, high: float, low: float, close: float,
                         timestamp: pd.Timestamp, leverage: float):
        """Check if any grid orders should be filled"""
        for order in self.grid_orders:
            if order.status != OrderStatus.PENDING:
                continue

            filled = False

            if order.side == OrderSide.LONG and low <= order.price:
                filled = True
            elif order.side == OrderSide.SHORT and high >= order.price:
                filled = True

            if filled:
                order.status = OrderStatus.FILLED
                order.entry_time = timestamp

                # Create position
                position = Position(
                    side=order.side,
                    entry_price=order.price,
                    size=order.size,
                    entry_time=timestamp,
                    leverage=leverage
                )
                self.positions.append(position)

                # Calculate entry fee
                fee = order.price * order.size * self.config.taker_fee
                self.total_fees += fee
                order.fees = fee

    def check_position_exits(self, high: float, low: float, close: float,
                            timestamp: pd.Timestamp, atr: float):
        """Check if positions should be closed"""
        positions_to_remove = []

        for i, pos in enumerate(self.positions):
            exit_triggered = False
            exit_price = close

            # Calculate take profit and stop loss levels
            if self.config.use_dynamic_spacing:
                tp_distance = atr * 1.5
            else:
                tp_distance = pos.entry_price * self.config.grid_spacing_pct / 100

            sl_distance = pos.entry_price * self.config.stop_loss_pct / 100

            if pos.side == OrderSide.LONG:
                tp_price = pos.entry_price + tp_distance
                sl_price = pos.entry_price - sl_distance

                if high >= tp_price:
                    exit_triggered = True
                    exit_price = tp_price
                elif low <= sl_price:
                    exit_triggered = True
                    exit_price = sl_price

            else:  # SHORT
                tp_price = pos.entry_price - tp_distance
                sl_price = pos.entry_price + sl_distance

                if low <= tp_price:
                    exit_triggered = True
                    exit_price = tp_price
                elif high >= sl_price:
                    exit_triggered = True
                    exit_price = sl_price

            if exit_triggered:
                # Calculate PnL
                if pos.side == OrderSide.LONG:
                    pnl = (exit_price - pos.entry_price) * pos.size * pos.leverage
                else:
                    pnl = (pos.entry_price - exit_price) * pos.size * pos.leverage

                # Calculate exit fee
                exit_fee = exit_price * pos.size * self.config.taker_fee
                self.total_fees += exit_fee

                net_pnl = pnl - exit_fee

                # Record trade
                trade = TradeResult(
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    side=pos.side,
                    size=pos.size,
                    pnl=net_pnl,
                    fees=exit_fee,
                    duration=timestamp - pos.entry_time,
                    leverage=pos.leverage
                )
                self.completed_trades.append(trade)

                # Update statistics
                self.total_trades += 1
                self.total_pnl += net_pnl
                self.capital += net_pnl

                if net_pnl > 0:
                    self.winning_trades += 1
                    self.risk_manager.record_win()
                else:
                    self.risk_manager.record_loss()

                positions_to_remove.append(i)

        # Remove closed positions
        for i in sorted(positions_to_remove, reverse=True):
            self.positions.pop(i)

    def update_unrealized_pnl(self, current_price: float) -> float:
        """Calculate total unrealized PnL"""
        unrealized = 0.0
        for pos in self.positions:
            if pos.side == OrderSide.LONG:
                unrealized += (current_price - pos.entry_price) * pos.size * pos.leverage
            else:
                unrealized += (pos.entry_price - current_price) * pos.size * pos.leverage
        return unrealized

    def should_reset_grid(self, current_price: float) -> bool:
        """Check if grid should be reset"""
        if self.grid_center is None:
            return True

        price_change_pct = abs(current_price - self.grid_center) / self.grid_center * 100

        if price_change_pct >= self.config.reset_threshold_pct:
            return True

        # Also reset if too many orders filled on one side
        filled_longs = sum(1 for o in self.grid_orders
                         if o.side == OrderSide.LONG and o.status == OrderStatus.FILLED)
        filled_shorts = sum(1 for o in self.grid_orders
                          if o.side == OrderSide.SHORT and o.status == OrderStatus.FILLED)

        if filled_longs >= self.config.num_grids * 0.7 or filled_shorts >= self.config.num_grids * 0.7:
            return True

        return False

    def compound_capital(self):
        """Compound profits if threshold is met"""
        if not self.config.compound_profits:
            return

        profit_pct = (self.capital - self.initial_capital) / self.initial_capital * 100

        if profit_pct >= self.config.compound_threshold:
            # Reset initial capital to current for compounding
            self.initial_capital = self.capital


class GridBacktester:
    """Comprehensive backtesting engine for grid trading"""

    def __init__(self, config: GridConfig):
        self.config = config
        self.trader: Optional[NeutralGridTrader] = None
        self.results: Dict = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and prepare OHLCV data"""
        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Data must contain columns: {required}")

        # Parse timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')

        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        return df

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        vol_calc = VolatilityCalculator()

        # ATR
        df['atr'] = vol_calc.calculate_atr(df, self.config.volatility_lookback)

        # Bollinger Width
        df['bb_width'] = vol_calc.calculate_bollinger_width(df)

        # Historical Volatility
        df['hist_vol'] = vol_calc.calculate_historical_volatility(df, self.config.volatility_lookback)

        # Trend indicators
        df['sma_fast'] = df['close'].rolling(window=20).mean()
        df['sma_slow'] = df['close'].rolling(window=self.config.trend_ma_period).mean()
        df['trend'] = np.where(df['sma_fast'] > df['sma_slow'], 1, -1)
        df['trend_strength'] = (df['sma_fast'] - df['sma_slow']) / df['sma_slow']

        # Volume analysis (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0

        # RSI for additional filtering
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Forward fill NaN values
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """Run the backtest simulation"""
        self.trader = NeutralGridTrader(self.config)

        # Skip warmup period
        warmup = max(self.config.volatility_lookback, self.config.trend_ma_period, 50)

        grid_initialized = False
        last_leverage = self.config.base_leverage

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]

            current_price = row['close']
            high = row['high']
            low = row['low']
            atr = row['atr']
            volatility = row['hist_vol']
            trend_strength = row['trend_strength']

            # Calculate dynamic leverage
            leverage = self.trader.risk_manager.calculate_dynamic_leverage(
                self.trader.capital, volatility, trend_strength
            )
            last_leverage = leverage

            # Initialize or reset grid
            if not grid_initialized or self.trader.should_reset_grid(current_price):
                # Close all open positions before reset
                for pos in self.trader.positions:
                    if pos.side == OrderSide.LONG:
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

                self.trader.positions = []
                self.trader.setup_grid(current_price, atr, volatility, leverage, timestamp)
                grid_initialized = True

            # Check for order fills
            self.trader.check_order_fills(high, low, current_price, timestamp, leverage)

            # Check for position exits
            self.trader.check_position_exits(high, low, current_price, timestamp, atr)

            # Update unrealized PnL
            unrealized = self.trader.update_unrealized_pnl(current_price)

            # Calculate current equity
            current_equity = self.trader.capital + unrealized

            # Update risk manager
            can_continue = self.trader.risk_manager.update_equity(current_equity)

            # Record equity curve
            self.trader.equity_curve.append(current_equity)
            self.trader.drawdown_curve.append(self.trader.risk_manager.current_drawdown)
            self.trader.timestamps.append(timestamp)

            # Compound profits
            self.trader.compound_capital()

            # Check if we should stop
            if not can_continue:
                print(f"Stopped at {timestamp}: Max drawdown reached")
                break

        # Close any remaining positions
        final_price = df.iloc[-1]['close']
        for pos in self.trader.positions:
            if pos.side == OrderSide.LONG:
                pnl = (final_price - pos.entry_price) * pos.size * pos.leverage
            else:
                pnl = (pos.entry_price - final_price) * pos.size * pos.leverage

            fee = final_price * pos.size * self.config.taker_fee
            self.trader.capital += pnl - fee
            self.trader.total_fees += fee
            self.trader.total_pnl += pnl - fee

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trader or len(self.trader.equity_curve) == 0:
            return {}

        equity = np.array(self.trader.equity_curve)
        drawdowns = np.array(self.trader.drawdown_curve)

        # Basic metrics
        initial_capital = self.config.initial_capital
        final_capital = self.trader.capital
        total_return = (final_capital - initial_capital) / initial_capital * 100

        # Trade statistics
        total_trades = self.trader.total_trades
        winning_trades = self.trader.winning_trades
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # Calculate returns for Sharpe ratio
        returns = np.diff(equity) / equity[:-1]
        sharpe_ratio = np.sqrt(365 * 24) * np.mean(returns) / np.std(returns) if len(returns) > 0 and np.std(returns) > 0 else 0

        # Sortino ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-10
        sortino_ratio = np.sqrt(365 * 24) * np.mean(returns) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        max_drawdown = np.max(drawdowns)

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Profit factor
        winning_pnl = sum(t.pnl for t in self.trader.completed_trades if t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in self.trader.completed_trades if t.pnl < 0))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

        # Average trade metrics
        avg_win = winning_pnl / winning_trades if winning_trades > 0 else 0
        losing_trades = total_trades - winning_trades
        avg_loss = losing_pnl / losing_trades if losing_trades > 0 else 0

        # Risk-reward ratio
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')

        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')

        # Trading frequency
        if len(self.trader.timestamps) > 1:
            duration = (self.trader.timestamps[-1] - self.trader.timestamps[0]).days
            trades_per_day = total_trades / duration if duration > 0 else 0
        else:
            trades_per_day = 0

        self.results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'risk_reward_ratio': risk_reward,
            'recovery_factor': recovery_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_fees': self.trader.total_fees,
            'net_pnl': self.trader.total_pnl,
            'trades_per_day': trades_per_day,
            'equity_curve': equity.tolist(),
            'drawdown_curve': drawdowns.tolist(),
            'timestamps': [str(t) for t in self.trader.timestamps]
        }

        return self.results

    def print_report(self):
        """Print formatted backtest report"""
        if not self.results:
            print("No results to display")
            return

        r = self.results

        print("\n" + "="*60)
        print("        GRID TRADING BACKTEST REPORT")
        print("="*60)

        print("\n📊 CAPITAL PERFORMANCE")
        print("-"*40)
        print(f"  Initial Capital:    ${r['initial_capital']:.2f}")
        print(f"  Final Capital:      ${r['final_capital']:.2f}")
        print(f"  Total Return:       {r['total_return_pct']:.2f}%")
        print(f"  Net PnL:            ${r['net_pnl']:.2f}")
        print(f"  Total Fees:         ${r['total_fees']:.4f}")

        print("\n📈 TRADE STATISTICS")
        print("-"*40)
        print(f"  Total Trades:       {r['total_trades']}")
        print(f"  Winning Trades:     {r['winning_trades']}")
        print(f"  Losing Trades:      {r['losing_trades']}")
        print(f"  Win Rate:           {r['win_rate_pct']:.2f}%")
        print(f"  Trades per Day:     {r['trades_per_day']:.2f}")

        print("\n💰 PROFIT METRICS")
        print("-"*40)
        print(f"  Profit Factor:      {r['profit_factor']:.2f}")
        print(f"  Risk/Reward Ratio:  {r['risk_reward_ratio']:.2f}")
        print(f"  Avg Win:            ${r['avg_win']:.4f}")
        print(f"  Avg Loss:           ${r['avg_loss']:.4f}")

        print("\n⚠️ RISK METRICS")
        print("-"*40)
        print(f"  Max Drawdown:       {r['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:       {r['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio:      {r['sortino_ratio']:.2f}")
        print(f"  Calmar Ratio:       {r['calmar_ratio']:.2f}")
        print(f"  Recovery Factor:    {r['recovery_factor']:.2f}")

        print("\n" + "="*60)


class GridOptimizer:
    """Optimize grid trading parameters"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.best_config: Optional[GridConfig] = None
        self.best_score: float = -np.inf
        self.optimization_results: List[Dict] = []

    def objective_function(self, config: GridConfig) -> float:
        """
        Calculate optimization score
        Prioritizes: High returns, Low drawdown, High win rate
        """
        backtester = GridBacktester(config)
        prepared_df = backtester.prepare_indicators(self.df.copy())
        results = backtester.run_backtest(prepared_df)

        if not results or results['total_trades'] < 10:
            return -np.inf

        # Multi-objective scoring
        return_score = results['total_return_pct'] * 2
        drawdown_penalty = results['max_drawdown_pct'] * 3
        win_rate_bonus = results['win_rate_pct'] * 0.5
        sharpe_bonus = results['sharpe_ratio'] * 10
        profit_factor_bonus = min(results['profit_factor'], 3) * 5

        # Heavy penalty for high drawdown
        if results['max_drawdown_pct'] > 15:
            drawdown_penalty *= 2

        score = return_score - drawdown_penalty + win_rate_bonus + sharpe_bonus + profit_factor_bonus

        return score

    def grid_search(self, param_ranges: Dict) -> GridConfig:
        """Perform grid search optimization"""
        print("\n🔍 Starting Grid Search Optimization...")
        print("-" * 50)

        best_score = -np.inf
        best_params = {}
        total_combinations = 1

        for key, values in param_ranges.items():
            total_combinations *= len(values)

        print(f"Testing {total_combinations} parameter combinations...\n")

        # Generate all combinations
        import itertools
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())

        for i, combo in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combo))
            config = GridConfig(**params)

            try:
                score = self.objective_function(config)

                self.optimization_results.append({
                    'params': params,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    best_params = params
                    self.best_config = config
                    print(f"  [{i+1}/{total_combinations}] New best score: {score:.2f}")
                    for k, v in params.items():
                        print(f"      {k}: {v}")

            except Exception as e:
                print(f"  [{i+1}/{total_combinations}] Error: {str(e)}")
                continue

        print(f"\n✅ Optimization complete! Best score: {best_score:.2f}")
        self.best_score = best_score

        return self.best_config

    def random_search(self, param_ranges: Dict, n_iterations: int = 50) -> GridConfig:
        """Perform random search optimization"""
        print(f"\n🎲 Starting Random Search Optimization ({n_iterations} iterations)...")
        print("-" * 50)

        best_score = -np.inf

        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for key, value_range in param_ranges.items():
                if isinstance(value_range, list):
                    params[key] = np.random.choice(value_range)
                elif isinstance(value_range, tuple):
                    if isinstance(value_range[0], int):
                        params[key] = np.random.randint(value_range[0], value_range[1] + 1)
                    else:
                        params[key] = np.random.uniform(value_range[0], value_range[1])

            config = GridConfig(**params)

            try:
                score = self.objective_function(config)

                self.optimization_results.append({
                    'params': params,
                    'score': score
                })

                if score > best_score:
                    best_score = score
                    self.best_config = config
                    self.best_score = best_score
                    print(f"  [Iter {i+1}] New best score: {score:.2f}")

            except Exception as e:
                continue

        print(f"\n✅ Random search complete! Best score: {best_score:.2f}")

        return self.best_config


def run_full_analysis():
    """Run complete grid trading analysis with optimization"""

    print("\n" + "="*70)
    print("    FUTURES NEUTRAL GRID TRADING SYSTEM - FULL ANALYSIS")
    print("    Optimized for Small Capital ($4) with Low Drawdown")
    print("="*70)

    # Load all available data files
    data_files = [
        ('BTCUSDT_1h_1year.csv', 'BTC/USDT'),
        ('SOLUSDT_1h_1year.csv', 'SOL/USDT'),
        ('XRPUSDT_1h_1year.csv', 'XRP/USDT'),
    ]

    all_results = {}

    for file_name, symbol in data_files:
        print(f"\n{'='*60}")
        print(f"  Analyzing {symbol}")
        print('='*60)

        try:
            # Load data
            backtester = GridBacktester(GridConfig())
            df = backtester.load_data(file_name)
            print(f"  Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

            # Define parameter ranges for optimization
            param_ranges = {
                'num_grids': [10, 15, 20, 25],
                'grid_spacing_pct': [0.2, 0.3, 0.4, 0.5],
                'base_leverage': [15, 20, 25, 30],
                'max_drawdown_pct': [10, 12, 15],
                'position_size_pct': [2.0, 2.5, 3.0],
                'stop_loss_pct': [5.0, 8.0, 10.0],
            }

            # Run optimization
            optimizer = GridOptimizer(df)
            best_config = optimizer.random_search(param_ranges, n_iterations=30)

            if best_config:
                print(f"\n📋 Best Configuration for {symbol}:")
                print(f"  - Num Grids: {best_config.num_grids}")
                print(f"  - Grid Spacing: {best_config.grid_spacing_pct}%")
                print(f"  - Base Leverage: {best_config.base_leverage}x")
                print(f"  - Max Drawdown: {best_config.max_drawdown_pct}%")
                print(f"  - Position Size: {best_config.position_size_pct}%")
                print(f"  - Stop Loss: {best_config.stop_loss_pct}%")

                # Run final backtest with best config
                final_backtester = GridBacktester(best_config)
                prepared_df = final_backtester.prepare_indicators(df.copy())
                results = final_backtester.run_backtest(prepared_df)
                final_backtester.print_report()

                all_results[symbol] = {
                    'config': best_config,
                    'results': results
                }

        except Exception as e:
            print(f"  Error analyzing {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "="*70)
    print("                    SUMMARY OF ALL ASSETS")
    print("="*70)
    print(f"{'Symbol':<12} {'Return %':<12} {'Max DD %':<12} {'Win Rate %':<12} {'Sharpe':<10}")
    print("-"*70)

    for symbol, data in all_results.items():
        r = data['results']
        print(f"{symbol:<12} {r['total_return_pct']:<12.2f} {r['max_drawdown_pct']:<12.2f} {r['win_rate_pct']:<12.2f} {r['sharpe_ratio']:<10.2f}")

    return all_results


if __name__ == "__main__":
    results = run_full_analysis()
