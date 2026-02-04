"""
Backtesting Framework for Small Account Strategy
Simulates trading with realistic conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from config import backtest_config, strategy_config
from strategy import SmallAccountStrategy, TradeSignal, SignalType, StrategyType
from indicators import calculate_all_indicators


@dataclass
class Trade:
    """Record of a single trade"""
    id: int
    symbol: str
    side: str  # LONG or SHORT
    strategy: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    take_profit_2: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED, TP1, TP2
    reason: str = ""
    confidence: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results"""
    initial_balance: float
    final_balance: float
    total_pnl: float
    total_pnl_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_duration: float
    best_trade: float
    worst_trade: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    strategy_breakdown: Dict[str, Dict] = field(default_factory=dict)
    monthly_returns: Dict[str, float] = field(default_factory=dict)


class Backtester:
    """
    Backtesting engine for small account strategy
    """

    def __init__(
        self,
        initial_balance: float = None,
        commission_rate: float = None,
        slippage_pct: float = None,
        leverage: int = None
    ):
        self.initial_balance = initial_balance or backtest_config.initial_balance
        self.commission_rate = commission_rate or backtest_config.commission_rate
        self.slippage_pct = slippage_pct or backtest_config.slippage_pct
        self.leverage = leverage or strategy_config.leverage

        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades: List[Trade] = []
        self.open_position: Optional[Trade] = None
        self.trade_counter = 0

        self.strategy = SmallAccountStrategy()

        # Performance tracking
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.daily_returns = []

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare OHLCV data"""
        df = pd.read_csv(filepath)

        # Standardize column names
        column_mapping = {
            'open_time': 'timestamp',
            'Open': 'open', 'open': 'open',
            'High': 'high', 'high': 'high',
            'Low': 'low', 'low': 'low',
            'Close': 'close', 'close': 'close',
            'Volume': 'volume', 'volume': 'volume',
            'quote_volume': 'quote_volume'
        }

        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Parse timestamp
        if 'timestamp' in df.columns or 'open_time' in df.columns:
            ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
            if df[ts_col].dtype == 'int64' or df[ts_col].dtype == 'float64':
                df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df[ts_col])
            df.set_index('timestamp', inplace=True)

        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna()
        return df

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        start_idx: int = 200,  # Need history for indicators
        progress_callback: callable = None
    ) -> BacktestResult:
        """
        Run backtest on historical data

        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            start_idx: Starting index (need history for indicators)
            progress_callback: Optional callback for progress updates
        """
        self.reset()

        total_bars = len(df) - start_idx
        last_date = None

        for i in range(start_idx, len(df)):
            # Get data window
            window = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            current_time = df.index[i]

            # Track daily returns
            if hasattr(current_time, 'date'):
                current_date = current_time.date()
                if last_date and current_date != last_date:
                    self.daily_returns.append(
                        (self.balance - self.equity_curve[-1]) / self.equity_curve[-1]
                        if self.equity_curve[-1] > 0 else 0
                    )
                last_date = current_date

            # Check open position
            if self.open_position:
                self._check_exit(current_bar, current_time)

            # Look for new signals if no position
            if not self.open_position:
                signal = self.strategy.analyze(window)
                if signal:
                    self._open_position(signal, current_bar, current_time, symbol)

            # Update equity curve
            self._update_equity(current_bar)

            # Progress callback
            if progress_callback and (i - start_idx) % 100 == 0:
                progress = (i - start_idx) / total_bars * 100
                progress_callback(progress, self.balance)

        # Close any remaining position
        if self.open_position:
            self._close_position(df.iloc[-1], df.index[-1], "End of backtest")

        return self._generate_results()

    def reset(self):
        """Reset backtester state"""
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.open_position = None
        self.trade_counter = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.daily_returns = []
        self.strategy = SmallAccountStrategy()

    def _open_position(
        self,
        signal: TradeSignal,
        bar: pd.Series,
        time: pd.Timestamp,
        symbol: str
    ):
        """Open a new position with proper risk-based sizing"""
        self.trade_counter += 1

        # Apply slippage to entry
        if signal.signal_type == SignalType.LONG:
            entry_price = signal.entry_price * (1 + self.slippage_pct)
        else:
            entry_price = signal.entry_price * (1 - self.slippage_pct)

        # RISK-BASED POSITION SIZING
        # Calculate the risk amount (how much we're willing to lose)
        risk_amount = self.balance * strategy_config.max_risk_per_trade

        # Calculate stop distance as percentage
        stop_distance_pct = abs(entry_price - signal.stop_loss) / entry_price

        # Prevent division by zero or tiny stops
        if stop_distance_pct < 0.001:
            stop_distance_pct = 0.01  # Minimum 1% stop

        # Position value based on risk (if stop hit, we lose risk_amount)
        position_value = risk_amount / stop_distance_pct

        # Cap position at max allowed
        max_position = self.balance * self.leverage * strategy_config.position_size_pct
        position_value = min(position_value, max_position)

        # Ensure minimum notional - if we can't meet it, scale up but limit total risk
        if position_value < strategy_config.min_notional:
            # Check if we can afford the min notional position
            min_margin_required = strategy_config.min_notional / self.leverage
            if self.balance >= min_margin_required:
                position_value = strategy_config.min_notional
            else:
                # Can't afford minimum position, skip this trade
                return

        quantity = position_value / entry_price

        # Calculate fees
        fees = position_value * self.commission_rate

        self.open_position = Trade(
            id=self.trade_counter,
            symbol=symbol,
            side="LONG" if signal.signal_type == SignalType.LONG else "SHORT",
            strategy=signal.strategy.value,
            entry_time=time,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            take_profit_2=signal.take_profit_2,
            fees=fees,
            reason=signal.reason,
            confidence=signal.confidence
        )

    def _check_exit(self, bar: pd.Series, time: pd.Timestamp):
        """Check if position should be closed"""
        if not self.open_position:
            return

        pos = self.open_position

        if pos.side == "LONG":
            # Check stop loss
            if bar['low'] <= pos.stop_loss:
                exit_price = pos.stop_loss * (1 - self.slippage_pct)
                self._close_position(bar, time, "Stop Loss", exit_price)
                return

            # Check take profit
            if bar['high'] >= pos.take_profit:
                exit_price = pos.take_profit * (1 - self.slippage_pct)
                self._close_position(bar, time, "Take Profit", exit_price)
                return

        else:  # SHORT
            # Check stop loss
            if bar['high'] >= pos.stop_loss:
                exit_price = pos.stop_loss * (1 + self.slippage_pct)
                self._close_position(bar, time, "Stop Loss", exit_price)
                return

            # Check take profit
            if bar['low'] <= pos.take_profit:
                exit_price = pos.take_profit * (1 + self.slippage_pct)
                self._close_position(bar, time, "Take Profit", exit_price)
                return

    def _close_position(
        self,
        bar: pd.Series,
        time: pd.Timestamp,
        reason: str,
        exit_price: float = None
    ):
        """Close the open position"""
        if not self.open_position:
            return

        pos = self.open_position

        if exit_price is None:
            exit_price = bar['close']
            if pos.side == "LONG":
                exit_price *= (1 - self.slippage_pct)
            else:
                exit_price *= (1 + self.slippage_pct)

        # Calculate PnL
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Subtract fees (entry + exit)
        exit_fees = pos.quantity * exit_price * self.commission_rate
        total_fees = pos.fees + exit_fees
        net_pnl = pnl - total_fees

        # Update position record
        pos.exit_time = time
        pos.exit_price = exit_price
        pos.pnl = net_pnl
        pos.pnl_pct = net_pnl / (pos.entry_price * pos.quantity) * 100
        pos.fees = total_fees
        pos.status = reason

        # Update balance
        self.balance += net_pnl

        # Update strategy performance
        self.strategy.update_performance(
            StrategyType(pos.strategy),
            pos.pnl_pct
        )

        # Save trade
        self.trades.append(pos)
        self.open_position = None

    def _update_equity(self, bar: pd.Series):
        """Update equity curve with unrealized PnL"""
        equity = self.balance

        if self.open_position:
            pos = self.open_position
            if pos.side == "LONG":
                unrealized = (bar['close'] - pos.entry_price) * pos.quantity
            else:
                unrealized = (pos.entry_price - bar['close']) * pos.quantity
            equity += unrealized

        self.equity_curve.append(equity)

        # Track drawdown
        if equity > self.peak_balance:
            self.peak_balance = equity
        else:
            drawdown = self.peak_balance - equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

    def _generate_results(self) -> BacktestResult:
        """Generate comprehensive backtest results"""
        if not self.trades:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=self.balance,
                total_pnl=0,
                total_pnl_pct=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                avg_trade_duration=0,
                best_trade=0,
                worst_trade=0
            )

        # Basic stats
        total_pnl = self.balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        winning = [t for t in self.trades if t.pnl > 0]
        losing = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winning) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = abs(np.mean([t.pnl for t in losing])) if losing else 0

        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown
        max_dd_pct = (self.max_drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0

        # Risk metrics
        returns = np.array(self.daily_returns) if self.daily_returns else np.array([0])
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        calmar = total_pnl_pct / max_dd_pct if max_dd_pct > 0 else 0

        # Trade duration
        durations = []
        for t in self.trades:
            if t.exit_time and t.entry_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # Best/worst trades
        pnls = [t.pnl for t in self.trades]
        best = max(pnls) if pnls else 0
        worst = min(pnls) if pnls else 0

        # Strategy breakdown
        strategy_stats = {}
        for strategy in StrategyType:
            strat_trades = [t for t in self.trades if t.strategy == strategy.value]
            if strat_trades:
                strat_wins = len([t for t in strat_trades if t.pnl > 0])
                strategy_stats[strategy.value] = {
                    'trades': len(strat_trades),
                    'win_rate': strat_wins / len(strat_trades) * 100,
                    'total_pnl': sum(t.pnl for t in strat_trades),
                    'avg_pnl': np.mean([t.pnl for t in strat_trades])
                }

        # Monthly returns
        monthly = {}
        for t in self.trades:
            if t.exit_time:
                month_key = t.exit_time.strftime('%Y-%m')
                monthly[month_key] = monthly.get(month_key, 0) + t.pnl

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_trade_duration=avg_duration,
            best_trade=best,
            worst_trade=worst,
            trades=self.trades,
            equity_curve=self.equity_curve,
            strategy_breakdown=strategy_stats,
            monthly_returns=monthly
        )

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate / 252
        if np.std(excess_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)

    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)


def print_results(result: BacktestResult):
    """Print formatted backtest results"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\n{'PERFORMANCE SUMMARY':^60}")
    print("-"*60)
    print(f"Initial Balance:     ${result.initial_balance:,.2f}")
    print(f"Final Balance:       ${result.final_balance:,.2f}")
    print(f"Total PnL:           ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")

    print(f"\n{'TRADE STATISTICS':^60}")
    print("-"*60)
    print(f"Total Trades:        {result.total_trades}")
    print(f"Winning Trades:      {result.winning_trades}")
    print(f"Losing Trades:       {result.losing_trades}")
    print(f"Win Rate:            {result.win_rate:.1f}%")
    print(f"Avg Win:             ${result.avg_win:.4f}")
    print(f"Avg Loss:            ${result.avg_loss:.4f}")
    print(f"Profit Factor:       {result.profit_factor:.2f}")

    print(f"\n{'RISK METRICS':^60}")
    print("-"*60)
    print(f"Max Drawdown:        ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Sharpe Ratio:        {result.sharpe_ratio:.2f}")
    print(f"Sortino Ratio:       {result.sortino_ratio:.2f}")
    print(f"Calmar Ratio:        {result.calmar_ratio:.2f}")

    print(f"\n{'TRADE DETAILS':^60}")
    print("-"*60)
    print(f"Best Trade:          ${result.best_trade:.4f}")
    print(f"Worst Trade:         ${result.worst_trade:.4f}")
    print(f"Avg Duration:        {result.avg_trade_duration:.1f} hours")

    if result.strategy_breakdown:
        print(f"\n{'STRATEGY BREAKDOWN':^60}")
        print("-"*60)
        for strat, stats in result.strategy_breakdown.items():
            print(f"\n{strat.upper()}:")
            print(f"  Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1f}%, "
                  f"Total PnL: ${stats['total_pnl']:.4f}")

    if result.monthly_returns:
        print(f"\n{'MONTHLY RETURNS':^60}")
        print("-"*60)
        for month, pnl in sorted(result.monthly_returns.items()):
            print(f"  {month}: ${pnl:+.4f}")

    print("\n" + "="*60)


def run_backtest(
    data_file: str,
    symbol: str = None,
    initial_balance: float = 4.5,
    show_progress: bool = True
) -> BacktestResult:
    """
    Convenience function to run a backtest

    Args:
        data_file: Path to CSV file with OHLCV data
        symbol: Trading pair symbol
        initial_balance: Starting balance
        show_progress: Whether to show progress updates
    """
    if symbol is None:
        symbol = os.path.basename(data_file).replace('.csv', '').split('_')[0]

    backtester = Backtester(initial_balance=initial_balance)
    df = backtester.load_data(data_file)

    def progress_cb(pct, balance):
        if show_progress:
            print(f"\rProgress: {pct:.1f}% | Balance: ${balance:.4f}", end="", flush=True)

    print(f"\nRunning backtest for {symbol}...")
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    result = backtester.run(df, symbol, progress_callback=progress_cb if show_progress else None)

    if show_progress:
        print()  # New line after progress

    return result


if __name__ == "__main__":
    # Run backtests on all available data
    data_dir = backtest_config.data_dir
    data_files = [
        os.path.join(data_dir, backtest_config.sol_data),
        os.path.join(data_dir, backtest_config.xrp_data),
        os.path.join(data_dir, backtest_config.btc_data),
    ]

    all_results = {}

    for data_file in data_files:
        if os.path.exists(data_file):
            result = run_backtest(data_file)
            print_results(result)
            symbol = os.path.basename(data_file).split('_')[0]
            all_results[symbol] = result
        else:
            print(f"Data file not found: {data_file}")

    # Summary across all assets
    if all_results:
        print("\n" + "="*60)
        print("COMBINED SUMMARY")
        print("="*60)
        total_pnl = sum(r.total_pnl for r in all_results.values())
        total_trades = sum(r.total_trades for r in all_results.values())
        avg_win_rate = np.mean([r.win_rate for r in all_results.values()])
        print(f"Total PnL across all assets: ${total_pnl:.4f}")
        print(f"Total trades: {total_trades}")
        print(f"Average win rate: {avg_win_rate:.1f}%")
