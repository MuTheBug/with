"""
Improved Backtesting Framework v2
Uses simpler strategy with better risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import os

from strategy_optimized import SimpleStrategy, TradeSignal, SignalType, StrategyType


@dataclass
class Trade:
    """Record of a single trade"""
    id: int
    symbol: str
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    quantity: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    status: str = "OPEN"
    reason: str = ""


@dataclass
class BacktestResult:
    """Backtest results summary"""
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
    best_trade: float
    worst_trade: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """Improved backtester with proper risk management"""

    def __init__(
        self,
        initial_balance: float = 4.5,
        leverage: int = 10,
        risk_per_trade: float = 0.02,  # 2% risk per trade
        commission: float = 0.0004,  # 0.04% fee
        slippage: float = 0.0005  # 0.05% slippage
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage

        self.balance = initial_balance
        self.equity_curve = [initial_balance]
        self.trades: List[Trade] = []
        self.open_position: Optional[Trade] = None
        self.trade_counter = 0

        self.strategy = SimpleStrategy()

        # Tracking
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load OHLCV data from CSV"""
        df = pd.read_csv(filepath)

        # Column mapping
        col_map = {
            'open_time': 'timestamp',
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume'
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Parse timestamp
        ts_col = 'timestamp' if 'timestamp' in df.columns else 'open_time'
        if ts_col in df.columns:
            if df[ts_col].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df[ts_col], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df[ts_col])
            df.set_index('timestamp', inplace=True)

        # Numeric conversion
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        start_idx: int = 250,
        show_progress: bool = True
    ) -> BacktestResult:
        """Run backtest"""
        self._reset()

        total_bars = len(df) - start_idx

        for i in range(start_idx, len(df)):
            window = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            current_time = df.index[i]

            # Check exit conditions for open position
            if self.open_position:
                self._check_exit(current_bar, current_time)

            # Look for new signals if no position
            if not self.open_position:
                signal = self.strategy.analyze(window)
                if signal:
                    self._open_position(signal, current_bar, current_time, symbol)

            # Update equity
            self._update_equity(current_bar)

            # Progress
            if show_progress and (i - start_idx) % 200 == 0:
                pct = (i - start_idx) / total_bars * 100
                print(f"\rProgress: {pct:.1f}% | Balance: ${self.balance:.4f}", end="", flush=True)

        # Close remaining position
        if self.open_position:
            self._close_position(df.iloc[-1], df.index[-1], "End of backtest")

        if show_progress:
            print()

        return self._generate_results()

    def _reset(self):
        """Reset state"""
        self.balance = self.initial_balance
        self.equity_curve = [self.initial_balance]
        self.trades = []
        self.open_position = None
        self.trade_counter = 0
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        self.strategy = SimpleStrategy()

    def _open_position(
        self,
        signal: TradeSignal,
        bar: pd.Series,
        time: pd.Timestamp,
        symbol: str
    ):
        """Open position with proper risk-based sizing"""
        self.trade_counter += 1

        # Apply slippage
        if signal.signal_type == SignalType.LONG:
            entry_price = signal.entry_price * (1 + self.slippage)
        else:
            entry_price = signal.entry_price * (1 - self.slippage)

        # RISK-BASED POSITION SIZING
        # Calculate what we're willing to lose
        risk_amount = self.balance * self.risk_per_trade

        # Calculate stop distance percentage
        stop_distance = abs(entry_price - signal.stop_loss)
        stop_pct = stop_distance / entry_price

        if stop_pct < 0.001:  # Minimum 0.1% stop
            stop_pct = 0.01

        # Position size = Risk Amount / Stop Distance
        # This ensures if stop is hit, we lose exactly risk_amount
        position_value = risk_amount / stop_pct

        # Cap at leveraged account value
        max_position = self.balance * self.leverage * 0.8
        position_value = min(position_value, max_position)

        # Minimum position (Binance requires ~$5 notional)
        min_notional = 5.0
        if position_value < min_notional:
            # Check if we can afford min notional
            if self.balance * self.leverage >= min_notional:
                position_value = min_notional
            else:
                return  # Skip trade

        quantity = position_value / entry_price
        fees = position_value * self.commission

        self.open_position = Trade(
            id=self.trade_counter,
            symbol=symbol,
            side="LONG" if signal.signal_type == SignalType.LONG else "SHORT",
            entry_time=time,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            fees=fees,
            reason=signal.reason
        )

    def _check_exit(self, bar: pd.Series, time: pd.Timestamp):
        """Check exit conditions"""
        if not self.open_position:
            return

        pos = self.open_position

        if pos.side == "LONG":
            # Stop loss hit
            if bar['low'] <= pos.stop_loss:
                exit_price = pos.stop_loss * (1 - self.slippage)
                self._close_position(bar, time, "Stop Loss", exit_price)
                return

            # Take profit hit
            if bar['high'] >= pos.take_profit:
                exit_price = pos.take_profit * (1 - self.slippage)
                self._close_position(bar, time, "Take Profit", exit_price)
                return

        else:  # SHORT
            if bar['high'] >= pos.stop_loss:
                exit_price = pos.stop_loss * (1 + self.slippage)
                self._close_position(bar, time, "Stop Loss", exit_price)
                return

            if bar['low'] <= pos.take_profit:
                exit_price = pos.take_profit * (1 + self.slippage)
                self._close_position(bar, time, "Take Profit", exit_price)
                return

    def _close_position(
        self,
        bar: pd.Series,
        time: pd.Timestamp,
        reason: str,
        exit_price: float = None
    ):
        """Close position and record trade"""
        if not self.open_position:
            return

        pos = self.open_position

        if exit_price is None:
            exit_price = bar['close']
            if pos.side == "LONG":
                exit_price *= (1 - self.slippage)
            else:
                exit_price *= (1 + self.slippage)

        # Calculate PnL
        if pos.side == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Fees
        exit_fees = pos.quantity * exit_price * self.commission
        total_fees = pos.fees + exit_fees
        net_pnl = pnl - total_fees

        # Update trade record
        pos.exit_time = time
        pos.exit_price = exit_price
        pos.pnl = net_pnl
        pos.pnl_pct = (net_pnl / (pos.entry_price * pos.quantity / self.leverage)) * 100
        pos.fees = total_fees
        pos.status = reason

        # Update balance
        self.balance += net_pnl

        # Save trade
        self.trades.append(pos)
        self.open_position = None

    def _update_equity(self, bar: pd.Series):
        """Update equity curve"""
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
            dd = self.peak_balance - equity
            self.max_drawdown = max(self.max_drawdown, dd)

    def _generate_results(self) -> BacktestResult:
        """Generate results summary"""
        if not self.trades:
            return BacktestResult(
                initial_balance=self.initial_balance,
                final_balance=self.balance,
                total_pnl=0, total_pnl_pct=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, max_drawdown=0, max_drawdown_pct=0,
                best_trade=0, worst_trade=0
            )

        total_pnl = self.balance - self.initial_balance
        total_pnl_pct = (total_pnl / self.initial_balance) * 100

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]

        win_rate = len(winners) / len(self.trades) * 100 if self.trades else 0

        avg_win = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0

        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        max_dd_pct = (self.max_drawdown / self.peak_balance) * 100 if self.peak_balance > 0 else 0

        pnls = [t.pnl for t in self.trades]

        return BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_dd_pct,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            trades=self.trades,
            equity_curve=self.equity_curve
        )


def print_results(result: BacktestResult):
    """Print formatted results"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    print(f"\n{'PERFORMANCE SUMMARY':^60}")
    print("-"*60)
    print(f"Initial Balance:     ${result.initial_balance:,.2f}")
    print(f"Final Balance:       ${result.final_balance:,.2f}")
    print(f"Total PnL:           ${result.total_pnl:,.4f} ({result.total_pnl_pct:+.2f}%)")

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
    print(f"Max Drawdown:        ${result.max_drawdown:.4f} ({result.max_drawdown_pct:.1f}%)")
    print(f"Best Trade:          ${result.best_trade:.4f}")
    print(f"Worst Trade:         ${result.worst_trade:.4f}")

    print("\n" + "="*60)


def run_backtest(
    data_file: str,
    initial_balance: float = 4.5,
    leverage: int = 10,
    risk_per_trade: float = 0.02
) -> BacktestResult:
    """Run backtest on a data file"""
    symbol = os.path.basename(data_file).split('_')[0]

    print(f"\n{'='*60}")
    print(f"Running backtest for {symbol}")
    print(f"{'='*60}")

    backtester = Backtester(
        initial_balance=initial_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade
    )

    df = backtester.load_data(data_file)
    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    result = backtester.run(df, symbol)

    return result


if __name__ == "__main__":
    # Data files
    data_dir = "/home/user/with"
    files = [
        "SOLUSDT_1h_1year.csv",
        "XRPUSDT_1h_1year.csv",
        "BTCUSDT_1h_1year.csv"
    ]

    all_results = {}

    for file in files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            result = run_backtest(filepath)
            print_results(result)
            symbol = file.split('_')[0]
            all_results[symbol] = result

    # Combined summary
    if all_results:
        print("\n" + "="*60)
        print("COMBINED SUMMARY")
        print("="*60)

        total_pnl = sum(r.total_pnl for r in all_results.values())
        total_trades = sum(r.total_trades for r in all_results.values())
        avg_win_rate = np.mean([r.win_rate for r in all_results.values()])

        print(f"Total PnL: ${total_pnl:.4f}")
        print(f"Total Trades: {total_trades}")
        print(f"Average Win Rate: {avg_win_rate:.1f}%")

        for symbol, result in all_results.items():
            print(f"\n{symbol}: ${result.total_pnl:+.4f} ({result.total_pnl_pct:+.2f}%) "
                  f"| {result.total_trades} trades | {result.win_rate:.1f}% win rate")
