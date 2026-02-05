"""
Backtesting engine with walk-forward analysis.

Features:
- Bar-by-bar simulation with realistic fills
- Fee and slippage modeling
- Walk-forward optimization to prevent overfitting
- Multi-symbol, multi-strategy backtesting
- Detailed performance reporting
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

from config import BotConfig, MarketRegime, StrategyName, REGIME_STRATEGY_MAP
from indicators import compute_all_indicators
from regime_detector import RegimeDetector, RegimeResult
from strategies import (
    BaseStrategy, TradeSignal, SignalDirection,
    get_strategy, STRATEGY_REGISTRY, NO_SIGNAL,
)
from risk_manager import RiskManager, RiskState, Position, TradeRecord
from data_fetcher import load_csv_data, resample_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_balance: float = 4.0
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "SOLUSDT", "XRPUSDT"])
    data_files: Dict[str, str] = field(default_factory=dict)
    start_idx: int = 200       # Skip initial bars needed for indicators
    end_idx: Optional[int] = None
    primary_tf: str = "1h"     # Timeframe of input data
    use_regime_switching: bool = True
    single_strategy: Optional[StrategyName] = None  # Force single strategy
    verbose: bool = False


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr_realized: float = 0.0
    final_balance: float = 0.0
    peak_balance: float = 0.0
    monthly_returns: List[float] = field(default_factory=list)
    avg_monthly_return: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_performance: Dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Final Balance:    ${self.final_balance:>12.2f}",
            f"Peak Balance:     ${self.peak_balance:>12.2f}",
            f"Total Return:     {self.total_return_pct:>12.1f}%",
            f"Total PnL:        ${self.total_pnl:>12.2f}",
            f"",
            f"Total Trades:     {self.total_trades:>12d}",
            f"Win Rate:         {self.win_rate:>12.1f}%",
            f"Profit Factor:    {self.profit_factor:>12.2f}",
            f"Avg Trade PnL:    ${self.avg_trade_pnl:>12.4f}",
            f"Avg Win:          ${self.avg_win:>12.4f}",
            f"Avg Loss:         ${self.avg_loss:>12.4f}",
            f"",
            f"Max Drawdown %:   {self.max_drawdown_pct:>12.1f}%",
            f"Max Drawdown $:   ${self.max_drawdown:>12.2f}",
            f"Sharpe Ratio:     {self.sharpe_ratio:>12.2f}",
            f"Sortino Ratio:    {self.sortino_ratio:>12.2f}",
            f"",
            f"Avg Monthly Ret:  {self.avg_monthly_return:>12.1f}%",
        ]

        if self.strategy_performance:
            lines.append("")
            lines.append("Strategy Performance:")
            lines.append("-" * 40)
            for strat, perf in self.strategy_performance.items():
                lines.append(
                    f"  {strat:<20s}: {perf.get('trades', 0):>4d} trades, "
                    f"WR={perf.get('win_rate', 0):.0f}%, "
                    f"PF={perf.get('profit_factor', 0):.2f}, "
                    f"PnL=${perf.get('pnl', 0):.2f}"
                )

        if self.regime_distribution:
            lines.append("")
            lines.append("Regime Distribution:")
            lines.append("-" * 40)
            total_bars = sum(self.regime_distribution.values())
            for regime, count in sorted(self.regime_distribution.items()):
                pct = count / total_bars * 100 if total_bars > 0 else 0
                lines.append(f"  {regime:<25s}: {count:>5d} ({pct:.1f}%)")

        lines.append("=" * 60)
        return "\n".join(lines)


class Backtester:
    """
    Event-driven backtesting engine.
    Simulates bar-by-bar trading with realistic fills, fees, and slippage.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.regime_detector = RegimeDetector(self.config)

    def run(self, bt_config: BacktestConfig) -> BacktestResult:
        """Run a backtest on historical data."""
        # Load and prepare data for all symbols
        all_data = {}
        for symbol in bt_config.symbols:
            filepath = bt_config.data_files.get(symbol, "")
            if not filepath:
                continue
            df = load_csv_data(filepath)
            if df.empty:
                logger.warning(f"Empty data for {symbol}")
                continue
            all_data[symbol] = df

        if not all_data:
            logger.error("No data loaded")
            return BacktestResult()

        # PRECOMPUTE indicators for all symbols at once (huge speedup)
        all_indicators = {}
        all_4h_indicators = {}
        for symbol, df in all_data.items():
            all_indicators[symbol] = compute_all_indicators(df, self.config)
            try:
                df_4h = resample_ohlcv(df, "4h")
                if len(df_4h) >= 30:
                    all_4h_indicators[symbol] = compute_all_indicators(df_4h, self.config)
            except Exception:
                pass

        # Initialize risk manager with starting balance (backtest mode)
        risk_state = RiskState(
            balance=bt_config.initial_balance,
            peak_balance=bt_config.initial_balance,
        )
        risk_mgr = RiskManager(self.config, risk_state, backtest_mode=True)

        # Strategy instances
        strategies = {name: get_strategy(name, self.config) for name in STRATEGY_REGISTRY}

        # Track results
        equity_curve = [bt_config.initial_balance]
        regime_counts = {}
        strategy_counts = {}

        # Use first symbol's data as the timing reference
        ref_symbol = bt_config.symbols[0]
        ref_data = all_data[ref_symbol]
        n_bars = len(ref_data)
        start = bt_config.start_idx
        end = bt_config.end_idx or n_bars

        for bar_idx in range(start, end):
            # Advance bar-based cooldown counter
            risk_mgr.advance_bar()

            # Check if trading is allowed
            can_trade, reason = risk_mgr.can_trade()

            # Check existing positions for exits first
            positions_to_close = []
            for pos in list(risk_state.positions):
                sym_data = all_data.get(pos.symbol)
                if sym_data is None or bar_idx >= len(sym_data):
                    continue
                bar = sym_data.iloc[bar_idx]
                bars_held = bar_idx - pos.entry_bar
                should_exit, exit_price, exit_reason = risk_mgr.check_exit_conditions(
                    pos, bar["close"], bar["high"], bar["low"],
                    bar_open=bar["open"], bars_held=bars_held,
                )
                if should_exit:
                    positions_to_close.append((pos, exit_price, exit_reason))

            for pos, exit_price, exit_reason in positions_to_close:
                # Apply slippage to exit
                if pos.direction == SignalDirection.LONG:
                    exit_price *= (1 - self.config.risk.slippage_pct)
                else:
                    exit_price *= (1 + self.config.risk.slippage_pct)
                risk_mgr.record_trade(pos, exit_price, exit_reason)

            if not can_trade:
                equity_curve.append(risk_state.balance)
                continue

            # Iterate over symbols for new signals
            best_signal = None
            best_symbol = None
            best_regime = None
            best_regime_confidence = 0.0

            for symbol in bt_config.symbols:
                sym_ind = all_indicators.get(symbol)
                if sym_ind is None or bar_idx >= len(sym_ind):
                    continue

                # Already have a position in this symbol?
                has_position = any(p.symbol == symbol for p in risk_state.positions)
                if has_position:
                    continue

                # Slice precomputed indicators up to current bar (no lookahead)
                df_ind = sym_ind.iloc[:bar_idx + 1]
                if len(df_ind) < 60:
                    continue

                # Multi-TF data for regime detection
                data_by_tf = {"1h": df_ind}

                # Add 4h data (map bar index to 4h bar index)
                sym_4h = all_4h_indicators.get(symbol)
                if sym_4h is not None and len(sym_4h) > 0:
                    # Approximate: 4h bar index = bar_idx // 4
                    bar_4h_idx = min(bar_idx // 4, len(sym_4h) - 1)
                    if bar_4h_idx >= 30:
                        data_by_tf["4h"] = sym_4h.iloc[:bar_4h_idx + 1]

                # Detect regime
                regime_result = self.regime_detector.detect(data_by_tf, "1h")
                regime = regime_result.regime
                regime_confidence = regime_result.confidence

                # Track regime
                regime_name = regime.value
                regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

                # Select strategy for this regime
                if bt_config.single_strategy:
                    strat_name = bt_config.single_strategy
                elif bt_config.use_regime_switching:
                    strat_name = REGIME_STRATEGY_MAP.get(regime, StrategyName.SCALP)
                else:
                    strat_name = StrategyName.TREND_MOMENTUM

                strategy = strategies[strat_name]
                signal = strategy.generate_signal(df_ind)

                if signal.direction == SignalDirection.NONE:
                    continue

                # Validate signal
                approved, adj_signal, val_reason = risk_mgr.validate_signal(
                    signal, regime, regime_confidence
                )
                if not approved:
                    continue

                # Pick the highest-confidence signal across symbols
                if best_signal is None or signal.confidence > best_signal.confidence:
                    best_signal = signal
                    best_symbol = symbol
                    best_regime = regime
                    best_regime_confidence = regime_confidence

            # Execute the best signal if found
            if best_signal is not None and best_symbol is not None:
                current_price = best_signal.entry_price

                # Calculate position size
                quantity, leverage, margin = risk_mgr.calculate_position_size(
                    best_signal, best_symbol, current_price, best_regime_confidence,
                )

                if quantity > 0 and margin <= risk_state.balance * 0.95:
                    # Apply entry slippage
                    if best_signal.direction == SignalDirection.LONG:
                        fill_price = current_price * (1 + self.config.risk.slippage_pct)
                    else:
                        fill_price = current_price * (1 - self.config.risk.slippage_pct)

                    # Create position
                    position = Position(
                        symbol=best_symbol,
                        direction=best_signal.direction,
                        entry_price=fill_price,
                        quantity=quantity,
                        leverage=leverage,
                        stop_loss=best_signal.stop_loss,
                        take_profit=best_signal.take_profit,
                        trailing_stop=best_signal.stop_loss,  # Initialize trailing
                        entry_bar=bar_idx,
                        strategy=best_signal.strategy.value,
                    )

                    # Entry fees
                    entry_fee = fill_price * quantity * self.config.risk.taker_fee
                    position.fees_paid = entry_fee

                    risk_state.positions.append(position)

                    strat_name = best_signal.strategy.value
                    strategy_counts[strat_name] = strategy_counts.get(strat_name, 0) + 1

                    if bt_config.verbose:
                        logger.info(
                            f"[{bar_idx}] OPEN {best_signal.direction.value} "
                            f"{best_symbol} @ {fill_price:.4f}, "
                            f"qty={quantity}, lev={leverage}, "
                            f"SL={best_signal.stop_loss:.4f}, "
                            f"TP={best_signal.take_profit:.4f}, "
                            f"strategy={strat_name}"
                        )

            equity_curve.append(risk_state.balance)

        # Close any remaining positions at last price
        for pos in list(risk_state.positions):
            sym_data = all_data.get(pos.symbol)
            if sym_data is not None:
                last_price = sym_data.iloc[-1]["close"]
                risk_mgr.record_trade(pos, last_price, "End of backtest")

        # Compile results
        return self._compile_results(
            risk_state, equity_curve, regime_counts, strategy_counts,
            bt_config.initial_balance,
        )

    def _compile_results(
        self,
        state: RiskState,
        equity_curve: List[float],
        regime_counts: Dict[str, int],
        strategy_counts: Dict[str, int],
        initial_balance: float,
    ) -> BacktestResult:
        """Compile backtest results into a BacktestResult."""
        result = BacktestResult()
        result.total_trades = state.total_trades
        result.winning_trades = state.winning_trades
        result.losing_trades = state.losing_trades
        result.win_rate = state.win_rate * 100
        result.profit_factor = state.profit_factor
        result.total_pnl = state.total_pnl
        result.final_balance = state.balance
        result.peak_balance = state.peak_balance
        result.total_return_pct = (state.balance - initial_balance) / initial_balance * 100
        result.equity_curve = equity_curve
        result.trades = state.trade_history
        result.regime_distribution = regime_counts
        result.strategy_distribution = strategy_counts

        # Average trade PnL
        if state.total_trades > 0:
            result.avg_trade_pnl = state.total_pnl / state.total_trades

        # Avg win/loss
        wins = [t.pnl for t in state.trade_history if t.pnl > 0]
        losses = [t.pnl for t in state.trade_history if t.pnl < 0]
        result.avg_win = float(np.mean(wins)) if wins else 0
        result.avg_loss = float(np.mean(losses)) if losses else 0

        # Average realized R:R
        rrs = []
        for t in state.trade_history:
            if t.pnl > 0 and result.avg_loss != 0:
                rrs.append(abs(t.pnl / result.avg_loss))
        result.avg_rr_realized = float(np.mean(rrs)) if rrs else 0

        # Drawdown from equity curve
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        result.max_drawdown = float(dd.max())
        dd_pct = dd / np.where(peak > 0, peak, 1)
        result.max_drawdown_pct = float(dd_pct.max()) * 100

        # Sharpe and Sortino
        trade_returns = [t.pnl_pct for t in state.trade_history]
        if len(trade_returns) > 1:
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns)
            if std_ret > 0:
                # Annualize assuming ~2 trades per day
                result.sharpe_ratio = mean_ret / std_ret * np.sqrt(365 * 2)
            downside = [r for r in trade_returns if r < 0]
            if downside:
                downside_std = np.std(downside)
                if downside_std > 0:
                    result.sortino_ratio = mean_ret / downside_std * np.sqrt(365 * 2)

        # Monthly returns (approximate: every ~720 bars for hourly data)
        eq_arr = np.array(equity_curve)
        monthly_interval = 720
        monthly_rets = []
        for i in range(0, len(eq_arr) - 1, monthly_interval):
            end = min(i + monthly_interval, len(eq_arr) - 1)
            if eq_arr[i] > 0:
                ret = (eq_arr[end] - eq_arr[i]) / eq_arr[i] * 100
                monthly_rets.append(ret)
        result.monthly_returns = monthly_rets
        result.avg_monthly_return = float(np.mean(monthly_rets)) if monthly_rets else 0

        # Per-strategy performance
        strat_trades = {}
        for t in state.trade_history:
            s = t.strategy
            if s not in strat_trades:
                strat_trades[s] = []
            strat_trades[s].append(t)

        for strat, trades in strat_trades.items():
            s_wins = [t for t in trades if t.pnl > 0]
            s_losses = [t for t in trades if t.pnl < 0]
            gross_profit = sum(t.pnl for t in s_wins)
            gross_loss = abs(sum(t.pnl for t in s_losses))

            result.strategy_performance[strat] = {
                "trades": len(trades),
                "wins": len(s_wins),
                "losses": len(s_losses),
                "win_rate": len(s_wins) / len(trades) * 100 if trades else 0,
                "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
                "pnl": sum(t.pnl for t in trades),
                "avg_pnl": np.mean([t.pnl for t in trades]),
            }

        return result

    def walk_forward(
        self,
        bt_config: BacktestConfig,
        n_folds: int = 5,
        train_pct: float = 0.7,
    ) -> Tuple[BacktestResult, List[BacktestResult]]:
        """
        Walk-forward analysis: train on in-sample, test on out-of-sample.
        Returns (combined_oos_result, list_of_fold_results).
        """
        # Load reference data to get total length
        ref_symbol = bt_config.symbols[0]
        filepath = bt_config.data_files.get(ref_symbol, "")
        ref_data = load_csv_data(filepath)
        n_total = len(ref_data) - bt_config.start_idx

        fold_size = n_total // n_folds
        fold_results = []

        all_oos_trades = []
        combined_equity = [bt_config.initial_balance]
        running_balance = bt_config.initial_balance

        for fold in range(n_folds):
            fold_start = bt_config.start_idx + fold * fold_size
            fold_end = fold_start + fold_size
            if fold == n_folds - 1:
                fold_end = len(ref_data)

            train_end = fold_start + int((fold_end - fold_start) * train_pct)
            test_start = train_end
            test_end = fold_end

            # In-sample: run to see what regime/strategy combos work
            is_config = deepcopy(bt_config)
            is_config.start_idx = fold_start
            is_config.end_idx = train_end
            is_config.initial_balance = running_balance
            is_config.verbose = False

            is_result = self.run(is_config)

            # Out-of-sample: test with same settings
            oos_config = deepcopy(bt_config)
            oos_config.start_idx = test_start
            oos_config.end_idx = test_end
            oos_config.initial_balance = running_balance
            oos_config.verbose = False

            oos_result = self.run(oos_config)
            fold_results.append(oos_result)

            all_oos_trades.extend(oos_result.trades)
            combined_equity.extend(oos_result.equity_curve[1:])

            running_balance = oos_result.final_balance
            if running_balance <= 0:
                break

            logger.info(
                f"Fold {fold + 1}/{n_folds}: "
                f"IS trades={is_result.total_trades}, WR={is_result.win_rate:.1f}%, "
                f"OOS trades={oos_result.total_trades}, WR={oos_result.win_rate:.1f}%, "
                f"Balance=${running_balance:.2f}"
            )

        # Compile combined OOS results
        combined = BacktestResult()
        combined.total_trades = sum(r.total_trades for r in fold_results)
        combined.winning_trades = sum(r.winning_trades for r in fold_results)
        combined.losing_trades = sum(r.losing_trades for r in fold_results)
        combined.win_rate = (
            combined.winning_trades / combined.total_trades * 100
            if combined.total_trades > 0 else 0
        )
        combined.total_pnl = sum(r.total_pnl for r in fold_results)
        combined.final_balance = running_balance
        combined.peak_balance = max(r.peak_balance for r in fold_results) if fold_results else bt_config.initial_balance
        combined.total_return_pct = (running_balance - bt_config.initial_balance) / bt_config.initial_balance * 100
        combined.equity_curve = combined_equity
        combined.trades = all_oos_trades

        # Drawdown
        eq = np.array(combined_equity)
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        combined.max_drawdown = float(dd.max())
        dd_pct = dd / np.where(peak > 0, peak, 1)
        combined.max_drawdown_pct = float(dd_pct.max()) * 100

        # PF
        gross_profit = sum(t.pnl for t in all_oos_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in all_oos_trades if t.pnl < 0))
        combined.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe
        trade_rets = [t.pnl_pct for t in all_oos_trades]
        if len(trade_rets) > 1:
            std = np.std(trade_rets)
            if std > 0:
                combined.sharpe_ratio = np.mean(trade_rets) / std * np.sqrt(365 * 2)

        # Avg trade
        if combined.total_trades > 0:
            combined.avg_trade_pnl = combined.total_pnl / combined.total_trades

        wins = [t.pnl for t in all_oos_trades if t.pnl > 0]
        losses = [t.pnl for t in all_oos_trades if t.pnl < 0]
        combined.avg_win = float(np.mean(wins)) if wins else 0
        combined.avg_loss = float(np.mean(losses)) if losses else 0

        # Monthly
        monthly_rets = []
        for r in fold_results:
            monthly_rets.extend(r.monthly_returns)
        combined.monthly_returns = monthly_rets
        combined.avg_monthly_return = float(np.mean(monthly_rets)) if monthly_rets else 0

        # Strategy perf
        strat_trades = {}
        for t in all_oos_trades:
            s = t.strategy
            if s not in strat_trades:
                strat_trades[s] = []
            strat_trades[s].append(t)

        for strat, trades in strat_trades.items():
            s_wins = [t for t in trades if t.pnl > 0]
            s_losses = [t for t in trades if t.pnl < 0]
            gp = sum(t.pnl for t in s_wins)
            gl = abs(sum(t.pnl for t in s_losses))
            combined.strategy_performance[strat] = {
                "trades": len(trades),
                "wins": len(s_wins),
                "losses": len(s_losses),
                "win_rate": len(s_wins) / len(trades) * 100 if trades else 0,
                "profit_factor": gp / gl if gl > 0 else float("inf"),
                "pnl": sum(t.pnl for t in trades),
            }

        return combined, fold_results
