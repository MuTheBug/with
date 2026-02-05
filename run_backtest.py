#!/usr/bin/env python3
"""
Backtest runner with walk-forward analysis.

Runs comprehensive backtests on historical data to validate the trading system.
Tests:
  1. Full regime-switching backtest on each symbol
  2. Per-strategy isolated backtests
  3. Walk-forward out-of-sample validation
  4. Multi-symbol combined backtest

Usage:
  python run_backtest.py                  # Full backtest suite
  python run_backtest.py --quick          # Quick single run
  python run_backtest.py --strategy trend_momentum  # Test single strategy
"""

import sys
import os
import logging
import argparse
import numpy as np
from typing import Dict

from config import BotConfig, StrategyName
from backtester import Backtester, BacktestConfig, BacktestResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Data files
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = {
    "BTCUSDT": os.path.join(DATA_DIR, "BTCUSDT_1h_1year.csv"),
    "SOLUSDT": os.path.join(DATA_DIR, "SOLUSDT_1h_1year.csv"),
    "XRPUSDT": os.path.join(DATA_DIR, "XRPUSDT_1h_1year.csv"),
}


def run_single_backtest(
    config: BotConfig,
    symbols: list,
    initial_balance: float = 4.0,
    strategy: StrategyName = None,
    label: str = "",
    verbose: bool = False,
) -> BacktestResult:
    """Run a single backtest configuration."""
    bt_config = BacktestConfig(
        initial_balance=initial_balance,
        symbols=symbols,
        data_files={s: DATA_FILES[s] for s in symbols if s in DATA_FILES},
        start_idx=210,
        primary_tf="1h",
        use_regime_switching=(strategy is None),
        single_strategy=strategy,
        verbose=verbose,
    )

    backtester = Backtester(config)
    result = backtester.run(bt_config)

    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")
    print(result.summary())
    return result


def run_walk_forward(
    config: BotConfig,
    symbols: list,
    initial_balance: float = 4.0,
    n_folds: int = 5,
) -> BacktestResult:
    """Run walk-forward analysis."""
    bt_config = BacktestConfig(
        initial_balance=initial_balance,
        symbols=symbols,
        data_files={s: DATA_FILES[s] for s in symbols if s in DATA_FILES},
        start_idx=210,
        primary_tf="1h",
        use_regime_switching=True,
    )

    backtester = Backtester(config)
    combined, folds = backtester.walk_forward(bt_config, n_folds=n_folds)

    print(f"\n{'=' * 60}")
    print(f"  WALK-FORWARD ANALYSIS ({n_folds} folds)")
    print(f"{'=' * 60}")

    for i, fold in enumerate(folds):
        print(f"\n  Fold {i + 1}: trades={fold.total_trades}, "
              f"WR={fold.win_rate:.1f}%, PF={fold.profit_factor:.2f}, "
              f"Return={fold.total_return_pct:.1f}%, "
              f"MaxDD={fold.max_drawdown_pct:.1f}%")

    print(f"\n  Combined OOS Results:")
    print(combined.summary())
    return combined


def run_full_suite():
    """Run the complete backtest suite."""
    config = BotConfig()
    all_symbols = ["BTCUSDT", "SOLUSDT", "XRPUSDT"]

    print("\n" + "#" * 70)
    print("#  BINANCE FUTURES TRADING BOT - BACKTEST SUITE")
    print("#" * 70)

    # 1. Full regime-switching backtest (all symbols, $4 start)
    print("\n\n>>> TEST 1: Full System - Regime Switching - All Symbols ($4 start)")
    full_result = run_single_backtest(
        config, all_symbols, 4.0,
        label="FULL SYSTEM: Regime-Switching, 3 Symbols, $4 Start",
    )

    # 2. Per-symbol backtests
    print("\n\n>>> TEST 2: Per-Symbol Backtests ($4 start)")
    symbol_results = {}
    for symbol in all_symbols:
        result = run_single_backtest(
            config, [symbol], 4.0,
            label=f"SINGLE SYMBOL: {symbol} ($4 Start)",
        )
        symbol_results[symbol] = result

    # 3. Per-strategy isolated backtests
    print("\n\n>>> TEST 3: Per-Strategy Backtests (All Symbols, $4 start)")
    strategy_results = {}
    for strat_name in StrategyName:
        result = run_single_backtest(
            config, all_symbols, 4.0,
            strategy=strat_name,
            label=f"STRATEGY: {strat_name.value}",
        )
        strategy_results[strat_name] = result

    # 4. Walk-forward analysis
    print("\n\n>>> TEST 4: Walk-Forward Analysis (5-fold)")
    wf_result = run_walk_forward(config, all_symbols, 4.0, n_folds=5)

    # 5. Different starting balances
    print("\n\n>>> TEST 5: Different Starting Balances")
    for balance in [4.0, 10.0, 20.0, 100.0]:
        run_single_backtest(
            config, all_symbols, balance,
            label=f"BALANCE TEST: ${balance:.0f} Start",
        )

    # Summary comparison
    print("\n\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Test':<40s} {'Trades':>7s} {'WR%':>7s} {'PF':>7s} "
          f"{'Return%':>10s} {'MaxDD%':>8s} {'Sharpe':>7s} {'Final$':>10s}")
    print("-" * 100)

    def row(name, r):
        print(f"{name:<40s} {r.total_trades:>7d} {r.win_rate:>7.1f} "
              f"{r.profit_factor:>7.2f} {r.total_return_pct:>10.1f} "
              f"{r.max_drawdown_pct:>8.1f} {r.sharpe_ratio:>7.2f} "
              f"{r.final_balance:>10.2f}")

    row("Full System (regime switch)", full_result)
    for sym, r in symbol_results.items():
        row(f"  {sym}", r)
    for strat, r in strategy_results.items():
        row(f"  Strategy: {strat.value}", r)
    row("Walk-Forward OOS", wf_result)

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Trading Bot Backtester")
    parser.add_argument("--quick", action="store_true", help="Quick single run")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Test single strategy: trend_momentum, mean_reversion, breakout, scalp, momentum_surf")
    parser.add_argument("--symbol", type=str, default=None, help="Single symbol")
    parser.add_argument("--balance", type=float, default=4.0, help="Starting balance")
    parser.add_argument("--walk-forward", action="store_true", help="Walk-forward only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    config = BotConfig()

    if args.quick:
        symbols = [args.symbol] if args.symbol else ["BTCUSDT", "SOLUSDT", "XRPUSDT"]
        strat = None
        if args.strategy:
            strat = StrategyName(args.strategy)
        run_single_backtest(
            config, symbols, args.balance,
            strategy=strat,
            label="Quick Backtest",
            verbose=args.verbose,
        )
    elif args.walk_forward:
        symbols = [args.symbol] if args.symbol else ["BTCUSDT", "SOLUSDT", "XRPUSDT"]
        run_walk_forward(config, symbols, args.balance)
    else:
        run_full_suite()


if __name__ == "__main__":
    main()
