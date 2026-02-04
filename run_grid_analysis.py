#!/usr/bin/env python3
"""
MAIN RUNNER: FUTURES GRID TRADING SYSTEM
=========================================
Complete analysis runner that executes all trading systems
and generates comprehensive reports.

Run this script to:
1. Test standard grid trading system
2. Test ultra-optimized system
3. Run full optimization
4. Generate comparison reports
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Import our modules
from grid_trading_system import (
    GridConfig, GridBacktester, GridOptimizer, run_full_analysis
)
from ultra_optimized_grid import (
    UltraGridConfig, UltraBacktester, run_ultra_analysis, optimize_ultra_config
)
from advanced_grid_features import (
    FeatureEngineering, MarketRegimeDetector, SignalGenerator,
    AdaptiveGridStrategy, run_advanced_analysis
)


def print_header():
    """Print system header"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "   FUTURES NEUTRAL GRID TRADING SYSTEM".center(68) + "║")
    print("║" + "   Ultra-Low Drawdown | Small Capital Optimized".center(68) + "║")
    print("║" + "   Starting Capital: $4.00".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    print("\n")


def load_all_data() -> dict:
    """Load all available data files"""
    data_files = {
        'BTC/USDT': 'BTCUSDT_1h_1year.csv',
        'SOL/USDT': 'SOLUSDT_1h_1year.csv',
        'XRP/USDT': 'XRPUSDT_1h_1year.csv',
    }

    loaded_data = {}

    for symbol, filename in data_files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df.columns = df.columns.str.lower()

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            else:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            loaded_data[symbol] = df
            print(f"  ✓ Loaded {symbol}: {len(df)} candles")
        else:
            print(f"  ✗ File not found: {filename}")

    return loaded_data


def run_standard_grid_analysis(data: dict) -> dict:
    """Run standard grid trading analysis"""
    print("\n" + "="*70)
    print("  PHASE 1: STANDARD GRID TRADING ANALYSIS")
    print("="*70)

    results = {}

    for symbol, df in data.items():
        print(f"\n  Analyzing {symbol}...")

        # Define optimized parameters for small accounts
        config = GridConfig(
            initial_capital=4.0,
            num_grids=15,
            grid_spacing_pct=0.3,
            use_dynamic_spacing=True,
            base_leverage=25,
            max_leverage=75,
            min_leverage=10,
            max_drawdown_pct=12,
            position_size_pct=2.5,
            stop_loss_pct=6.0,
            compound_profits=True,
            compound_threshold=3.0,
        )

        backtester = GridBacktester(config)
        prepared_df = backtester.prepare_indicators(df.copy())
        result = backtester.run_backtest(prepared_df)

        results[symbol] = {
            'config': config,
            'results': result,
            'backtester': backtester
        }

        print(f"    Return: {result['total_return_pct']:.2f}%")
        print(f"    Max DD: {result['max_drawdown_pct']:.2f}%")
        print(f"    Win Rate: {result['win_rate_pct']:.2f}%")
        print(f"    Trades: {result['total_trades']}")

    return results


def run_ultra_grid_analysis(data: dict) -> dict:
    """Run ultra-optimized grid analysis"""
    print("\n" + "="*70)
    print("  PHASE 2: ULTRA-OPTIMIZED GRID ANALYSIS")
    print("="*70)

    results = {}

    for symbol, df in data.items():
        print(f"\n  Optimizing {symbol}...")

        # Find best configuration
        best_config = optimize_ultra_config(df.copy(), n_iterations=30)

        if best_config:
            backtester = UltraBacktester(best_config)
            result = backtester.run_backtest(df.copy())

            results[symbol] = {
                'config': best_config,
                'results': result,
                'backtester': backtester
            }

            print(f"    Return: {result['total_return_pct']:.2f}%")
            print(f"    Max DD: {result['max_drawdown_pct']:.2f}%")
            print(f"    Win Rate: {result['win_rate_pct']:.2f}%")
            print(f"    Trades: {result['total_trades']}")
        else:
            print(f"    Optimization failed for {symbol}")

    return results


def run_advanced_market_analysis(data: dict) -> dict:
    """Run advanced market analysis"""
    print("\n" + "="*70)
    print("  PHASE 3: ADVANCED MARKET ANALYSIS")
    print("="*70)

    results = {}

    for symbol, df in data.items():
        print(f"\n  Analyzing market conditions for {symbol}...")

        analysis = run_advanced_analysis(df.copy(), symbol)
        results[symbol] = analysis

    return results


def generate_final_report(standard_results: dict, ultra_results: dict, market_analysis: dict):
    """Generate comprehensive final report"""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + "  COMPREHENSIVE ANALYSIS REPORT".center(68) + "║")
    print("╚" + "═"*68 + "╝")

    # Standard Grid Results
    print("\n" + "─"*70)
    print("  STANDARD GRID TRADING RESULTS")
    print("─"*70)
    print(f"  {'Asset':<12} {'Return %':<12} {'Max DD %':<12} {'Win Rate':<12} {'Sharpe':<10}")
    print("  " + "-"*58)

    for symbol, data in standard_results.items():
        r = data['results']
        print(f"  {symbol:<12} {r['total_return_pct']:<12.2f} {r['max_drawdown_pct']:<12.2f} {r['win_rate_pct']:<12.2f} {r['sharpe_ratio']:<10.2f}")

    # Ultra Grid Results
    print("\n" + "─"*70)
    print("  ULTRA-OPTIMIZED GRID RESULTS")
    print("─"*70)
    print(f"  {'Asset':<12} {'Return %':<12} {'Max DD %':<12} {'Win Rate':<12} {'Sharpe':<10}")
    print("  " + "-"*58)

    for symbol, data in ultra_results.items():
        r = data['results']
        print(f"  {symbol:<12} {r['total_return_pct']:<12.2f} {r['max_drawdown_pct']:<12.2f} {r['win_rate_pct']:<12.2f} {r['sharpe_ratio']:<10.2f}")

    # Best Performing Setup
    print("\n" + "─"*70)
    print("  BEST PERFORMING CONFIGURATIONS")
    print("─"*70)

    # Find best from each category
    best_standard = max(standard_results.items(), key=lambda x: x[1]['results']['total_return_pct'])
    best_ultra = max(ultra_results.items(), key=lambda x: x[1]['results']['total_return_pct'])

    print(f"\n  Best Standard Grid: {best_standard[0]}")
    print(f"    Return: {best_standard[1]['results']['total_return_pct']:.2f}%")
    print(f"    Final Capital: ${best_standard[1]['results']['final_capital']:.2f}")

    print(f"\n  Best Ultra Grid: {best_ultra[0]}")
    print(f"    Return: {best_ultra[1]['results']['total_return_pct']:.2f}%")
    print(f"    Final Capital: ${best_ultra[1]['results']['final_capital']:.2f}")

    # Recommended Configuration
    print("\n" + "─"*70)
    print("  RECOMMENDED CONFIGURATION FOR LIVE TRADING")
    print("─"*70)

    # Use the best performing config
    if best_ultra[1]['results']['total_return_pct'] > best_standard[1]['results']['total_return_pct']:
        best_overall = best_ultra
        config_type = "Ultra-Optimized"
    else:
        best_overall = best_standard
        config_type = "Standard"

    print(f"\n  Recommended System: {config_type}")
    print(f"  Recommended Asset: {best_overall[0]}")
    print(f"\n  Configuration Parameters:")

    if config_type == "Ultra-Optimized":
        cfg = best_overall[1]['config']
        print(f"    Inner Grids: {cfg.inner_grids}")
        print(f"    Outer Grids: {cfg.outer_grids}")
        print(f"    Inner Spacing: {cfg.inner_spacing_pct:.2f}%")
        print(f"    Outer Spacing: {cfg.outer_spacing_pct:.2f}%")
        print(f"    Base Leverage: {cfg.base_leverage}x")
        print(f"    Max Leverage: {cfg.max_leverage}x")
        print(f"    Max Drawdown: {cfg.max_drawdown}%")
        print(f"    Stop Loss: {cfg.position_stop_loss:.1f}%")
        print(f"    Take Profit: {cfg.base_take_profit:.2f}%")
    else:
        cfg = best_overall[1]['config']
        print(f"    Num Grids: {cfg.num_grids}")
        print(f"    Grid Spacing: {cfg.grid_spacing_pct:.2f}%")
        print(f"    Base Leverage: {cfg.base_leverage}x")
        print(f"    Max Leverage: {cfg.max_leverage}x")
        print(f"    Max Drawdown: {cfg.max_drawdown_pct}%")
        print(f"    Stop Loss: {cfg.stop_loss_pct:.1f}%")

    # Growth Projection
    print("\n" + "─"*70)
    print("  PROJECTED GROWTH (Based on Backtest Performance)")
    print("─"*70)

    daily_return = best_overall[1]['results']['total_return_pct'] / 365
    starting_cap = 4.0

    print(f"\n  Starting Capital: ${starting_cap:.2f}")
    print(f"  Estimated Daily Return: {daily_return:.2f}%")
    print(f"\n  Projected Capital:")

    capital = starting_cap
    milestones = [7, 14, 30, 60, 90, 180, 365]

    for days in milestones:
        projected = starting_cap * ((1 + daily_return/100) ** days)
        print(f"    Day {days:>3}: ${projected:>12.2f}")

    # Risk Warning
    print("\n" + "─"*70)
    print("  ⚠️ IMPORTANT RISK DISCLOSURES")
    print("─"*70)
    print("""
  1. Past performance does not guarantee future results
  2. Futures trading involves significant risk of loss
  3. High leverage amplifies both gains AND losses
  4. Never risk more than you can afford to lose
  5. Backtests may not account for slippage, latency, or liquidity
  6. Market conditions can change rapidly
  7. Start with paper trading before using real capital
    """)

    print("\n" + "═"*70)
    print("  Analysis Complete - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("═"*70 + "\n")


def main():
    """Main execution function"""
    print_header()

    print("Loading market data...")
    data = load_all_data()

    if not data:
        print("No data files found. Please ensure CSV files are present.")
        return

    # Run all analyses
    standard_results = run_standard_grid_analysis(data)
    ultra_results = run_ultra_grid_analysis(data)
    market_analysis = run_advanced_market_analysis(data)

    # Generate final report
    generate_final_report(standard_results, ultra_results, market_analysis)

    return {
        'standard': standard_results,
        'ultra': ultra_results,
        'market': market_analysis
    }


if __name__ == "__main__":
    results = main()
