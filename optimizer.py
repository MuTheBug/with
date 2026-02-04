"""
Strategy Optimizer
Uses grid search and genetic algorithms to optimize strategy parameters
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from itertools import product
import random
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

from config import strategy_config, backtest_config
from backtest import Backtester, BacktestResult, print_results


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    best_result: BacktestResult
    all_results: List[Tuple[Dict, float]] = field(default_factory=list)
    iterations: int = 0


# Parameter search space for optimization
PARAM_GRID = {
    # EMA periods
    'fast_ema': [5, 8, 10, 12],
    'slow_ema': [15, 21, 26, 30],

    # RSI settings
    'rsi_period': [10, 14, 20],
    'rsi_oversold': [25, 30, 35],
    'rsi_overbought': [65, 70, 75],

    # ATR multipliers
    'atr_multiplier_sl': [1.0, 1.5, 2.0, 2.5],
    'atr_multiplier_tp': [2.0, 2.5, 3.0, 3.5],

    # Position sizing
    'position_size_pct': [0.7, 0.8, 0.9, 1.0],
    'max_risk_per_trade': [0.02, 0.03, 0.04, 0.05],

    # Trailing stop
    'trailing_stop_pct': [0.01, 0.015, 0.02, 0.025],
}

# More granular search space for fine-tuning
PARAM_GRID_FINE = {
    'fast_ema': [7, 8, 9],
    'slow_ema': [19, 21, 23],
    'rsi_period': [12, 14, 16],
    'atr_multiplier_sl': [1.3, 1.5, 1.7],
    'atr_multiplier_tp': [2.3, 2.5, 2.7],
}


def calculate_score(result: BacktestResult, weights: Dict[str, float] = None) -> float:
    """
    Calculate optimization score from backtest result

    Scoring prioritizes:
    1. Profit factor (most important for small accounts)
    2. Win rate (need consistency)
    3. Total return (growth potential)
    4. Low drawdown (preservation)
    """
    weights = weights or {
        'profit_factor': 0.30,
        'win_rate': 0.25,
        'return': 0.25,
        'drawdown_penalty': 0.20
    }

    if result.total_trades < 10:
        return -1000  # Not enough trades

    # Profit factor score (capped at 3)
    pf_score = min(result.profit_factor, 3) / 3 * 100

    # Win rate score
    wr_score = result.win_rate

    # Return score (using log to prevent extreme values)
    if result.total_pnl_pct > 0:
        return_score = min(np.log1p(result.total_pnl_pct) * 20, 100)
    else:
        return_score = max(result.total_pnl_pct, -100)

    # Drawdown penalty
    dd_penalty = min(result.max_drawdown_pct, 50)

    # Combined score
    score = (
        weights['profit_factor'] * pf_score +
        weights['win_rate'] * wr_score +
        weights['return'] * return_score -
        weights['drawdown_penalty'] * dd_penalty
    )

    # Bonus for consistent performance
    if result.sharpe_ratio > 1:
        score += 10
    if result.sortino_ratio > 1.5:
        score += 10

    return score


def run_single_backtest(
    data_file: str,
    params: Dict[str, Any],
    initial_balance: float = 4.5
) -> BacktestResult:
    """Run a single backtest with given parameters"""

    # Create modified config
    from config import StrategyConfig
    config = StrategyConfig()

    # Apply parameters
    for key, value in params.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Import strategy fresh to use new config
    from strategy import SmallAccountStrategy
    strategy = SmallAccountStrategy(config)

    # Run backtest
    backtester = Backtester(initial_balance=initial_balance)
    backtester.strategy = strategy

    df = backtester.load_data(data_file)
    result = backtester.run(df, progress_callback=None)

    return result


def grid_search(
    data_files: List[str],
    param_grid: Dict[str, List] = None,
    max_combinations: int = 500,
    initial_balance: float = 4.5
) -> OptimizationResult:
    """
    Grid search optimization

    Tests combinations of parameters to find optimal settings
    """
    param_grid = param_grid or PARAM_GRID

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combinations = list(product(*values))

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        print(f"Sampling {max_combinations} from {len(all_combinations)} combinations")
        all_combinations = random.sample(all_combinations, max_combinations)

    print(f"Testing {len(all_combinations)} parameter combinations...")

    best_params = None
    best_score = -float('inf')
    best_result = None
    all_results = []

    for i, combo in enumerate(all_combinations):
        params = dict(zip(keys, combo))

        # Run backtest on each data file
        total_score = 0
        combined_result = None

        for data_file in data_files:
            if not os.path.exists(data_file):
                continue

            try:
                result = run_single_backtest(data_file, params, initial_balance)
                score = calculate_score(result)
                total_score += score

                if combined_result is None:
                    combined_result = result
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue

        # Average score across files
        avg_score = total_score / len(data_files)
        all_results.append((params, avg_score))

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_result = combined_result

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(all_combinations)} | Best Score: {best_score:.2f}")

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        best_result=best_result,
        all_results=sorted(all_results, key=lambda x: x[1], reverse=True),
        iterations=len(all_combinations)
    )


def genetic_optimize(
    data_files: List[str],
    param_ranges: Dict[str, Tuple[float, float]] = None,
    population_size: int = 50,
    generations: int = 20,
    mutation_rate: float = 0.1,
    initial_balance: float = 4.5
) -> OptimizationResult:
    """
    Genetic algorithm optimization

    Evolves parameter sets to find optimal configuration
    """
    # Default ranges
    if param_ranges is None:
        param_ranges = {
            'fast_ema': (5, 15),
            'slow_ema': (15, 35),
            'rsi_period': (8, 20),
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 80),
            'atr_multiplier_sl': (1.0, 3.0),
            'atr_multiplier_tp': (1.5, 4.0),
            'position_size_pct': (0.6, 1.0),
            'max_risk_per_trade': (0.01, 0.06),
            'trailing_stop_pct': (0.008, 0.03),
        }

    def create_individual() -> Dict[str, Any]:
        """Create random individual"""
        individual = {}
        for key, (min_val, max_val) in param_ranges.items():
            if key in ['fast_ema', 'slow_ema', 'rsi_period']:
                individual[key] = random.randint(int(min_val), int(max_val))
            else:
                individual[key] = random.uniform(min_val, max_val)
        return individual

    def crossover(parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two parents"""
        child = {}
        for key in parent1.keys():
            child[key] = random.choice([parent1[key], parent2[key]])
        return child

    def mutate(individual: Dict) -> Dict:
        """Mutate an individual"""
        mutated = individual.copy()
        for key, (min_val, max_val) in param_ranges.items():
            if random.random() < mutation_rate:
                if key in ['fast_ema', 'slow_ema', 'rsi_period']:
                    mutated[key] = random.randint(int(min_val), int(max_val))
                else:
                    mutated[key] = random.uniform(min_val, max_val)
        return mutated

    def evaluate(individual: Dict) -> float:
        """Evaluate fitness of individual"""
        total_score = 0
        for data_file in data_files:
            if not os.path.exists(data_file):
                continue
            try:
                result = run_single_backtest(data_file, individual, initial_balance)
                total_score += calculate_score(result)
            except:
                return -1000
        return total_score / len(data_files)

    # Initialize population
    population = [create_individual() for _ in range(population_size)]

    best_ever = None
    best_score_ever = -float('inf')
    best_result_ever = None
    all_results = []

    print(f"Starting genetic optimization: {generations} generations, {population_size} population")

    for gen in range(generations):
        # Evaluate population
        fitness = []
        for ind in population:
            score = evaluate(ind)
            fitness.append((ind, score))
            all_results.append((ind.copy(), score))

        # Sort by fitness
        fitness.sort(key=lambda x: x[1], reverse=True)

        # Track best
        if fitness[0][1] > best_score_ever:
            best_score_ever = fitness[0][1]
            best_ever = fitness[0][0].copy()
            # Get actual result
            try:
                best_result_ever = run_single_backtest(
                    data_files[0], best_ever, initial_balance
                )
            except:
                pass

        print(f"Gen {gen+1}/{generations} | Best: {fitness[0][1]:.2f} | "
              f"Avg: {np.mean([f[1] for f in fitness]):.2f}")

        # Selection (top 50%)
        survivors = [f[0] for f in fitness[:population_size // 2]]

        # Create next generation
        new_population = survivors.copy()

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return OptimizationResult(
        best_params=best_ever,
        best_score=best_score_ever,
        best_result=best_result_ever,
        all_results=sorted(all_results, key=lambda x: x[1], reverse=True)[:100],
        iterations=generations * population_size
    )


def walk_forward_optimization(
    data_file: str,
    train_pct: float = 0.7,
    num_folds: int = 3,
    param_grid: Dict[str, List] = None,
    initial_balance: float = 4.5
) -> List[OptimizationResult]:
    """
    Walk-forward optimization

    Trains on historical data, tests on future data to prevent overfitting
    """
    param_grid = param_grid or PARAM_GRID_FINE

    # Load full data
    backtester = Backtester(initial_balance=initial_balance)
    df = backtester.load_data(data_file)

    total_len = len(df)
    fold_size = total_len // num_folds

    results = []

    for fold in range(num_folds):
        # Calculate train/test split for this fold
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, total_len)

        train_start = max(0, test_start - int(fold_size * train_pct / (1 - train_pct)))
        train_end = test_start

        if train_end - train_start < 200:
            continue

        print(f"\nFold {fold + 1}/{num_folds}")
        print(f"Train: {train_start} to {train_end} ({train_end - train_start} bars)")
        print(f"Test: {test_start} to {test_end} ({test_end - test_start} bars)")

        # Save train data to temp file
        train_df = df.iloc[train_start:train_end]
        train_file = f'/tmp/train_fold_{fold}.csv'
        train_df.to_csv(train_file)

        # Optimize on train data
        opt_result = grid_search(
            [train_file],
            param_grid,
            max_combinations=100,
            initial_balance=initial_balance
        )

        print(f"Best params: {opt_result.best_params}")
        print(f"Train score: {opt_result.best_score:.2f}")

        # Test on test data
        test_df = df.iloc[test_start:test_end]
        test_file = f'/tmp/test_fold_{fold}.csv'
        test_df.to_csv(test_file)

        test_result = run_single_backtest(
            test_file,
            opt_result.best_params,
            initial_balance
        )

        test_score = calculate_score(test_result)
        print(f"Test score: {test_score:.2f}")
        print(f"Test PnL: ${test_result.total_pnl:.4f} ({test_result.total_pnl_pct:.2f}%)")

        opt_result.best_result = test_result
        opt_result.best_score = test_score
        results.append(opt_result)

        # Cleanup temp files
        os.remove(train_file)
        os.remove(test_file)

    return results


def save_optimized_params(params: Dict[str, Any], filename: str = 'optimized_params.json'):
    """Save optimized parameters to file"""
    with open(filename, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved to {filename}")


def load_optimized_params(filename: str = 'optimized_params.json') -> Dict[str, Any]:
    """Load optimized parameters from file"""
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    """Run optimization"""
    data_dir = backtest_config.data_dir

    data_files = [
        os.path.join(data_dir, backtest_config.sol_data),
        os.path.join(data_dir, backtest_config.xrp_data),
        os.path.join(data_dir, backtest_config.btc_data),
    ]

    # Filter existing files
    data_files = [f for f in data_files if os.path.exists(f)]

    if not data_files:
        print("No data files found!")
        return

    print("="*60)
    print("STRATEGY OPTIMIZATION")
    print("="*60)
    print(f"Data files: {data_files}")
    print(f"Initial balance: ${backtest_config.initial_balance}")

    # Run genetic optimization (faster and often better)
    print("\n--- Running Genetic Optimization ---")
    result = genetic_optimize(
        data_files,
        population_size=30,
        generations=15,
        initial_balance=backtest_config.initial_balance
    )

    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Score: {result.best_score:.2f}")
    print(f"Iterations: {result.iterations}")
    print("\nBest Parameters:")
    for key, value in result.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if result.best_result:
        print("\nBacktest with Best Params:")
        print_results(result.best_result)

    # Save best params
    save_optimized_params(result.best_params)

    # Top 5 parameter sets
    print("\n--- Top 5 Parameter Sets ---")
    for i, (params, score) in enumerate(result.all_results[:5], 1):
        print(f"\n{i}. Score: {score:.2f}")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")


if __name__ == "__main__":
    main()
