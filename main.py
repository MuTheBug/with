#!/usr/bin/env python3
"""
Binance USDT-M Futures Trading Bot
===================================
Adaptive regime-switching bot optimized for small-balance aggressive compounding.

Architecture:
  1. Fetch multi-TF data for configured pairs
  2. Compute indicators across all timeframes
  3. Detect market regime (trending, ranging, breakout, choppy)
  4. Select optimal strategy for detected regime
  5. Generate and validate trade signals
  6. Execute with precision risk management
  7. Monitor positions and manage exits

Usage:
  # Live trading (testnet)
  export BINANCE_API_KEY=your_key
  export BINANCE_API_SECRET=your_secret
  export BINANCE_TESTNET=true
  python main.py

  # Live trading (production)
  export BINANCE_TESTNET=false
  python main.py

  # Backtest mode
  python run_backtest.py
"""

import sys
import time
import signal
import logging
import json
from datetime import datetime, timezone
from typing import Dict, Optional

from config import (
    BotConfig, MarketRegime, StrategyName,
    REGIME_STRATEGY_MAP,
)
from indicators import compute_all_indicators
from regime_detector import RegimeDetector, RegimeResult
from strategies import (
    get_strategy, TradeSignal, SignalDirection,
    STRATEGY_REGISTRY,
)
from risk_manager import RiskManager, RiskState, Position
from data_fetcher import BinanceFuturesClient


# ---------- Logging Setup ----------

def setup_logging(config: BotConfig):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.log_file),
        ],
    )


logger = logging.getLogger("bot")


# ---------- Trading Bot ----------

class TradingBot:
    """
    Main trading bot orchestrator.
    Coordinates data fetching, analysis, signal generation, and execution.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.client = BinanceFuturesClient(self.config)
        self.regime_detector = RegimeDetector(self.config)
        self.risk_state = RiskState.load(self.config.state_file)
        self.risk_manager = RiskManager(self.config, self.risk_state)
        self.strategies = {
            name: get_strategy(name, self.config) for name in STRATEGY_REGISTRY
        }
        self.running = True
        self._last_regime: Dict[str, RegimeResult] = {}
        self._iteration = 0

    def start(self):
        """Start the trading bot main loop."""
        logger.info("=" * 60)
        logger.info("BINANCE FUTURES TRADING BOT STARTING")
        logger.info(f"Testnet: {self.config.binance.testnet}")
        logger.info(f"Pairs: {self.config.pairs.primary}")
        logger.info(f"Balance: ${self.risk_state.balance:.2f}")
        logger.info("=" * 60)

        # Fetch initial balance from exchange
        try:
            exchange_balance = self.client.get_balance()
            if exchange_balance > 0:
                self.risk_state.balance = exchange_balance
                self.risk_state.peak_balance = max(
                    self.risk_state.peak_balance, exchange_balance
                )
                logger.info(f"Exchange balance: ${exchange_balance:.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch exchange balance: {e}")
            logger.info(f"Using saved balance: ${self.risk_state.balance:.2f}")

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Main loop
        while self.running:
            try:
                self._iteration += 1
                self._trading_cycle()

                # Save state periodically
                if self._iteration % 10 == 0:
                    self.risk_state.save(self.config.state_file)

                time.sleep(self.config.loop_interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)
                time.sleep(30)  # Wait longer after errors

        self._shutdown()

    def _trading_cycle(self):
        """Single iteration of the trading loop."""
        # 1. Check if trading is allowed
        can_trade, reason = self.risk_manager.can_trade()

        # 2. Check existing positions
        self._manage_positions()

        if not can_trade:
            if self._iteration % 30 == 0:  # Log every ~5 min
                logger.info(f"Trading paused: {reason}")
            return

        # 3. Scan pairs for opportunities
        for symbol in self.config.pairs.primary:
            # Skip if already have position in this symbol
            if any(p.symbol == symbol for p in self.risk_state.positions):
                continue

            try:
                self._analyze_and_trade(symbol)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

            time.sleep(0.5)  # Rate limit spacing

    def _analyze_and_trade(self, symbol: str):
        """Analyze a single symbol and potentially enter a trade."""
        # Fetch multi-TF data
        data_by_tf = self.client.get_multi_tf_data(symbol)
        if not data_by_tf:
            return

        # Compute indicators for each timeframe
        ind_by_tf = {}
        for tf, df in data_by_tf.items():
            if len(df) >= 50:
                ind_by_tf[tf] = compute_all_indicators(df, self.config)

        if not ind_by_tf:
            return

        # Detect regime
        regime_result = self.regime_detector.detect(ind_by_tf, "1h")
        self._last_regime[symbol] = regime_result

        # Log regime periodically
        if self._iteration % 60 == 0:
            logger.info(
                f"[{symbol}] Regime: {regime_result.regime.value} "
                f"(conf={regime_result.confidence:.2f}, "
                f"trend={regime_result.trend_score:.2f}, "
                f"vol={regime_result.volatility_score:.2f})"
            )

        # Select strategy for this regime
        strat_name = REGIME_STRATEGY_MAP.get(
            regime_result.regime, StrategyName.SCALP
        )
        strategy = self.strategies[strat_name]

        # Use the signal timeframe for signal generation
        signal_tf = self.config.timeframes.signal_tf
        signal_df = ind_by_tf.get(signal_tf) or ind_by_tf.get("1h") or next(iter(ind_by_tf.values()))

        # Generate signal
        trade_signal = strategy.generate_signal(signal_df)

        if trade_signal.direction == SignalDirection.NONE:
            return

        # Validate signal
        approved, adj_signal, val_reason = self.risk_manager.validate_signal(
            trade_signal, regime_result.regime, regime_result.confidence,
        )

        if not approved:
            logger.debug(f"[{symbol}] Signal rejected: {val_reason}")
            return

        # Calculate position size
        current_price = self.client.get_ticker_price(symbol)
        if current_price <= 0:
            return

        quantity, leverage, margin = self.risk_manager.calculate_position_size(
            trade_signal, symbol, current_price, regime_result.confidence,
        )

        if quantity <= 0:
            return

        logger.info(
            f"[{symbol}] SIGNAL: {trade_signal.direction.value} "
            f"qty={quantity}, lev={leverage}x, margin=${margin:.2f}, "
            f"SL={trade_signal.stop_loss:.4f}, TP={trade_signal.take_profit:.4f}, "
            f"strategy={strat_name.value}, regime={regime_result.regime.value}, "
            f"reason={trade_signal.reason}"
        )

        # Round prices to exchange precision
        sl_price = self.risk_manager.round_price(trade_signal.stop_loss, symbol)
        tp_price = self.risk_manager.round_price(trade_signal.take_profit, symbol)

        # Execute
        entry_result, sl_result, tp_result = self.client.open_position(
            symbol=symbol,
            direction=trade_signal.direction,
            quantity=quantity,
            leverage=leverage,
            stop_loss=sl_price,
            take_profit=tp_price,
        )

        if isinstance(entry_result, dict) and "error" in entry_result:
            logger.error(f"[{symbol}] Order failed: {entry_result}")
            return

        # Track position locally
        fill_price = float(entry_result.get("avgPrice", current_price))
        position = Position(
            symbol=symbol,
            direction=trade_signal.direction,
            entry_price=fill_price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=sl_price,
            take_profit=tp_price,
            trailing_stop=sl_price,
            strategy=strat_name.value,
        )
        self.risk_state.positions.append(position)

        logger.info(
            f"[{symbol}] OPENED {trade_signal.direction.value} "
            f"@ {fill_price:.4f}, qty={quantity}, lev={leverage}x"
        )

    def _manage_positions(self):
        """Monitor and manage open positions."""
        for pos in list(self.risk_state.positions):
            try:
                # Get current price
                price_data = self.client.get_mark_price(pos.symbol)
                current_price = price_data["mark_price"]
                if current_price <= 0:
                    continue

                # Update PnL
                pos.update_pnl(current_price)

                # Check for funding rate impact
                funding_rate = price_data["funding_rate"]
                if abs(funding_rate) > 0.001:  # High funding rate
                    logger.warning(
                        f"[{pos.symbol}] High funding rate: {funding_rate:.4f}"
                    )

                # Check exit conditions (trailing stop logic)
                # Get recent high/low from 1m candle
                recent = self.client.get_klines(pos.symbol, "1m", 2)
                if not recent.empty:
                    current_high = recent["high"].iloc[-1]
                    current_low = recent["low"].iloc[-1]
                else:
                    current_high = current_price
                    current_low = current_price

                should_exit, exit_price, exit_reason = self.risk_manager.check_exit_conditions(
                    pos, current_price, current_high, current_low,
                )

                if should_exit:
                    logger.info(
                        f"[{pos.symbol}] CLOSING: {exit_reason}, "
                        f"PnL={pos.pnl:.4f}"
                    )
                    self.client.close_position(
                        pos.symbol, pos.direction, pos.quantity,
                    )
                    record = self.risk_manager.record_trade(
                        pos, current_price, exit_reason,
                    )
                    logger.info(
                        f"[{pos.symbol}] CLOSED: PnL=${record.pnl:.4f} "
                        f"({record.pnl_pct:.2%}), Balance=${self.risk_state.balance:.2f}"
                    )

            except Exception as e:
                logger.error(f"Error managing position {pos.symbol}: {e}")

    def _shutdown(self, signum=None, frame=None):
        """Graceful shutdown."""
        logger.info("Shutting down trading bot...")
        self.running = False
        self.risk_state.save(self.config.state_file)

        # Log final stats
        perf = self.risk_manager.get_performance_summary()
        logger.info("Final Performance:")
        for k, v in perf.items():
            logger.info(f"  {k}: {v}")

    def status(self) -> dict:
        """Get current bot status."""
        return {
            "running": self.running,
            "balance": self.risk_state.balance,
            "positions": len(self.risk_state.positions),
            "total_trades": self.risk_state.total_trades,
            "win_rate": self.risk_state.win_rate,
            "pnl": self.risk_state.total_pnl,
            "drawdown": self.risk_state.current_drawdown,
            "regimes": {
                sym: r.regime.value for sym, r in self._last_regime.items()
            },
            "performance": self.risk_manager.get_performance_summary(),
        }


def main():
    config = BotConfig()
    setup_logging(config)

    # Validate configuration
    if not config.binance.api_key or not config.binance.api_secret:
        logger.warning(
            "No API keys configured. Set BINANCE_API_KEY and BINANCE_API_SECRET "
            "environment variables. Running in dry-run/backtest mode only."
        )
        print("\n[!] No API keys set. For live trading, set environment variables:")
        print("    export BINANCE_API_KEY=your_key")
        print("    export BINANCE_API_SECRET=your_secret")
        print("    export BINANCE_TESTNET=true  # Start with testnet!")
        print("\n[*] To run backtests instead: python run_backtest.py")
        return

    bot = TradingBot(config)
    bot.start()


if __name__ == "__main__":
    main()
