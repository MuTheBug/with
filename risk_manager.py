"""
Risk management system for small-balance aggressive compounding.

Key principles:
1. Never risk more than X% of balance per trade (fractional Kelly)
2. Scale position size with account growth (geometric compounding)
3. Circuit breakers for drawdown protection
4. Dynamic leverage based on confidence + regime
5. Respect Binance minimum order sizes and fee structure
"""

import numpy as np
import time
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from datetime import datetime, timezone

from config import BotConfig, RiskConfig, MarketRegime
from strategies import TradeSignal, SignalDirection


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: SignalDirection
    entry_price: float
    quantity: float
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float] = None
    entry_time: float = 0.0
    entry_bar: int = 0
    strategy: str = ""
    pnl: float = 0.0
    max_pnl: float = 0.0
    fees_paid: float = 0.0

    def __post_init__(self):
        if self.entry_time == 0.0:
            self.entry_time = time.time()

    def update_pnl(self, current_price: float):
        if self.direction == SignalDirection.LONG:
            self.pnl = (current_price - self.entry_price) * self.quantity - self.fees_paid
        else:
            self.pnl = (self.entry_price - current_price) * self.quantity - self.fees_paid
        self.max_pnl = max(self.max_pnl, self.pnl)

    def notional_value(self) -> float:
        return self.entry_price * self.quantity

    def margin_used(self) -> float:
        return self.notional_value() / self.leverage if self.leverage > 0 else self.notional_value()


@dataclass
class TradeRecord:
    """Historical trade record for performance tracking."""
    symbol: str
    direction: str
    strategy: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    pnl: float
    pnl_pct: float
    fees: float
    entry_time: float
    exit_time: float
    exit_reason: str = ""
    balance_after: float = 0.0


@dataclass
class RiskState:
    """Persistent risk state."""
    balance: float = 4.0
    peak_balance: float = 4.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown_seen: float = 0.0
    cooldown_until: float = 0.0
    daily_trades: int = 0
    daily_pnl: float = 0.0
    last_trade_day: str = ""
    positions: List[Position] = field(default_factory=list)
    trade_history: List[TradeRecord] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def current_drawdown(self) -> float:
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trade_history if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trade_history if t.pnl < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_win(self) -> float:
        wins = [t.pnl for t in self.trade_history if t.pnl > 0]
        return np.mean(wins) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [t.pnl for t in self.trade_history if t.pnl < 0]
        return abs(np.mean(losses)) if losses else 0.0

    def save(self, filepath: str = "bot_state.json"):
        """Save state to disk."""
        data = {
            "balance": self.balance,
            "peak_balance": self.peak_balance,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "max_drawdown_seen": self.max_drawdown_seen,
            "cooldown_until": self.cooldown_until,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "last_trade_day": self.last_trade_day,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str = "bot_state.json") -> "RiskState":
        """Load state from disk."""
        if not os.path.exists(filepath):
            return cls()
        with open(filepath) as f:
            data = json.load(f)
        state = cls()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state


class RiskManager:
    """
    Manages all risk decisions:
    - Position sizing (fractional Kelly)
    - Leverage selection
    - Drawdown circuit breakers
    - Consecutive loss cooldowns
    - Binance constraint compliance
    """

    # Binance Futures quantity step size (decimals) and minimums
    QUANTITY_PRECISION = {
        "BTCUSDT": 3,     # min 0.001 BTC (~$100 at $100K)
        "ETHUSDT": 3,     # min 0.001 ETH (~$3 at $3K)
        "SOLUSDT": 1,     # min 0.1 SOL (~$20)
        "XRPUSDT": 1,     # min 0.1 XRP (~$0.20)
        "BNBUSDT": 2,     # min 0.01 BNB
        "DOGEUSDT": 0,    # min 1 DOGE
        "AVAXUSDT": 1,    # min 0.1 AVAX
        "ADAUSDT": 0,     # min 1 ADA
    }

    PRICE_PRECISION = {
        "BTCUSDT": 1,
        "ETHUSDT": 2,
        "SOLUSDT": 2,
        "XRPUSDT": 4,
        "BNBUSDT": 2,
        "DOGEUSDT": 5,
        "AVAXUSDT": 2,
        "ADAUSDT": 4,
    }

    def __init__(self, config: Optional[BotConfig] = None, state: Optional[RiskState] = None,
                 backtest_mode: bool = False):
        self.config = config or BotConfig()
        self.rc = self.config.risk
        self.state = state or RiskState()
        self.backtest_mode = backtest_mode
        self._bar_counter = 0  # For backtest cooldown tracking
        self._cooldown_bars_remaining = 0

    def advance_bar(self):
        """Advance bar counter (call once per bar in backtest)."""
        self._bar_counter += 1
        if self._cooldown_bars_remaining > 0:
            self._cooldown_bars_remaining -= 1

    def can_trade(self) -> tuple:
        """Check if trading is allowed. Returns (allowed, reason)."""
        # Balance check
        if self.state.balance < self.rc.min_balance_to_trade:
            return False, f"Balance {self.state.balance:.2f} below minimum {self.rc.min_balance_to_trade}"

        # Cooldown check
        if self.backtest_mode:
            if self._cooldown_bars_remaining > 0:
                return False, f"In cooldown for {self._cooldown_bars_remaining} more bars"
        else:
            if time.time() < self.state.cooldown_until:
                remaining = (self.state.cooldown_until - time.time()) / 3600
                return False, f"In cooldown for {remaining:.1f} more hours"

        # Hard drawdown circuit breaker
        dd = self.state.current_drawdown
        if dd >= self.rc.max_drawdown_hard:
            return False, f"Drawdown {dd:.1%} exceeds hard limit {self.rc.max_drawdown_hard:.1%}"

        # Maximum positions
        max_pos = self._max_positions()
        if len(self.state.positions) >= max_pos:
            return False, f"Maximum {max_pos} positions already open"

        # Consecutive loss protection
        if self.state.consecutive_losses >= self.rc.max_consecutive_losses:
            if self.backtest_mode:
                self._cooldown_bars_remaining = self.rc.cooldown_after_max_losses  # bars
            else:
                self.state.cooldown_until = time.time() + self.rc.cooldown_after_max_losses * 3600
            self.state.consecutive_losses = 0  # Reset after setting cooldown
            return False, f"Hit {self.rc.max_consecutive_losses} consecutive losses, cooling down"

        return True, "OK"

    def calculate_position_size(
        self,
        signal: TradeSignal,
        symbol: str,
        current_price: float,
        regime_confidence: float = 0.5,
    ) -> tuple:
        """
        Calculate position size in base asset quantity.
        Returns (quantity, leverage, margin_required).

        For small accounts ($4-$20), the key challenge is meeting exchange
        minimum order sizes. We use higher leverage + smaller risk fraction.
        """
        balance = self.state.balance

        # Determine risk per trade
        risk_pct = self._risk_per_trade(signal.confidence, regime_confidence)

        # Kelly criterion adjustment
        kelly = self._kelly_size(signal)
        risk_pct = min(risk_pct, kelly)

        # Reduce risk during drawdown
        dd = self.state.current_drawdown
        if dd > self.rc.max_drawdown_soft:
            scale = max(0.3, 1.0 - (dd - self.rc.max_drawdown_soft) / 0.1)
            risk_pct *= scale

        # Risk amount in USDT
        risk_amount = balance * risk_pct

        # Determine leverage
        leverage = self._select_leverage(signal, regime_confidence)

        # Calculate from stop distance
        stop_dist_pct = signal.risk_pct  # e.g. 0.03 = 3%
        if stop_dist_pct <= 0:
            return 0, leverage, 0

        # Position notional = risk_amount / stop_dist_pct
        notional = risk_amount / stop_dist_pct

        # Get minimum quantity for this symbol
        precision = self.QUANTITY_PRECISION.get(symbol, 3)
        min_qty = 10 ** (-precision)  # e.g., 0.001 BTC, 1 SOL, 0.1 XRP
        min_notional_for_qty = min_qty * current_price
        min_notional = max(min_notional_for_qty, self.rc.min_notional)

        # If risk-based notional is below minimum, scale up to minimum
        if notional < min_notional:
            # Can we afford the minimum at max leverage?
            min_margin = min_notional / self.rc.max_leverage
            if min_margin > balance * 0.90:
                return 0, leverage, 0  # Can't afford this symbol

            notional = min_notional
            # Set leverage to afford this margin
            needed_lev = int(np.ceil(notional / (balance * 0.85)))
            leverage = min(max(needed_lev, leverage), self.rc.max_leverage)

        # Cap notional at balance * leverage
        max_notional = balance * leverage
        notional = min(notional, max_notional)

        # Quantity in base asset
        quantity = notional / current_price

        # Round to exchange precision (round up to at least min_qty)
        quantity = max(round(quantity, precision), min_qty)

        # Recalculate actual notional and margin
        actual_notional = quantity * current_price
        margin_required = actual_notional / leverage

        # Check actual risk doesn't exceed budget
        actual_risk = actual_notional * stop_dist_pct
        max_allowed_risk = balance * min(risk_pct * 3, 0.05)  # Allow up to 3x target or 5%
        if actual_risk > max_allowed_risk:
            return 0, leverage, 0  # Position too risky for our balance

        # Final check: margin must not exceed balance
        if margin_required > balance * 0.95:
            # Try bumping leverage to max
            leverage = self.rc.max_leverage
            margin_required = actual_notional / leverage
            if margin_required > balance * 0.95:
                return 0, leverage, 0  # Still can't afford

        return quantity, leverage, margin_required

    def _risk_per_trade(self, signal_confidence: float, regime_confidence: float) -> float:
        """Dynamic risk per trade based on confidence levels."""
        base_risk = self.rc.max_risk_per_trade

        # Scale up with confidence
        confidence_avg = (signal_confidence + regime_confidence) / 2
        if confidence_avg > 0.75:
            base_risk = self.rc.max_risk_per_trade_aggressive

        # Scale with winning streak (compound faster when hot)
        if self.state.consecutive_wins >= 3:
            base_risk *= 1.2
        elif self.state.consecutive_wins >= 5:
            base_risk *= 1.4

        # Cap at 5% regardless
        return min(base_risk, 0.05)

    def _kelly_size(self, signal: TradeSignal) -> float:
        """
        Fractional Kelly criterion for position sizing.
        Kelly% = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        We use fraction of Kelly for safety.
        """
        # Use historical stats if available, otherwise estimate from signal
        if self.state.total_trades >= 20:
            wr = self.state.win_rate
            avg_w = self.state.avg_win
            avg_l = self.state.avg_loss
        else:
            # Estimate from R:R ratio
            rr = signal.risk_reward
            wr = 0.55  # Conservative estimate
            avg_w = rr
            avg_l = 1.0

        if avg_w <= 0:
            return 0.01

        kelly = (wr * avg_w - (1 - wr) * avg_l) / avg_w
        kelly = max(kelly, 0.005)  # Minimum tiny size
        kelly *= self.rc.kelly_fraction  # Fractional Kelly

        return min(kelly, 0.05)  # Cap at 5%

    def _select_leverage(self, signal: TradeSignal, regime_confidence: float) -> int:
        """Select leverage based on signal quality and regime."""
        # Start with signal's suggestion
        lev = signal.leverage_hint

        # Adjust for regime confidence
        if regime_confidence < 0.5:
            lev = max(lev - 3, self.rc.min_leverage)
        elif regime_confidence > 0.75:
            lev = min(lev + 2, self.rc.max_leverage)

        # Reduce during drawdown
        dd = self.state.current_drawdown
        if dd > self.rc.max_drawdown_soft:
            lev = max(lev - 5, self.rc.min_leverage)

        # Small balance needs more leverage to meet minimums
        if self.state.balance < 10:
            lev = max(lev, 10)
        elif self.state.balance < 20:
            lev = max(lev, 7)

        return int(np.clip(lev, self.rc.min_leverage, self.rc.max_leverage))

    def _max_positions(self) -> int:
        """Maximum positions based on balance size."""
        balance = self.state.balance
        if balance < 50:
            return self.rc.max_positions
        elif balance < 500:
            return self.rc.max_positions_medium
        return self.rc.max_positions_large

    def validate_signal(
        self,
        signal: TradeSignal,
        regime: MarketRegime,
        regime_confidence: float,
    ) -> tuple:
        """
        Validate a signal against risk rules.
        Returns (approved, adjusted_signal_or_none, reason).
        """
        if signal.direction == SignalDirection.NONE:
            return False, None, "No signal"

        # Minimum confidence
        if signal.confidence < 0.40:
            return False, None, f"Signal confidence {signal.confidence:.2f} too low"

        # Minimum R:R ratio
        rr = signal.risk_reward
        if rr < 1.2:
            return False, None, f"R:R {rr:.2f} below minimum 1.2"

        # Risk per trade check
        risk_pct = signal.risk_pct
        if risk_pct > 0.08:  # More than 8% risk per unit
            return False, None, f"Risk per trade {risk_pct:.1%} too high"

        if risk_pct <= 0:
            return False, None, "Invalid stop loss (zero risk)"

        # Regime confidence filter
        if regime_confidence < self.config.regime.min_regime_confidence:
            return False, None, f"Regime confidence {regime_confidence:.2f} too low"

        # Don't fight strong trends with counter-trend signals
        if regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.BREAKOUT_UP):
            if signal.direction == SignalDirection.SHORT:
                return False, None, "Shorting in strong uptrend"
        elif regime in (MarketRegime.STRONG_TREND_DOWN, MarketRegime.BREAKOUT_DOWN):
            if signal.direction == SignalDirection.LONG:
                return False, None, "Going long in strong downtrend"

        return True, signal, "Approved"

    def record_trade(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str = "",
    ) -> TradeRecord:
        """Record a completed trade and update state."""
        # Calculate PnL
        if position.direction == SignalDirection.LONG:
            raw_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            raw_pnl = (position.entry_price - exit_price) * position.quantity

        # Fees (entry + exit)
        entry_fee = position.entry_price * position.quantity * self.rc.taker_fee
        exit_fee = exit_price * position.quantity * self.rc.taker_fee
        total_fees = entry_fee + exit_fee

        net_pnl = raw_pnl - total_fees
        margin = position.margin_used()
        pnl_pct = net_pnl / margin if margin > 0 else 0.0

        record = TradeRecord(
            symbol=position.symbol,
            direction=position.direction.value,
            strategy=position.strategy,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            leverage=position.leverage,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=total_fees,
            entry_time=position.entry_time,
            exit_time=time.time(),
            exit_reason=exit_reason,
        )

        # Update state
        self.state.balance += net_pnl
        self.state.total_pnl += net_pnl
        self.state.total_fees += total_fees
        self.state.total_trades += 1
        record.balance_after = self.state.balance

        if net_pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losing_trades += 1
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0

        # Update peak balance and drawdown
        if self.state.balance > self.state.peak_balance:
            self.state.peak_balance = self.state.balance
        self.state.max_drawdown_seen = max(
            self.state.max_drawdown_seen, self.state.current_drawdown
        )

        self.state.trade_history.append(record)

        # Remove position
        if position in self.state.positions:
            self.state.positions.remove(position)

        return record

    def check_exit_conditions(
        self, position: Position, current_price: float, current_high: float,
        current_low: float, bar_open: float = 0, bars_held: int = 0,
    ) -> tuple:
        """
        Check if position should be exited.
        Returns (should_exit, exit_price, reason).

        Uses bar open price to resolve ambiguity when both SL and TP
        could be hit in the same bar.
        """
        position.update_pnl(current_price)
        if bar_open == 0:
            bar_open = current_price

        if position.direction == SignalDirection.LONG:
            sl_hit = current_low <= position.stop_loss
            tp_hit = current_high >= position.take_profit

            if sl_hit and tp_hit:
                # Both could be hit: use open to determine which was first
                if bar_open <= position.stop_loss:
                    return True, position.stop_loss, "Stop loss hit"
                elif bar_open >= position.take_profit:
                    return True, position.take_profit, "Take profit hit"
                # Open is between SL and TP: check which is closer to open
                sl_dist = bar_open - position.stop_loss
                tp_dist = position.take_profit - bar_open
                if sl_dist < tp_dist:
                    return True, position.stop_loss, "Stop loss hit"
                else:
                    return True, position.take_profit, "Take profit hit"
            elif tp_hit:
                return True, position.take_profit, "Take profit hit"
            elif sl_hit:
                return True, position.stop_loss, "Stop loss hit"
        else:
            sl_hit = current_high >= position.stop_loss
            tp_hit = current_low <= position.take_profit

            if sl_hit and tp_hit:
                if bar_open >= position.stop_loss:
                    return True, position.stop_loss, "Stop loss hit"
                elif bar_open <= position.take_profit:
                    return True, position.take_profit, "Take profit hit"
                sl_dist = position.stop_loss - bar_open
                tp_dist = bar_open - position.take_profit
                if sl_dist < tp_dist:
                    return True, position.stop_loss, "Stop loss hit"
                else:
                    return True, position.take_profit, "Take profit hit"
            elif tp_hit:
                return True, position.take_profit, "Take profit hit"
            elif sl_hit:
                return True, position.stop_loss, "Stop loss hit"

        # Trailing stop update and check
        if position.trailing_stop is not None:
            if position.direction == SignalDirection.LONG:
                sl_width = position.entry_price - position.stop_loss
                new_trail = current_high - sl_width * 1.0
                if new_trail > position.trailing_stop:
                    position.trailing_stop = new_trail
                if current_low <= position.trailing_stop and position.trailing_stop > position.entry_price:
                    return True, position.trailing_stop, "Trailing stop hit"
            else:
                sl_width = position.stop_loss - position.entry_price
                new_trail = current_low + sl_width * 1.0
                if new_trail < position.trailing_stop:
                    position.trailing_stop = new_trail
                if current_high >= position.trailing_stop and position.trailing_stop < position.entry_price:
                    return True, position.trailing_stop, "Trailing stop hit"

        # Break-even stop: move SL to entry after 1.5x risk profit
        if position.direction == SignalDirection.LONG:
            risk = position.entry_price - position.stop_loss
            if risk > 0 and current_price > position.entry_price + risk * 1.5:
                if position.stop_loss < position.entry_price:
                    position.stop_loss = position.entry_price + risk * 0.1
        else:
            risk = position.stop_loss - position.entry_price
            if risk > 0 and current_price < position.entry_price - risk * 1.5:
                if position.stop_loss > position.entry_price:
                    position.stop_loss = position.entry_price - risk * 0.1

        # Max holding period (prevents capital lock-up)
        max_hold = 48  # 48 bars = 2 days for 1h timeframe
        if "scalp" in position.strategy:
            max_hold = 12  # 12 hours for scalps
        elif "momentum" in position.strategy:
            max_hold = 72  # 3 days for momentum
        if bars_held >= max_hold:
            return True, current_price, f"Max hold ({max_hold} bars)"

        return False, 0, ""

    def round_price(self, price: float, symbol: str) -> float:
        """Round price to exchange precision."""
        precision = self.PRICE_PRECISION.get(symbol, 2)
        return round(price, precision)

    def get_performance_summary(self) -> dict:
        """Get performance statistics."""
        s = self.state
        trades = s.trade_history

        if not trades:
            return {"total_trades": 0, "balance": s.balance}

        returns = [t.pnl_pct for t in trades]
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]

        # Sharpe ratio (simplified, per-trade)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Max drawdown from trade history
        cumulative = np.cumsum([t.pnl for t in trades])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative)
        max_dd_amount = drawdowns.max() if len(drawdowns) > 0 else 0

        return {
            "total_trades": s.total_trades,
            "win_rate": s.win_rate,
            "profit_factor": s.profit_factor,
            "total_pnl": s.total_pnl,
            "total_fees": s.total_fees,
            "avg_win": float(np.mean(wins)) if wins else 0,
            "avg_loss": float(np.mean(losses)) if losses else 0,
            "max_drawdown": s.max_drawdown_seen,
            "max_dd_amount": float(max_dd_amount),
            "sharpe_ratio": sharpe,
            "balance": s.balance,
            "peak_balance": s.peak_balance,
            "return_pct": (s.balance - 4.0) / 4.0 * 100,  # Assuming $4 start
            "consecutive_wins": s.consecutive_wins,
            "consecutive_losses": s.consecutive_losses,
        }
