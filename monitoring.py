#!/usr/bin/env python3
"""
GRID BOT MONITORING & ALERTING SYSTEM
======================================
Real-time monitoring dashboard and alert system for the grid trading bot.

Features:
- Real-time P&L tracking
- Position monitoring
- Alert notifications (Telegram, Discord, Email)
- Performance metrics
- Risk alerts
"""

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Monitor')


@dataclass
class PerformanceMetrics:
    """Trading performance metrics"""
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    trades_per_hour: float = 0.0


@dataclass
class AlertConfig:
    """Alert configuration"""
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""

    # Alert thresholds
    pnl_alert_threshold: float = 5.0  # Alert on 5% P&L change
    drawdown_alert_threshold: float = 8.0  # Alert at 8% drawdown
    position_alert_threshold: float = 3  # Alert when positions > 3

    # Alert cooldowns (seconds)
    alert_cooldown: int = 300  # 5 minutes between similar alerts


class AlertManager:
    """Manages alert notifications"""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.last_alerts: Dict[str, float] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    def _can_alert(self, alert_type: str) -> bool:
        """Check if alert cooldown has passed"""
        last_time = self.last_alerts.get(alert_type, 0)
        return time.time() - last_time >= self.config.alert_cooldown

    async def send_alert(self, alert_type: str, message: str, priority: str = "normal"):
        """Send alert through configured channels"""
        if not self._can_alert(alert_type):
            return

        self.last_alerts[alert_type] = time.time()

        # Add timestamp and priority
        full_message = f"[{priority.upper()}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{message}"

        tasks = []

        if self.config.telegram_enabled:
            tasks.append(self._send_telegram(full_message))

        if self.config.discord_enabled:
            tasks.append(self._send_discord(full_message, priority))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(f"Alert sent [{alert_type}]: {message}")

    async def _send_telegram(self, message: str):
        """Send Telegram notification"""
        await self._ensure_session()

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            'chat_id': self.config.telegram_chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }

        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    logger.error(f"Telegram error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _send_discord(self, message: str, priority: str):
        """Send Discord notification"""
        await self._ensure_session()

        # Color based on priority
        colors = {
            'low': 0x808080,
            'normal': 0x0099ff,
            'high': 0xff9900,
            'critical': 0xff0000
        }

        payload = {
            'embeds': [{
                'title': 'Grid Bot Alert',
                'description': message,
                'color': colors.get(priority, 0x0099ff),
                'timestamp': datetime.utcnow().isoformat()
            }]
        }

        try:
            async with self.session.post(self.config.discord_webhook_url, json=payload) as resp:
                if resp.status not in [200, 204]:
                    logger.error(f"Discord error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Discord send failed: {e}")


class PerformanceTracker:
    """Track and calculate trading performance"""

    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: deque = deque(maxlen=10000)
        self.initial_balance: float = 0.0
        self.peak_balance: float = 0.0
        self.start_time: Optional[datetime] = None

    def set_initial_balance(self, balance: float):
        """Set initial balance"""
        self.initial_balance = balance
        self.peak_balance = balance
        self.start_time = datetime.now()
        self.equity_curve.append({
            'time': datetime.now(),
            'balance': balance
        })

    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self.trades.append({
            **trade,
            'time': datetime.now()
        })

    def update_balance(self, balance: float):
        """Update current balance"""
        if balance > self.peak_balance:
            self.peak_balance = balance

        self.equity_curve.append({
            'time': datetime.now(),
            'balance': balance
        })

    def calculate_metrics(self, current_balance: float) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        metrics = PerformanceMetrics()

        if self.initial_balance == 0:
            return metrics

        # P&L calculations
        metrics.total_pnl = current_balance - self.initial_balance
        metrics.total_pnl_pct = metrics.total_pnl / self.initial_balance * 100

        # Drawdown
        if self.peak_balance > 0:
            metrics.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance * 100
            metrics.max_drawdown = max(
                metrics.current_drawdown,
                *[
                    (self.peak_balance - e['balance']) / self.peak_balance * 100
                    for e in self.equity_curve if e['balance'] < self.peak_balance
                ]
            ) if self.equity_curve else metrics.current_drawdown

        # Trade statistics
        metrics.total_trades = len(self.trades)

        if self.trades:
            wins = [t for t in self.trades if t.get('pnl', 0) > 0]
            losses = [t for t in self.trades if t.get('pnl', 0) < 0]

            metrics.winning_trades = len(wins)
            metrics.losing_trades = len(losses)
            metrics.win_rate = metrics.winning_trades / metrics.total_trades * 100 if metrics.total_trades > 0 else 0

            win_pnls = [t['pnl'] for t in wins]
            loss_pnls = [abs(t['pnl']) for t in losses]

            if win_pnls:
                metrics.avg_win = sum(win_pnls) / len(win_pnls)
                metrics.largest_win = max(win_pnls)

            if loss_pnls:
                metrics.avg_loss = sum(loss_pnls) / len(loss_pnls)
                metrics.largest_loss = max(loss_pnls)

            # Profit factor
            total_wins = sum(win_pnls) if win_pnls else 0
            total_losses = sum(loss_pnls) if loss_pnls else 1
            metrics.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

            # Trades per hour
            if self.start_time:
                hours = (datetime.now() - self.start_time).total_seconds() / 3600
                metrics.trades_per_hour = metrics.total_trades / hours if hours > 0 else 0

        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = []
            equity_list = list(self.equity_curve)
            for i in range(1, len(equity_list)):
                ret = (equity_list[i]['balance'] - equity_list[i-1]['balance']) / equity_list[i-1]['balance']
                returns.append(ret)

            if returns:
                import statistics
                mean_return = statistics.mean(returns)
                std_return = statistics.stdev(returns) if len(returns) > 1 else 0.001
                metrics.sharpe_ratio = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0

        return metrics


class MonitoringDashboard:
    """Real-time monitoring dashboard"""

    def __init__(self, alert_config: AlertConfig = None):
        self.alert_config = alert_config or AlertConfig()
        self.alert_manager = AlertManager(self.alert_config)
        self.performance_tracker = PerformanceTracker()

        # Current state
        self.current_price: float = 0.0
        self.current_balance: float = 0.0
        self.positions: List[Dict] = []
        self.open_orders: int = 0
        self.is_running: bool = False
        self.last_update: Optional[datetime] = None

    async def start(self):
        """Start the monitoring dashboard"""
        self.is_running = True
        logger.info("Monitoring dashboard started")

    async def stop(self):
        """Stop the monitoring dashboard"""
        self.is_running = False
        await self.alert_manager.close()
        logger.info("Monitoring dashboard stopped")

    async def update_state(self,
                          price: float = None,
                          balance: float = None,
                          positions: List[Dict] = None,
                          open_orders: int = None):
        """Update monitoring state"""
        if price is not None:
            self.current_price = price

        if balance is not None:
            if self.performance_tracker.initial_balance == 0:
                self.performance_tracker.set_initial_balance(balance)
            self.current_balance = balance
            self.performance_tracker.update_balance(balance)

        if positions is not None:
            self.positions = positions

        if open_orders is not None:
            self.open_orders = open_orders

        self.last_update = datetime.now()

        # Check for alerts
        await self._check_alerts()

    async def record_trade(self, trade: Dict):
        """Record a trade"""
        self.performance_tracker.record_trade(trade)

        # Send trade alert
        pnl = trade.get('pnl', 0)
        pnl_str = f"+${pnl:.4f}" if pnl > 0 else f"-${abs(pnl):.4f}"

        await self.alert_manager.send_alert(
            'trade',
            f"Trade Completed: {trade.get('side')} at ${trade.get('price', 0):.2f}\n"
            f"P&L: {pnl_str}",
            priority='low'
        )

    async def _check_alerts(self):
        """Check conditions and send alerts"""
        metrics = self.get_metrics()

        # Drawdown alert
        if metrics.current_drawdown >= self.alert_config.drawdown_alert_threshold:
            await self.alert_manager.send_alert(
                'drawdown',
                f"⚠️ HIGH DRAWDOWN ALERT\n"
                f"Current Drawdown: {metrics.current_drawdown:.2f}%\n"
                f"Balance: ${self.current_balance:.2f}",
                priority='high'
            )

        # Position count alert
        if len(self.positions) >= self.alert_config.position_alert_threshold:
            await self.alert_manager.send_alert(
                'positions',
                f"📊 Position Alert\n"
                f"Open Positions: {len(self.positions)}\n"
                f"Consider reducing exposure",
                priority='normal'
            )

        # Significant P&L change
        if abs(metrics.total_pnl_pct) >= self.alert_config.pnl_alert_threshold:
            priority = 'high' if metrics.total_pnl_pct < 0 else 'normal'
            emoji = "📈" if metrics.total_pnl_pct > 0 else "📉"
            await self.alert_manager.send_alert(
                'pnl',
                f"{emoji} Significant P&L Change\n"
                f"Total P&L: {metrics.total_pnl_pct:.2f}%\n"
                f"Balance: ${self.current_balance:.2f}",
                priority=priority
            )

    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.performance_tracker.calculate_metrics(self.current_balance)

    def get_dashboard_data(self) -> Dict:
        """Get all dashboard data"""
        metrics = self.get_metrics()

        return {
            'timestamp': datetime.now().isoformat(),
            'price': self.current_price,
            'balance': self.current_balance,
            'positions': self.positions,
            'open_orders': self.open_orders,
            'metrics': {
                'total_pnl': metrics.total_pnl,
                'total_pnl_pct': metrics.total_pnl_pct,
                'total_trades': metrics.total_trades,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_drawdown': metrics.max_drawdown,
                'current_drawdown': metrics.current_drawdown,
                'sharpe_ratio': metrics.sharpe_ratio,
                'trades_per_hour': metrics.trades_per_hour,
            }
        }

    def print_dashboard(self):
        """Print dashboard to console"""
        data = self.get_dashboard_data()
        metrics = data['metrics']

        print("\033[2J\033[H")  # Clear screen
        print("╔" + "═"*58 + "╗")
        print("║" + "  GRID BOT MONITORING DASHBOARD".center(58) + "║")
        print("╠" + "═"*58 + "╣")
        print(f"║  Last Update: {data['timestamp']:<42} ║")
        print("╠" + "═"*58 + "╣")
        print("║  MARKET DATA".ljust(59) + "║")
        print(f"║    Price: ${self.current_price:,.2f}".ljust(59) + "║")
        print("╠" + "═"*58 + "╣")
        print("║  ACCOUNT".ljust(59) + "║")
        print(f"║    Balance: ${self.current_balance:,.2f}".ljust(59) + "║")
        pnl_color = "\033[92m" if metrics['total_pnl'] >= 0 else "\033[91m"
        print(f"║    Total P&L: {pnl_color}{metrics['total_pnl_pct']:.2f}%\033[0m".ljust(68) + "║")
        print(f"║    Drawdown: {metrics['current_drawdown']:.2f}%".ljust(59) + "║")
        print("╠" + "═"*58 + "╣")
        print("║  TRADING STATS".ljust(59) + "║")
        print(f"║    Trades: {metrics['total_trades']}".ljust(59) + "║")
        print(f"║    Win Rate: {metrics['win_rate']:.1f}%".ljust(59) + "║")
        print(f"║    Profit Factor: {metrics['profit_factor']:.2f}".ljust(59) + "║")
        print(f"║    Trades/Hour: {metrics['trades_per_hour']:.2f}".ljust(59) + "║")
        print("╠" + "═"*58 + "╣")
        print("║  POSITIONS".ljust(59) + "║")
        print(f"║    Open: {len(self.positions)}".ljust(59) + "║")
        print(f"║    Orders: {self.open_orders}".ljust(59) + "║")
        print("╚" + "═"*58 + "╝")


def load_alert_config_from_env() -> AlertConfig:
    """Load alert configuration from environment variables"""
    return AlertConfig(
        telegram_enabled=os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true',
        telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN', ''),
        telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID', ''),
        discord_enabled=os.getenv('DISCORD_ENABLED', 'false').lower() == 'true',
        discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL', ''),
        pnl_alert_threshold=float(os.getenv('PNL_ALERT_THRESHOLD', '5.0')),
        drawdown_alert_threshold=float(os.getenv('DRAWDOWN_ALERT_THRESHOLD', '8.0')),
    )


if __name__ == "__main__":
    # Demo
    async def demo():
        config = load_alert_config_from_env()
        dashboard = MonitoringDashboard(config)

        await dashboard.start()
        await dashboard.update_state(
            price=65000.0,
            balance=4.5,
            positions=[{'side': 'LONG', 'size': 0.001}],
            open_orders=10
        )

        dashboard.print_dashboard()
        await dashboard.stop()

    asyncio.run(demo())
