#!/usr/bin/env python3
"""
GRID BOT LAUNCHER
=================
Easy-to-use launcher for the autonomous grid trading bot.
Handles configuration, validation, and startup.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from live_grid_bot import (
    LiveGridBot, BinanceGridAlgoBot,
    LiveGridConfig, load_config_from_env
)
from binance_connector import BinanceConfig
from monitoring import MonitoringDashboard, load_alert_config_from_env


def print_banner():
    """Print startup banner"""
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║            ██████╗ ██████╗ ██╗██████╗     ██████╗  ██████╗ ████████╗
    ║           ██╔════╝ ██╔══██╗██║██╔══██╗    ██╔══██╗██╔═══██╗╚══██╔══╝
    ║           ██║  ███╗██████╔╝██║██║  ██║    ██████╔╝██║   ██║   ██║
    ║           ██║   ██║██╔══██╗██║██║  ██║    ██╔══██╗██║   ██║   ██║
    ║           ╚██████╔╝██║  ██║██║██████╔╝    ██████╔╝╚██████╔╝   ██║
    ║            ╚═════╝ ╚═╝  ╚═╝╚═╝╚═════╝     ╚═════╝  ╚═════╝    ╚═╝
    ║                                                                   ║
    ║              Autonomous Futures Grid Trading System               ║
    ║                      For Binance Futures                          ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)


def print_config(config: LiveGridConfig, binance_config: BinanceConfig):
    """Print configuration summary"""
    print("\n" + "="*60)
    print("  CONFIGURATION")
    print("="*60)
    print(f"  Environment: {'TESTNET' if binance_config.testnet else '⚠️ MAINNET'}")
    print(f"  Symbol: {config.symbol}")
    print(f"  Investment: ${config.total_investment}")
    print(f"  Leverage: {config.leverage}x")
    print(f"  Grid Count: {config.num_grids} each side")
    print(f"  Grid Spacing: {config.grid_spacing_pct}%")
    print(f"  Max Drawdown: {config.max_drawdown_pct}%")
    print(f"  Take Profit: {config.take_profit_pct}%")
    print(f"  Stop Loss: {config.position_stop_loss_pct}%")
    print("="*60 + "\n")


def create_config_file():
    """Create a configuration file template"""
    config_template = {
        "binance": {
            "api_key": "YOUR_API_KEY_HERE",
            "api_secret": "YOUR_API_SECRET_HERE",
            "testnet": True
        },
        "trading": {
            "symbol": "BTCUSDT",
            "total_investment": 4.0,
            "leverage": 20,
            "num_grids": 10,
            "grid_spacing_pct": 0.3,
            "max_drawdown_pct": 10.0,
            "position_stop_loss_pct": 3.0,
            "take_profit_pct": 0.5,
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.3
        },
        "alerts": {
            "telegram_enabled": False,
            "telegram_bot_token": "",
            "telegram_chat_id": "",
            "discord_enabled": False,
            "discord_webhook_url": ""
        }
    }

    config_path = Path("config.json")
    if config_path.exists():
        print(f"Config file already exists at: {config_path}")
        return

    with open(config_path, 'w') as f:
        json.dump(config_template, f, indent=2)

    print(f"Created config template at: {config_path}")
    print("Please edit this file with your API credentials before starting the bot.")


def load_config_from_file(path: str = "config.json"):
    """Load configuration from JSON file"""
    config_path = Path(path)

    if not config_path.exists():
        return None, None

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    binance_config = BinanceConfig(
        api_key=config_data['binance']['api_key'],
        api_secret=config_data['binance']['api_secret'],
        testnet=config_data['binance'].get('testnet', True)
    )

    trading = config_data.get('trading', {})
    live_config = LiveGridConfig(
        symbol=trading.get('symbol', 'BTCUSDT'),
        total_investment=trading.get('total_investment', 4.0),
        leverage=trading.get('leverage', 20),
        num_grids=trading.get('num_grids', 10),
        grid_spacing_pct=trading.get('grid_spacing_pct', 0.3),
        max_drawdown_pct=trading.get('max_drawdown_pct', 10.0),
        position_stop_loss_pct=trading.get('position_stop_loss_pct', 3.0),
        take_profit_pct=trading.get('take_profit_pct', 0.5),
        use_trailing_stop=trading.get('use_trailing_stop', True),
        trailing_stop_pct=trading.get('trailing_stop_pct', 0.3),
    )

    return live_config, binance_config


def validate_config(config: LiveGridConfig, binance_config: BinanceConfig) -> bool:
    """Validate configuration"""
    errors = []

    if not binance_config.api_key or binance_config.api_key == "YOUR_API_KEY_HERE":
        errors.append("API key is not set")

    if not binance_config.api_secret or binance_config.api_secret == "YOUR_API_SECRET_HERE":
        errors.append("API secret is not set")

    if config.leverage < 1 or config.leverage > 125:
        errors.append(f"Invalid leverage: {config.leverage}")

    if config.total_investment <= 0:
        errors.append(f"Invalid investment amount: {config.total_investment}")

    if config.num_grids < 2:
        errors.append(f"Need at least 2 grids, got: {config.num_grids}")

    if errors:
        print("\n⚠️ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def interactive_setup():
    """Interactive configuration setup"""
    print("\n📋 Interactive Setup")
    print("-" * 40)

    # API credentials
    print("\n1. Binance API Credentials")
    api_key = input("   API Key: ").strip()
    api_secret = input("   API Secret: ").strip()
    testnet = input("   Use Testnet? (yes/no) [yes]: ").strip().lower()
    testnet = testnet != 'no'

    # Trading parameters
    print("\n2. Trading Parameters")
    symbol = input("   Symbol [BTCUSDT]: ").strip() or "BTCUSDT"
    investment = float(input("   Investment Amount [$4]: ").strip() or "4")
    leverage = int(input("   Leverage [20]: ").strip() or "20")
    num_grids = int(input("   Number of Grids [10]: ").strip() or "10")
    spacing = float(input("   Grid Spacing % [0.3]: ").strip() or "0.3")

    # Risk management
    print("\n3. Risk Management")
    max_dd = float(input("   Max Drawdown % [10]: ").strip() or "10")
    stop_loss = float(input("   Stop Loss % [3]: ").strip() or "3")
    take_profit = float(input("   Take Profit % [0.5]: ").strip() or "0.5")

    # Create configs
    binance_config = BinanceConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet
    )

    live_config = LiveGridConfig(
        symbol=symbol,
        total_investment=investment,
        leverage=leverage,
        num_grids=num_grids,
        grid_spacing_pct=spacing,
        max_drawdown_pct=max_dd,
        position_stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )

    # Save config
    save = input("\nSave configuration to file? (yes/no) [yes]: ").strip().lower()
    if save != 'no':
        config_data = {
            "binance": {
                "api_key": api_key,
                "api_secret": api_secret,
                "testnet": testnet
            },
            "trading": {
                "symbol": symbol,
                "total_investment": investment,
                "leverage": leverage,
                "num_grids": num_grids,
                "grid_spacing_pct": spacing,
                "max_drawdown_pct": max_dd,
                "position_stop_loss_pct": stop_loss,
                "take_profit_pct": take_profit,
            }
        }
        with open("config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        print("✅ Configuration saved to config.json")

    return live_config, binance_config


async def run_bot(mode: str = "custom"):
    """Run the trading bot"""
    print_banner()

    # Load configuration
    live_config, binance_config = load_config_from_file()

    if live_config is None:
        print("\nNo config.json found.")
        choice = input("Create config interactively? (yes/no): ").strip().lower()
        if choice == 'yes':
            live_config, binance_config = interactive_setup()
        else:
            # Try environment variables
            live_config, binance_config = load_config_from_env()

    if not validate_config(live_config, binance_config):
        print("\n❌ Invalid configuration. Please fix the errors and try again.")
        return

    print_config(live_config, binance_config)

    # Confirmation
    if not binance_config.testnet:
        print("\n⚠️ WARNING: You are about to trade on MAINNET with REAL money!")
        confirm = input("Type 'I UNDERSTAND THE RISKS' to continue: ").strip()
        if confirm != 'I UNDERSTAND THE RISKS':
            print("Aborted.")
            return
    else:
        print("ℹ️ Running on TESTNET (paper trading)")
        confirm = input("Press Enter to start or 'q' to quit: ").strip()
        if confirm.lower() == 'q':
            print("Aborted.")
            return

    # Start bot based on mode
    if mode == "native":
        # Use Binance's native grid algo
        print("\n🚀 Starting Binance Native Grid Algo...")
        bot = BinanceGridAlgoBot(live_config, binance_config)
        await bot.run()
    else:
        # Use custom grid bot
        print("\n🚀 Starting Custom Grid Bot...")
        bot = LiveGridBot(live_config, binance_config)

        try:
            await bot.run()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutdown requested...")
        finally:
            await bot.stop()
            print("✅ Bot stopped successfully")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Grid Trading Bot Launcher")
    parser.add_argument(
        '--mode',
        choices=['custom', 'native', 'setup'],
        default='custom',
        help='Bot mode: custom (our grid logic), native (Binance Algo API), setup (create config)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )

    args = parser.parse_args()

    if args.mode == 'setup':
        print_banner()
        create_config_file()
        return

    asyncio.run(run_bot(args.mode))


if __name__ == "__main__":
    main()
