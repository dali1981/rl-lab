#!/usr/bin/env python
"""
Test script to demonstrate logging level control.
Shows trade logs at DEBUG level, hides them at INFO level.
"""

import logging
import numpy as np
import pandas as pd
from rl_trading_lab.environment.trading_env import TradingEnv

# Test both log levels
for level_name, level in [("INFO", logging.INFO), ("DEBUG", logging.DEBUG)]:
    print("\n" + "="*60)
    print(f"Testing with log level: {level_name}")
    print("="*60 + "\n")

    # Configure logging
    logging.basicConfig(level=level, force=True, format='[%(levelname)s] %(name)s: %(message)s')

    # Create simple test data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=50, freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(50) * 0.5),
        'high': 102 + np.cumsum(np.random.randn(50) * 0.5),
        'low': 98 + np.cumsum(np.random.randn(50) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(50) * 0.5),
        'volume': 1000 + np.random.randn(50) * 100,
    })

    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()

    # Create environment
    env = TradingEnv(
        df=df,
        lookback_window=5,
        initial_balance=10000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        reward_type="returns",
        discrete_actions=True,
        randomize_start=False,
        hold_closes_position=True,
    )

    # Run a few trades
    env.reset()
    env.step(1)  # Buy (should log "Trade #1" at DEBUG)
    env.step(1)  # Buy again (no new trade)
    env.step(2)  # Sell (should log "Position closed" and "Trade #2" at DEBUG)
    env.step(0)  # Hold (should log "Position closed" at DEBUG)

    print(f"\nTotal trades executed: {env.num_trades}")
    print(f"Final portfolio value: ${env._get_portfolio_value():.2f}\n")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\nINFO level (default):")
print("  ✓ Shows: Environment initialization")
print("  ✗ Hides: Individual trade logs")
print("  → Clean output for training\n")

print("DEBUG level:")
print("  ✓ Shows: Environment initialization")
print("  ✓ Shows: Every trade execution, position change")
print("  → Detailed output for debugging\n")

print("To run with DEBUG level:")
print("  PYTHONLOGLEVEL=DEBUG python test_logging_levels.py\n")
