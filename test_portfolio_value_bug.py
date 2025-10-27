#!/usr/bin/env python
"""
Test to demonstrate the portfolio value calculation bug.
"""

import pandas as pd
import numpy as np
from rl_trading_lab.environment.trading_env import TradingEnv

# Create simple test data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='1h')
df = pd.DataFrame({
    'timestamp': dates,
    'open': 100 + np.random.randn(100),
    'high': 102 + np.random.randn(100),
    'low': 98 + np.random.randn(100),
    'close': 100 + np.random.randn(100),
    'volume': 1000 + np.random.randn(100) * 100,
})

# Add some features
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
)

# Reset and take a few steps
obs, info = env.reset()
print(f"Initial state:")
print(f"  Balance: ${env.balance:.2f}")
print(f"  Position size: {env.position.size:.4f}")
print(f"  Portfolio value: ${env._get_portfolio_value():.2f}")
print(f"  Expected: $10,000.00")
print()

# Execute a BUY action (action=1)
print(f"Step 1: BUY (action=1)")
current_price = env._get_current_price()
print(f"  Current price: ${current_price:.2f}")

obs, reward, terminated, truncated, info = env.step(1)  # Buy

print(f"After BUY:")
print(f"  Balance: ${env.balance:.2f}")
print(f"  Position size: {env.position.size:.4f}")
print(f"  Position entry price: ${env.position.entry_price:.2f}")
print(f"  Portfolio value: ${env._get_portfolio_value():.2f}")

# Calculate expected values
expected_position_size = (10000 * 0.95) / env.position.entry_price
expected_trade_value = abs(expected_position_size * env.position.entry_price)
expected_commission = expected_trade_value * 0.001
print()
print(f"Expected calculations:")
print(f"  Position value: ${expected_trade_value:.2f}")
print(f"  Commission: ${expected_commission:.2f}")
print(f"  Expected balance (if capital locked): ${10000 - expected_trade_value - expected_commission:.2f}")
print(f"  ACTUAL balance: ${env.balance:.2f}")
print(f"  BUG: Capital was NOT locked! Balance only decreased by commission.")
print()

# Take another step with HOLD
obs, reward, terminated, truncated, info = env.step(0)  # Hold
print(f"Step 2: HOLD (action=0)")
print(f"  Current price: ${env._get_current_price():.2f}")
print(f"  Portfolio value: ${env._get_portfolio_value():.2f}")
print(f"  Unrealized PnL: ${env.position.size * (env._get_current_price() - env.position.entry_price):.2f}")
print()

# Close position with SELL
print(f"Step 3: SELL (action=2) - This closes the long position")
obs, reward, terminated, truncated, info = env.step(2)  # Sell
print(f"After SELL:")
print(f"  Balance: ${env.balance:.2f}")
print(f"  Position size: {env.position.size:.4f}")
print(f"  Portfolio value: ${env._get_portfolio_value():.2f}")
print(f"  Expected portfolio value: ${10000 + info.get('total_return', 0) * 10000:.2f}")
print()

print(f"SUMMARY:")
print(f"  Initial balance: $10,000.00")
print(f"  Final portfolio value: ${env._get_portfolio_value():.2f}")
print(f"  Total return (from info): {info.get('total_return', 0):.4%}")
print(f"  Actual return: ${env._get_portfolio_value() - 10000:.2f}")
print()
print(f"DIAGNOSIS:")
print(f"  The portfolio value calculation is BROKEN because:")
print(f"  1. When opening a position, capital is NOT deducted from balance")
print(f"  2. Only commission is deducted")
print(f"  3. _get_portfolio_value() calculates position_value but doesn't use it!")
print(f"  4. This causes portfolio value to be incorrectly calculated during backtesting")