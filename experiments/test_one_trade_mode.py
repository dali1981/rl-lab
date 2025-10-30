#!/usr/bin/env python
"""
Quick test script to verify ONE_TRADE mode functionality.

Tests that:
1. Episode terminates after first position close when one_trade_mode=True
2. Episode continues normally when one_trade_mode=False
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_trading_lab.environment.trading_env import TradingEnv, Action


def create_test_data(n_bars=500):
    """Create simple test data"""
    np.random.seed(42)

    # Simple random walk price
    returns = np.random.randn(n_bars) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'close': prices,
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
        'feature_1': np.random.randn(n_bars),
        'feature_2': np.random.randn(n_bars),
    })

    return df


def test_one_trade_mode_enabled():
    """Test that episode terminates after first position close"""
    print("\n=== Test 1: ONE_TRADE Mode ENABLED ===")

    df = create_test_data()

    env = TradingEnv(
        df=df,
        lookback_window=20,
        initial_balance=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        reward_type="returns",
        discrete_actions=True,
        randomize_start=False,
        min_episode_length=50,
        min_holding_period=1,
        hold_closes_position=True,
        one_trade_mode=True,  # ENABLE ONE_TRADE MODE
    )

    obs, info = env.reset()

    # Force a trade sequence: BUY -> hold for a bit -> SELL
    actions = [Action.BUY] + [Action.HOLD] * 5 + [Action.HOLD]

    terminated = False
    truncated = False
    step_count = 0

    print(f"Initial state: position={info['position']:.4f}")

    for i, action in enumerate(actions):
        if terminated or truncated:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        print(f"Step {step_count}: action={Action(action).name}, "
              f"position={info['position']:.4f}, "
              f"terminated={terminated}, truncated={truncated}")

        if terminated:
            print(f"✓ Episode terminated after {step_count} steps (expected behavior)")
            break

    # Verify that episode terminated after position closed
    if terminated and step_count < len(actions):
        print("✓ Test PASSED: Episode terminated after first position close")
        return True
    else:
        print(f"✗ Test FAILED: Episode did not terminate (steps={step_count}, terminated={terminated})")
        return False


def test_one_trade_mode_disabled():
    """Test that episode continues normally with one_trade_mode=False"""
    print("\n=== Test 2: ONE_TRADE Mode DISABLED ===")

    df = create_test_data()

    env = TradingEnv(
        df=df,
        lookback_window=20,
        initial_balance=10000,
        commission_rate=0.0,
        slippage_rate=0.0,
        reward_type="returns",
        discrete_actions=True,
        randomize_start=False,
        min_episode_length=50,
        min_holding_period=1,
        hold_closes_position=True,
        one_trade_mode=False,  # DISABLE ONE_TRADE MODE
    )

    obs, info = env.reset()

    # Same trade sequence: BUY -> hold for a bit -> SELL -> continue
    actions = [Action.BUY] + [Action.HOLD] * 5 + [Action.HOLD] + [Action.HOLD] * 10

    terminated = False
    truncated = False
    step_count = 0
    position_closed = False

    print(f"Initial state: position={info['position']:.4f}")

    for i, action in enumerate(actions):
        if terminated or truncated:
            break

        prev_position = info['position']
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        # Track when position closes
        if abs(prev_position) > 0.001 and abs(info['position']) < 0.001:
            position_closed = True
            print(f"  -> Position closed at step {step_count}")

        if step_count % 5 == 0:
            print(f"Step {step_count}: position={info['position']:.4f}, "
                  f"terminated={terminated}")

    # Verify that episode continued after position closed
    if position_closed and not terminated:
        print(f"✓ Test PASSED: Episode continued after position close ({step_count} steps)")
        return True
    else:
        print(f"✗ Test FAILED: Unexpected termination (position_closed={position_closed}, "
              f"terminated={terminated})")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing ONE_TRADE Mode Implementation")
    print("=" * 60)

    test1_passed = test_one_trade_mode_enabled()
    test2_passed = test_one_trade_mode_disabled()

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Test 1 (ONE_TRADE enabled):  {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Test 2 (ONE_TRADE disabled): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n✓ All tests PASSED")
        return 0
    else:
        print("\n✗ Some tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
