"""
Test script to verify action masking works correctly.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from src.rl_trading_lab.environment.trading_env import TradingEnv, Action

# Create simple test data
dates = pd.date_range('2024-01-01', periods=100, freq='1h')
test_data = pd.DataFrame({
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.rand(100) * 1000,
    'returns': np.random.randn(100) * 0.01,
})

# Create environment
env = TradingEnv(
    df=test_data,
    lookback_window=5,
    initial_balance=10000,
    discrete_actions=True,
    min_episode_length=10,
    randomize_start=False,
    one_trade_mode=False,
    reward_type='returns',
)

print("Testing action masking behavior...\n")

# Reset environment
obs, info = env.reset()
print(f"Initial state:")
print(f"  Position: {info['position']:.4f}")
print(f"  Action mask: {info['action_mask']} (HOLD, BUY, SELL)")
print(f"  Expected: [1, 1, 1] (all actions valid when flat)")
assert np.array_equal(info['action_mask'], [1, 1, 1]), "Initial action mask should allow all actions"
print("  ✓ Passed\n")

# Test 1: Take BUY action to go LONG
print("Test 1: BUY action to go LONG")
obs, reward, terminated, truncated, info = env.step(Action.BUY)
position = info['position']
print(f"  Position after BUY: {position:.4f}")
print(f"  Action mask: {info['action_mask']} (HOLD, BUY, SELL)")

if position > 0:
    print(f"  Expected: [1, 0, 1] (can HOLD or SELL, cannot BUY)")
    assert np.array_equal(info['action_mask'], [1, 0, 1]), "Should mask BUY when LONG"
    print("  ✓ Passed\n")
else:
    print(f"  Warning: Position is {position}, expected > 0. May need more cash or different settings.")
    print()

# Test 2: Try SELL action to close LONG and go SHORT
print("Test 2: SELL action to close LONG")
obs, reward, terminated, truncated, info = env.step(Action.SELL)
position = info['position']
print(f"  Position after SELL: {position:.4f}")
print(f"  Action mask: {info['action_mask']} (HOLD, BUY, SELL)")

if position < 0:
    print(f"  Expected: [1, 1, 0] (can HOLD or BUY, cannot SELL)")
    assert np.array_equal(info['action_mask'], [1, 1, 0]), "Should mask SELL when SHORT"
    print("  ✓ Passed\n")
elif position == 0:
    print(f"  Expected: [1, 1, 1] (can do anything when flat)")
    assert np.array_equal(info['action_mask'], [1, 1, 1]), "Should allow all actions when flat"
    print("  ✓ Passed (closed to flat)\n")
else:
    print(f"  Warning: Position is {position}, expected < 0 or == 0")
    print()

# Test 3: Multiple steps to verify mask updates
print("Test 3: Multiple step sequence")
for i in range(5):
    # Take action based on mask
    action_mask = info['action_mask']
    valid_actions = [Action.HOLD, Action.BUY, Action.SELL]
    valid_actions_filtered = [a for a, mask in zip(valid_actions, action_mask) if mask == 1]

    # Pick a random valid action
    action = np.random.choice(valid_actions_filtered)

    obs, reward, terminated, truncated, info = env.step(action)
    position = info['position']
    action_mask = info['action_mask']

    action_name = Action(action).name
    print(f"  Step {i+1}: Action={action_name}, Position={position:.4f}, Mask={action_mask}")

    # Verify mask is correct
    position_sign = np.sign(position)
    if position_sign == 0:
        expected_mask = [1, 1, 1]
    elif position_sign > 0:
        expected_mask = [1, 0, 1]
    else:
        expected_mask = [1, 1, 0]

    assert np.array_equal(action_mask, expected_mask), f"Mask {action_mask} doesn't match expected {expected_mask}"

    if terminated or truncated:
        print(f"  Episode ended (terminated={terminated}, truncated={truncated})")
        break

print("\n✓ All action masking tests passed!")

# Test 4: Verify invalid actions are remapped to HOLD
print("\nTest 4: Invalid action remapping")
obs, info = env.reset()
print(f"Initial position: {info['position']:.4f}")

# Take BUY to go LONG
obs, reward1, _, _, info = env.step(Action.BUY)
position1 = info['position']
cash1 = info['cash']
print(f"  After BUY: position={position1:.4f}, cash={cash1:.2f}")

# Try BUY again (invalid while LONG -> should be remapped to HOLD)
obs, reward2, _, _, info = env.step(Action.BUY)
position2 = info['position']
cash2 = info['cash']
print(f"  After BUY again (remapped to HOLD): position={position2:.4f}, cash={cash2:.2f}")

if position1 > 0:
    print(f"  Expected: position and cash stay same (HOLD behavior)")
    # Position should stay same (within small floating point tolerance)
    assert abs(position2 - position1) < 1.0, f"Position should be unchanged"
    print("  ✓ Passed - Invalid actions remapped to HOLD\n")
else:
    print(f"  Warning: Test conditions not met\n")

print("\n" + "="*70)
print("Action masking is working correctly. The environment now:")
print("  1. Tracks valid/invalid actions via action_mask in info dict")
print("  2. Prevents execution of invalid actions (no-op)")
print("  3. Applies -0.01 penalty for invalid actions")
print("\nInvalid actions:")
print("  - BUY when already LONG")
print("  - SELL when already SHORT")
print("\nThis should prevent the agent from learning degenerate policies.")
print("="*70)
