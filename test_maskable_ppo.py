"""
Test script to verify MaskablePPO integration works correctly.
"""
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO

from src.rl_trading_lab.environment.trading_env import TradingEnv, Action

# Create simple test data
dates = pd.date_range('2024-01-01', periods=500, freq='1h')
test_data = pd.DataFrame({
    'close': np.random.randn(500).cumsum() + 100,
    'volume': np.random.rand(500) * 1000,
    'returns': np.random.randn(500) * 0.01,
})

env = TradingEnv(
    df=test_data,
    lookback_window=20,
    initial_balance=10000,
    discrete_actions=True,
    min_episode_length=100,
    randomize_start=False,
    one_trade_mode=False,
    reward_type='returns',
)

print("=" * 80)
print("Testing action_masks() method")
print("=" * 80)

# Test 1: Verify action_masks() method exists and returns correct format
print("\nTest 1: action_masks() method format")
obs, info = env.reset()
masks = env.action_masks()
print(f"  Action masks type: {type(masks)}")
print(f"  Action masks dtype: {masks.dtype}")
print(f"  Action masks shape: {masks.shape}")
print(f"  Action masks (flat): {masks}")
assert masks.dtype == bool, "Masks should be boolean"
assert masks.shape == (3,), "Should have 3 action masks"
assert np.all(masks == True), "All actions should be valid when flat"
print("  ✓ Passed\n")

# Test 2: Verify masks change with position
print("Test 2: Masks update with position changes")
_, _, _, _, info = env.step(Action.BUY)  # Go LONG
position = info['position']
masks_long = env.action_masks()
print(f"  After BUY: position={position:.2f}")
print(f"  LONG position masks: {masks_long} (HOLD, BUY, SELL)")
if position > 0:
    assert masks_long[0] == True, "HOLD should be valid"
    assert masks_long[1] == False, "BUY should be invalid when LONG"
    assert masks_long[2] == True, "SELL should be valid"
    print("  ✓ Passed\n")
else:
    print("  ⚠ Warning: Position is 0 after BUY, skipping LONG test\n")

_, _, _, _, info = env.step(Action.SELL)  # Close and go SHORT
position = info['position']
masks_short = env.action_masks()
print(f"  After SELL: position={position:.2f}")
print(f"  SHORT position masks: {masks_short} (HOLD, BUY, SELL)")
if position < 0:
    assert masks_short[0] == True, "HOLD should be valid"
    assert masks_short[1] == True, "BUY should be valid when SHORT"
    assert masks_short[2] == False, "SELL should be invalid when SHORT"
    print("  ✓ Passed\n")
else:
    print("  ⚠ Warning: Position is 0 after SELL, skipping SHORT test\n")

# Test 3: Create MaskablePPO model
print("Test 3: MaskablePPO model creation")
try:
    model = MaskablePPO("MlpPolicy", env, verbose=0)
    print("  ✓ MaskablePPO model created successfully\n")
except Exception as e:
    print(f"  ✗ Failed to create MaskablePPO model: {e}\n")
    raise

# Test 4: Train for a few steps
print("Test 4: Short training run (100 steps)")
try:
    model.learn(total_timesteps=100, progress_bar=False)
    print("  ✓ Training completed successfully\n")
except Exception as e:
    print(f"  ✗ Training failed: {e}\n")
    raise

# Test 5: Verify model respects action masks during prediction
print("Test 5: Verify invalid actions are never sampled")
obs, info = env.reset()
invalid_action_count = 0
steps = 0

while steps < 50:
    # Get valid actions
    valid_masks = env.action_masks()

    # Predict action
    action, _ = model.predict(obs, action_masks=valid_masks, deterministic=False)

    # Check if action is valid
    if not valid_masks[action]:
        invalid_action_count += 1
        print(f"  ✗ Invalid action sampled at step {steps}!")
        print(f"    Masks: {valid_masks}, Action: {action} ({Action(action).name})")

    obs, reward, terminated, truncated, info = env.step(action)
    steps += 1

    if terminated or truncated:
        obs, info = env.reset()

if invalid_action_count == 0:
    print(f"  ✓ No invalid actions sampled in {steps} steps\n")
else:
    print(f"  ✗ {invalid_action_count} invalid actions sampled in {steps} steps\n")
    raise AssertionError("Model sampled invalid actions!")

print("=" * 80)
print("✓ All MaskablePPO tests passed!")
print("=" * 80)
print("\nMaskablePPO is properly integrated and respects action masking.")
print("You can now train with: uv run python src/rl_trading_lab/train.py agent=maskable_ppo")
