"""
Simple diagnostic: load model and check invalid action rate.
"""
import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from src.rl_trading_lab.environment.trading_env import TradingEnv, Action

# Create simple test environment
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

# Load model
model_path = "checkpoints/A2C_returns_20251030_210908/best_model/best_model.zip"
print(f"Loading model: {model_path}\n")
model = A2C.load(model_path)

print("="*80)
print("Running episode to diagnose invalid actions...")
print("="*80 + "\n")

obs, info = env.reset()
done = False
step = 0
invalid_count = 0
action_counts = {Action.HOLD: 0, Action.BUY: 0, Action.SELL: 0}
rewards = []

invalid_log = []

while not done and step < 200:
    action, _ = model.predict(obs, deterministic=False)

    # Check if valid
    action_mask = info['action_mask']
    is_valid = action_mask[action] == 1

    if not is_valid:
        invalid_count += 1
        pos = info['position']
        pos_type = "LONG" if pos > 0 else "SHORT" if pos < 0 else "FLAT"
        invalid_log.append({
            'step': step,
            'pos': pos,
            'pos_type': pos_type,
            'action': Action(action).name,
        })

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    action_counts[Action(action)] += 1
    rewards.append(reward)
    step += 1

print(f"Episode: {step} steps\n")

print(f"Action Distribution:")
for act, cnt in action_counts.items():
    print(f"  {act.name:5s}: {cnt:4d} ({100*cnt/step:5.1f}%)")

print(f"\n❌ Invalid Actions: {invalid_count} / {step} ({100*invalid_count/step:.1f}%)\n")

if invalid_count > 0:
    print("First 10 invalid actions:")
    for log in invalid_log[:10]:
        print(f"  Step {log['step']:3d}: {log['pos_type']:5s} ({log['pos']:+8.2f}) -> {log['action']:4s}")

rewards_arr = np.array(rewards)
print(f"\nReward stats:")
print(f"  Mean: {rewards_arr.mean():.6f}")
print(f"  Std:  {rewards_arr.std():.6f}")
print(f"  Penalty count (-0.01): {np.sum(rewards_arr == -0.01)}")

print("\n" + "="*80)
print("PROBLEM IDENTIFIED:")
print("="*80)
print("The agent is taking invalid actions despite the penalty.")
print("This means the penalty is NOT being learned properly during training.")
print("\nPossible reasons:")
print("1. Penalty happens AFTER invalid action in history (confusing signal)")
print("2. A2C doesn't see the action mask during training")
print("3. The invalid action is stored in history but not the invalidity")
print("\nWe need a different approach - either:")
print("A. Use MaskablePPO (sb3-contrib) which properly handles action masking")
print("B. Modify environment to force close position on same-direction action")
print("C. Increase penalty significantly (e.g., -1.0 or -10.0)")
