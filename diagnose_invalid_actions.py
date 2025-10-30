"""
Diagnose why agent is still taking invalid actions.
Load trained model and analyze behavior.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import A2C

from src.rl_trading_lab.environment.trading_env import Action
from src.rl_trading_lab.environment.factory import create_make_env
from hydra import compose, initialize
from omegaconf import OmegaConf

# Load config
with initialize(config_path="configs", version_base=None):
    config = compose(config_name="config", overrides=["agent=a2c"])

# Create environment factory
make_env = create_make_env(
    data_path=config.data.train_path,
    observation_config=config.observation,
    feature_engineering_config=config.feature_engineering,
    env_config=config.env,
    val_split=config.data.val_split,
    test_split=config.data.test_split,
)

# Create test environment
env = make_env('train')

# Load model
model_path = "checkpoints/A2C_returns_20251030_210908/best_model/best_model.zip"
print(f"Loading model from {model_path}")
model = A2C.load(model_path)

print("\n" + "="*80)
print("Running episode and tracking invalid actions...")
print("="*80 + "\n")

obs, info = env.reset()
done = False
step = 0
invalid_action_count = 0
action_counts = {Action.HOLD: 0, Action.BUY: 0, Action.SELL: 0}
rewards_list = []
invalid_actions_log = []

while not done and step < 200:
    # Get action from model
    action, _ = model.predict(obs, deterministic=False)

    # Get current position and action mask
    position = info['position']
    action_mask = info['action_mask']

    # Check if action is valid
    is_valid = action_mask[action] == 1

    if not is_valid:
        invalid_action_count += 1
        position_type = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"
        invalid_actions_log.append({
            'step': step,
            'position': position,
            'position_type': position_type,
            'action': Action(action).name,
            'action_mask': action_mask.tolist(),
        })

    # Take step
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Track statistics
    action_counts[Action(action)] += 1
    rewards_list.append(reward)
    step += 1

print(f"Episode completed: {step} steps")
print(f"\nAction Distribution:")
for action_type, count in action_counts.items():
    pct = 100 * count / step
    print(f"  {action_type.name:5s}: {count:4d} ({pct:5.1f}%)")

print(f"\nInvalid Actions: {invalid_action_count} / {step} ({100*invalid_action_count/step:.1f}%)")

if invalid_action_count > 0:
    print("\nInvalid Action Details (first 20):")
    for log in invalid_actions_log[:20]:
        print(f"  Step {log['step']:3d}: {log['position_type']:5s} position ({log['position']:+8.2f}) "
              f"tried {log['action']:4s}, mask={log['action_mask']}")

print(f"\nReward Statistics:")
rewards_array = np.array(rewards_list)
print(f"  Mean reward: {rewards_array.mean():.6f}")
print(f"  Std reward:  {rewards_array.std():.6f}")
print(f"  Min reward:  {rewards_array.min():.6f}")
print(f"  Max reward:  {rewards_array.max():.6f}")

# Count how many times we got the -0.01 penalty
penalty_count = np.sum(rewards_array == -0.01)
print(f"\nPenalty (-0.01) occurrences: {penalty_count}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

if invalid_action_count > 0:
    avg_reward_magnitude = np.abs(rewards_array[rewards_array != -0.01]).mean()
    print(f"❌ Agent is still taking invalid actions!")
    print(f"\nProblem: Penalty of -0.01 is too small")
    print(f"  - Average reward magnitude: {avg_reward_magnitude:.6f}")
    print(f"  - Penalty magnitude: 0.01")
    print(f"  - Ratio: {0.01/avg_reward_magnitude:.2f}x")
    print(f"\nRecommendation: Increase penalty to at least -0.05 or -0.1")
else:
    print("✓ Agent successfully avoids invalid actions!")
