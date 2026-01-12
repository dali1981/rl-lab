#!/usr/bin/env python
"""
Debug script to run an episode and diagnose issues.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rl_trading_lab.environment.trading_env import TradingEnv
from rl_trading_lab.utils.data_loader import TradingDataLoader
from rl_trading_lab.config import load_config
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

plt.style.use('seaborn-v0_8-darkgrid')

# Load config using Hydra
print("Loading configuration...")
config_dir = str(Path(__file__).parent / "configs")
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name="config")
    config = load_config(cfg)

print(f"Reward type: {config.env.environment_params.reward_type}")
print(f"Initial balance: ${config.env.environment_params.initial_balance:,.2f}")
print(f"Commission rate: {config.env.environment_params.commission_rate}")

# Load data
print("\nLoading data...")
data_loader = TradingDataLoader(
    data_path=config.data.train_data_path,
    features_config=config.features,
    val_split=config.data.val_split,
    test_split=config.data.test_split,
)

train_df, val_df, test_df, feature_names = data_loader.load_and_prepare()

print(f"Train: {len(train_df)} bars")
print(f"Val: {len(val_df)} bars")
print(f"Test: {len(test_df)} bars")
print(f"Features: {len(feature_names)}")

# Create test environment
print("\nCreating environment...")
env_params = config.env.environment_params

test_env = TradingEnv(
    df=test_df,
    lookback_window=env_params.lookback_window,
    initial_balance=env_params.initial_balance,
    commission_rate=env_params.commission_rate,
    slippage_rate=env_params.slippage_rate,
    reward_type=env_params.reward_type,
    discrete_actions=env_params.discrete_actions,
    max_position_pct=env_params.max_position_pct,
    features_to_use=feature_names,
    randomize_start=False,
    min_episode_length=env_params.min_episode_length,
    hold_closes_position=env_params.hold_closes_position,
)

print(f"Environment created")
print(f"Max steps: {test_env.max_steps}")
print(f"Hold closes position: {test_env.hold_closes_position}")

# Try to load trained agent
print("\nAttempting to load trained agent...")
model_path = "checkpoints/PPO_sharpe_20251027_113016/best_model/best_model.zip"

agent_available = False
try:
    model = PPO.load(model_path)

    # Wrap test env to match training
    test_env_func = lambda: test_env
    test_vec_env = DummyVecEnv([test_env_func])
    test_vec_env = VecNormalize(
        test_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        training=False,
    )

    # Try to load normalization stats
    vecnormalize_path = model_path.replace('best_model.zip', 'vecnormalize.pkl')
    try:
        test_vec_env = VecNormalize.load(vecnormalize_path, test_vec_env)
        print(f"✓ Loaded normalization stats from {vecnormalize_path}")
    except:
        print("⚠️  Warning: Could not load normalization stats")

    print(f"✓ Loaded model from {model_path}")
    agent_available = True
except Exception as e:
    print(f"✗ Could not load agent: {e}")
    print("Using random actions instead")

# Run episode
print("\n" + "="*60)
print("RUNNING EPISODE")
print("="*60)

def run_agent_episode(vec_env, model):
    """Run episode with trained agent"""
    obs = vec_env.reset()

    # Get the unwrapped environment for direct access
    unwrapped_env = vec_env.venv.envs[0]

    data = {
        'step': [],
        'action': [],
        'reward': [],
        'position': [],
        'balance': [],
        'portfolio_value': [],
        'price': [],
        'terminated': [],
    }

    done = False
    step = 0

    while not done:
        # Get action from agent
        action, _ = model.predict(obs, deterministic=True)

        # Step
        obs, reward, done, info = vec_env.step(action)

        # Extract from vectorized format
        if isinstance(done, np.ndarray):
            done = done[0]
        if isinstance(action, np.ndarray):
            action = action[0]
        if isinstance(reward, np.ndarray):
            reward = reward[0]

        # Collect data from unwrapped env
        data['step'].append(step)
        data['action'].append(action)
        data['reward'].append(reward)
        data['position'].append(unwrapped_env.position.size)
        data['balance'].append(unwrapped_env.balance)
        data['portfolio_value'].append(unwrapped_env._get_portfolio_value())
        data['price'].append(unwrapped_env._get_current_price())
        data['terminated'].append(done)

        step += 1

        if step > 500:  # Safety limit
            break

    # Close any remaining positions (like the backtest does)
    unwrapped_env.close_all_positions()

    return pd.DataFrame(data)

def run_random_episode(env, seed=42):
    """Run episode with random actions"""
    obs, info = env.reset(seed=seed)

    data = {
        'step': [],
        'action': [],
        'reward': [],
        'position': [],
        'balance': [],
        'portfolio_value': [],
        'price': [],
        'terminated': [],
    }

    done = False
    truncated = False
    step = 0

    while not done and not truncated:
        # Random action
        action = env.action_space.sample()

        # Step
        obs, reward, done, truncated, info = env.step(action)

        # Collect data
        data['step'].append(step)
        data['action'].append(action)
        data['reward'].append(reward)
        data['position'].append(env.position.size)
        data['balance'].append(env.balance)
        data['portfolio_value'].append(env._get_portfolio_value())
        data['price'].append(env._get_current_price())
        data['terminated'].append(done)

        step += 1

        if step > 500:  # Safety limit
            break

    # Close any remaining positions (like the backtest does)
    env.close_all_positions()

    return pd.DataFrame(data)

# Run the episode
if agent_available:
    print("Running with trained agent...")
    df = run_agent_episode(test_vec_env, model)
else:
    print("Running with random actions...")
    df = run_random_episode(test_env)

print(f"Episode completed: {len(df)} steps")

# Analysis
print("\n" + "="*60)
print("EPISODE SUMMARY")
print("="*60)
print(f"Total steps: {len(df)}")
print(f"Initial balance: ${env_params.initial_balance:,.2f}")
print(f"Final portfolio value: ${df['portfolio_value'].iloc[-1]:,.2f}")
print(f"Final return: {(df['portfolio_value'].iloc[-1] / env_params.initial_balance - 1) * 100:.2f}%")
print()

print("Action Distribution:")
print(df['action'].value_counts().sort_index())
print()

print("Reward Statistics:")
print(f"  Mean: {df['reward'].mean():.4f}")
print(f"  Std: {df['reward'].std():.4f}")
print(f"  Min: {df['reward'].min():.4f}")
print(f"  Max: {df['reward'].max():.4f}")
print()

print("Position Statistics:")
print(f"  Final position: {df['position'].iloc[-1]:.4f}")
print(f"  Max position: {df['position'].max():.4f}")
print(f"  Min position: {df['position'].min():.4f}")
print(f"  % of time in position: {(df['position'] != 0).sum() / len(df) * 100:.1f}%")
print()

# Get actual environment state AFTER close_all_positions() was called
if agent_available:
    unwrapped_env = test_vec_env.venv.envs[0]
else:
    unwrapped_env = test_env

# Check actual position state (should be 0 after close_all_positions())
actual_final_position = unwrapped_env.position.size
actual_final_portfolio = unwrapped_env._get_portfolio_value()
actual_final_return = (actual_final_portfolio / env_params.initial_balance - 1) * 100

print(f"\nPosition state DURING episode (from data): {df['position'].iloc[-1]:.4f}")
print(f"Position state AFTER close_all_positions(): {actual_final_position:.4f}")
print()

if abs(actual_final_position) > 0.01:
    print("⚠️  ERROR: Position still open after close_all_positions()!")
else:
    print("✓ Position successfully closed at episode end")
    print(f"  Final portfolio value: ${actual_final_portfolio:.2f}")
    print(f"  Final return (realized): {actual_final_return:.2f}%")
print()

# Check accounting
print("\n" + "="*60)
print("ACCOUNTING CHECK")
print("="*60)
df['position_value'] = abs(df['position']) * df['price']
df['balance_plus_position'] = df['balance'] + df['position_value']

diff = (df['portfolio_value'] - df['balance_plus_position']).abs().max()
print(f"Max difference between env.portfolio_value and balance+position: ${diff:.2f}")
if diff > 1.0:
    print("⚠️  Large discrepancy detected! This suggests an accounting bug.")
else:
    print("✓ Values match (within $1 tolerance)")

# Save results
output_file = "debug_episode_results.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Results saved to {output_file}")

# Create plots
print("\nGenerating plots...")
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# 1. Portfolio Value
axes[0].plot(df['step'], df['portfolio_value'], label='Portfolio Value', linewidth=2)
axes[0].axhline(y=env_params.initial_balance, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].set_title('Portfolio Value Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Price
axes[1].plot(df['step'], df['price'], label='Price', color='orange', linewidth=1.5)
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Asset Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Actions
action_colors = {0: 'gray', 1: 'green', 2: 'red'}
action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
for action_val in [0, 1, 2]:
    mask = df['action'] == action_val
    axes[2].scatter(df[mask]['step'], df[mask]['action'],
                   c=action_colors[action_val], label=action_labels[action_val], alpha=0.6, s=20)
axes[2].set_ylabel('Action')
axes[2].set_title('Actions Taken')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['Hold', 'Buy', 'Sell'])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Position
axes[3].plot(df['step'], df['position'], label='Position Size', color='purple', linewidth=1.5)
axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[3].set_ylabel('Position Size')
axes[3].set_title('Position Size Over Time')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# 5. Rewards
axes[4].plot(df['step'], df['reward'], label='Reward', color='blue', alpha=0.7, linewidth=1)
axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[4].set_ylabel('Reward')
axes[4].set_xlabel('Step')
axes[4].set_title('Rewards')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = 'debug_episode_plot.png'
plt.savefig(plot_file, dpi=150)
print(f"✓ Plot saved to {plot_file}")

print("\n" + "="*60)
print("DONE")
print("="*60)
