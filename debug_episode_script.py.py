#!/usr/bin/env python
# coding: utf-8

# # Debug Trading Environment Episode
# 
# Run an episode and visualize actions, rewards, positions, and portfolio value.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from rl_trading_lab.environment.trading_env import TradingEnv
from rl_trading_lab.utils.data_loader import TradingDataLoader
from rl_trading_lab.config import load_config
from omegaconf import OmegaConf

# For loading trained agent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

plt.style.use('seaborn-v0_8-darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Configuration and Data

# In[ ]:


# Load config
cfg = OmegaConf.load('configs/config.yaml')
config = load_config(cfg)

print(f"Reward type: {config.env.environment_params.reward_type}")
print(f"Initial balance: ${config.env.environment_params.initial_balance:,.2f}")
print(f"Commission rate: {config.env.environment_params.commission_rate}")


# In[ ]:


# Load data
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


# ## Create Environment

# In[ ]:


# Create test environment
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


# ## Option 1: Run with Random Actions

# In[ ]:


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

    return pd.DataFrame(data)

# Run random episode
df_random = run_random_episode(test_env)
print(f"Episode completed: {len(df_random)} steps")
print(f"Final portfolio value: ${df_random['portfolio_value'].iloc[-1]:,.2f}")
print(f"Final return: {(df_random['portfolio_value'].iloc[-1] / env_params.initial_balance - 1) * 100:.2f}%")
print(f"Final position: {df_random['position'].iloc[-1]:.4f}")
df_random.head()


# ## Option 2: Run with Trained Agent

# In[ ]:


# Load trained agent (adjust path to your latest model)
model_path = "models/PPO_sharpe_20251027_113016/best_model.zip"

try:
    # Load the model
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
        print("Loaded normalization stats")
    except:
        print("Warning: Could not load normalization stats")

    print(f"Loaded model from {model_path}")
    agent_available = True
except Exception as e:
    print(f"Could not load agent: {e}")
    print("Using random actions instead")
    agent_available = False


# In[ ]:


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

    return pd.DataFrame(data)

if agent_available:
    df_agent = run_agent_episode(test_vec_env, model)
    print(f"Episode completed: {len(df_agent)} steps")
    print(f"Final portfolio value: ${df_agent['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Final return: {(df_agent['portfolio_value'].iloc[-1] / env_params.initial_balance - 1) * 100:.2f}%")
    print(f"Final position: {df_agent['position'].iloc[-1]:.4f}")
    df_agent.head()
else:
    df_agent = df_random  # Use random data for plotting


# ## Visualize Episode

# In[ ]:


# Choose which data to plot
df_plot = df_agent if agent_available else df_random

fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

# 1. Portfolio Value
axes[0].plot(df_plot['step'], df_plot['portfolio_value'], label='Portfolio Value', linewidth=2)
axes[0].axhline(y=env_params.initial_balance, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
axes[0].set_ylabel('Portfolio Value ($)')
axes[0].set_title('Portfolio Value Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Price
axes[1].plot(df_plot['step'], df_plot['price'], label='Price', color='orange', linewidth=1.5)
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Asset Price')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Actions
action_colors = {0: 'gray', 1: 'green', 2: 'red'}
action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
for action_val in [0, 1, 2]:
    mask = df_plot['action'] == action_val
    axes[2].scatter(df_plot[mask]['step'], df_plot[mask]['action'], 
                   c=action_colors[action_val], label=action_labels[action_val], alpha=0.6, s=20)
axes[2].set_ylabel('Action')
axes[2].set_title('Actions Taken')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['Hold', 'Buy', 'Sell'])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 4. Position
axes[3].plot(df_plot['step'], df_plot['position'], label='Position Size', color='purple', linewidth=1.5)
axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[3].set_ylabel('Position Size')
axes[3].set_title('Position Size Over Time')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# 5. Rewards
axes[4].plot(df_plot['step'], df_plot['reward'], label='Reward', color='blue', alpha=0.7, linewidth=1)
axes[4].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[4].set_ylabel('Reward')
axes[4].set_xlabel('Step')
axes[4].set_title('Rewards')
axes[4].legend()
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ## Analysis: Check for Issues

# In[ ]:


print("=" * 60)
print("EPISODE SUMMARY")
print("=" * 60)
print(f"Total steps: {len(df_plot)}")
print(f"Initial balance: ${env_params.initial_balance:,.2f}")
print(f"Final portfolio value: ${df_plot['portfolio_value'].iloc[-1]:,.2f}")
print(f"Final return: {(df_plot['portfolio_value'].iloc[-1] / env_params.initial_balance - 1) * 100:.2f}%")
print()

print("Action Distribution:")
print(df_plot['action'].value_counts().sort_index())
print()

print("Reward Statistics:")
print(f"  Mean: {df_plot['reward'].mean():.4f}")
print(f"  Std: {df_plot['reward'].std():.4f}")
print(f"  Min: {df_plot['reward'].min():.4f}")
print(f"  Max: {df_plot['reward'].max():.4f}")
print()

print("Position Statistics:")
print(f"  Final position: {df_plot['position'].iloc[-1]:.4f}")
print(f"  Max position: {df_plot['position'].max():.4f}")
print(f"  Min position: {df_plot['position'].min():.4f}")
print(f"  % of time in position: {(df_plot['position'] != 0).sum() / len(df_plot) * 100:.1f}%")
print()

# Check if position is open at end
if abs(df_plot['position'].iloc[-1]) > 0.01:
    print("⚠️  WARNING: Position is still OPEN at end of episode!")
    print(f"   Open position size: {df_plot['position'].iloc[-1]:.4f}")
    print(f"   This unrealized P&L is NOT reflected in the return calculation!")
    print()

    # Calculate what the return would be if we closed the position
    unwrapped_env = test_env
    current_price = df_plot['price'].iloc[-1]
    position_size = df_plot['position'].iloc[-1]

    if hasattr(unwrapped_env, 'position') and unwrapped_env.position.entry_price > 0:
        entry_price = unwrapped_env.position.entry_price
        unrealized_pnl = position_size * (current_price - entry_price)
        print(f"   Entry price: ${entry_price:.2f}")
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Unrealized P&L: ${unrealized_pnl:.2f}")
else:
    print("✓ Position is FLAT at end of episode (good!)")
    print()


# ## Deep Dive: Balance vs Portfolio Value

# In[ ]:


# Calculate what balance + position value should equal
df_plot['position_value'] = abs(df_plot['position']) * df_plot['price']
df_plot['balance_plus_position'] = df_plot['balance'] + df_plot['position_value']

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Balance
axes[0].plot(df_plot['step'], df_plot['balance'], label='Balance (Cash)', linewidth=2)
axes[0].set_ylabel('Balance ($)')
axes[0].set_title('Cash Balance Over Time')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Position Value
axes[1].plot(df_plot['step'], df_plot['position_value'], label='Position Value', color='orange', linewidth=2)
axes[1].set_ylabel('Position Value ($)')
axes[1].set_title('Position Value Over Time')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Comparison
axes[2].plot(df_plot['step'], df_plot['portfolio_value'], label='Portfolio Value (from env)', linewidth=2)
axes[2].plot(df_plot['step'], df_plot['balance_plus_position'], 
            label='Balance + Position Value', linestyle='--', linewidth=2, alpha=0.7)
axes[2].set_ylabel('Value ($)')
axes[2].set_xlabel('Step')
axes[2].set_title('Portfolio Value: Environment vs Manual Calculation')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Check if they match
diff = (df_plot['portfolio_value'] - df_plot['balance_plus_position']).abs().max()
print(f"Max difference between env.portfolio_value and balance+position: ${diff:.2f}")
if diff > 1.0:
    print("⚠️  Large discrepancy detected! This suggests an accounting bug.")
else:
    print("✓ Values match (within $1 tolerance)")

