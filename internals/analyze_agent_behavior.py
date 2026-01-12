"""
Analyze why the DQN agent plateaus at 0.0186 mean reward.
Investigates action distribution, trading frequency, and policy conservatism.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rl_trading_lab.environment.trading_env import TradingEnv
from stable_baselines3 import DQN


def analyze_exploration_schedule():
    """Calculate where we are in the exploration schedule"""
    # Load DQN config
    with open("configs/agent/dqn.yaml", "r") as f:
        dqn_config = yaml.safe_load(f)

    exploration_fraction = dqn_config['hyperparameters']['exploration_fraction']
    total_timesteps = 290_922  # From the screenshot
    initial_eps = dqn_config['hyperparameters']['exploration_initial_eps']
    final_eps = dqn_config['hyperparameters']['exploration_final_eps']

    exploration_timesteps = exploration_fraction * total_timesteps
    current_epsilon = max(
        final_eps,
        initial_eps - (initial_eps - final_eps) * (total_timesteps / exploration_timesteps)
    )

    print("=" * 80)
    print("EXPLORATION ANALYSIS")
    print("=" * 80)
    print(f"Total timesteps trained: {total_timesteps:,}")
    print(f"Exploration fraction: {exploration_fraction} ({exploration_fraction*100:.1f}%)")
    print(f"Exploration ended at: {int(exploration_fraction * total_timesteps):,} steps")
    print(f"Initial epsilon: {initial_eps}")
    print(f"Final epsilon: {final_eps}")
    print(f"Current epsilon: {current_epsilon:.4f}")
    print(f"\nConclusion: Agent is only exploring {current_epsilon*100:.1f}% of the time!")
    print("            It's exploiting a learned policy 95% of the time.")
    print()


def test_agent_behavior(model_path: str, data_path: str, n_episodes: int = 10):
    """Run the trained agent and analyze its behavior"""

    # Analyze exploration first
    analyze_exploration_schedule()

    # Load configs manually
    with open("configs/env/default.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    # Load data
    print("=" * 80)
    print("LOADING MODEL AND DATA")
    print("=" * 80)
    df = pd.read_parquet(data_path)
    print(f"Data loaded: {len(df)} rows")

    # Create environment (disable randomization for consistent testing)
    env_params = env_config['environment_params']
    env = TradingEnv(
        df=df,
        lookback_window=env_params['lookback_window'],
        initial_balance=env_params['initial_balance'],
        commission_rate=env_params['commission_rate'],
        slippage_rate=env_params['slippage_rate'],
        reward_type=env_params['reward_type'],
        discrete_actions=env_params['discrete_actions'],
        max_position_pct=env_params['max_position_pct'],
        randomize_start=False,  # Start from beginning for analysis
        min_episode_length=env_params['min_episode_length'],
        hold_closes_position=env_params['hold_closes_position'],
        price_column=env_config['price_column'],
    )

    # Load model
    try:
        model = DQN.load(model_path, env=env)
        print(f"Model loaded from: {model_path}\n")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Analyzing environment behavior with random agent instead...\n")
        model = None

    # Run episodes and collect statistics
    print("=" * 80)
    print("RUNNING EPISODES AND ANALYZING BEHAVIOR")
    print("=" * 80)

    action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
    action_names = {0: "Hold", 1: "Buy", 2: "Sell"}
    episode_stats = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_actions = []
        episode_reward = 0
        steps = 0

        while not done:
            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            action_counts[int(action)] += 1
            episode_actions.append(int(action))

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        # Close any remaining positions
        env.close_all_positions()
        final_info = env._get_info()

        episode_stats.append({
            'episode': episode + 1,
            'steps': steps,
            'total_reward': episode_reward,
            'mean_reward': episode_reward / steps,
            'total_return': final_info['total_return'],
            'num_trades': final_info['num_trades'],
            'final_balance': final_info['portfolio_value'],
        })

        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Steps={steps}, Trades={final_info['num_trades']}, "
              f"Return={final_info['total_return']:.4f}, "
              f"Mean Reward={episode_reward/steps:.6f}")

    # Print statistics
    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)
    total_actions = sum(action_counts.values())
    for action, count in action_counts.items():
        pct = count / total_actions * 100
        print(f"{action_names[action]:>6}: {count:>6} ({pct:>6.2f}%)")

    print("\n" + "=" * 80)
    print("EPISODE STATISTICS")
    print("=" * 80)
    stats_df = pd.DataFrame(episode_stats)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Average trades per episode: {stats_df['num_trades'].mean():.1f}")
    print(f"Average mean reward: {stats_df['mean_reward'].mean():.6f}")
    print(f"Average return: {stats_df['total_return'].mean():.4f} ({stats_df['total_return'].mean()*100:.2f}%)")
    print(f"Max return: {stats_df['total_return'].max():.4f} ({stats_df['total_return'].max()*100:.2f}%)")
    print(f"Min return: {stats_df['total_return'].min():.4f} ({stats_df['total_return'].min()*100:.2f}%)")

    # Analyze conservatism
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    hold_pct = action_counts[0] / total_actions * 100
    trade_rate = stats_df['num_trades'].mean() / stats_df['steps'].mean()

    if hold_pct > 70:
        print(f"⚠️  PROBLEM: Agent is VERY CONSERVATIVE!")
        print(f"    - Holding {hold_pct:.1f}% of the time")
        print(f"    - Only making {stats_df['num_trades'].mean():.1f} trades per episode")
        print(f"    - Trading on only {trade_rate*100:.1f}% of steps")
    elif hold_pct > 50:
        print(f"⚠️  Agent is somewhat conservative")
        print(f"    - Holding {hold_pct:.1f}% of the time")
    else:
        print(f"✓  Agent is actively trading")
        print(f"    - Holding {hold_pct:.1f}% of the time")

    print("\nWHY THIS HAPPENS:")
    print("1. Risk Aversion: Agent learned that holding avoids losses")
    print("2. Exploration Decay: Only exploring 5% of the time (epsilon=0.05)")
    print("3. Local Optimum: Found a 'safe' strategy and stopped exploring better ones")
    print("4. Reward Structure: Small positive rewards for not losing > risky trades")


if __name__ == "__main__":
    # Update these paths to your model and data
    model_path = "checkpoints/best_model.zip"  # Update to your actual checkpoint
    data_path = "../tools/examples/btcusdt_fractional_indicators.parquet"

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Will analyze with random agent instead...\n")

    test_agent_behavior(model_path, data_path, n_episodes=10)
