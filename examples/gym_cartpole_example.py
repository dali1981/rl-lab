#!/usr/bin/env python
"""
Showcase: RL Trainer plugged into a standard Gym environment (CartPole).

Demonstrates that the extracted RL module is reusable beyond trading:
- No trading domain, no market data, no Binance, no MLflow
- Just Trainer + EnvWrapperBuilder + CallbackFactory + a gym env
"""

from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO

from rl_trading_lab.agents.sb3_agents import Trainer
from rl_trading_lab.agents.env_wrapper import EnvWrapperBuilder
from rl_trading_lab.agents.callback_factory import CallbackFactory


def main():
    save_path = Path("outputs/cartpole_demo")

    # --- 1. Build environments using EnvWrapperBuilder ---
    builder = EnvWrapperBuilder(
        vec_normalize_enabled=True,
        norm_obs=True,
        norm_reward=True,
    )

    train_env = builder.build(gym.make("CartPole-v1"), is_eval=False, gamma=0.99)
    eval_env = builder.build(gym.make("CartPole-v1"), is_eval=True, gamma=0.99)

    # --- 2. Create callbacks using CallbackFactory ---
    cb_factory = CallbackFactory()
    callbacks = cb_factory.create_all(
        eval_env=eval_env,
        save_path=save_path,
        eval_freq=2_000,
        n_eval_episodes=5,
        save_freq=5_000,
    )

    # --- 3. Create Trainer (no config objects, no trading domain) ---
    trainer = Trainer(
        algo_class=PPO,
        env=train_env,
        eval_env=eval_env,
        hyperparams={
            "learning_rate": 3e-4,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
        },
        save_path=save_path,
        device="cpu",
        verbose=1,
    )

    # --- 4. Train ---
    print("\n=== Training PPO on CartPole-v1 ===\n")
    trainer.train(
        total_timesteps=20_000,
        callbacks=callbacks,
        progress_bar=True,
    )

    # --- 5. Evaluate ---
    print("\n=== Evaluation ===\n")
    metrics = trainer.evaluate(env=eval_env, n_episodes=20, deterministic=True)
    print(f"  Mean reward:  {metrics['mean_reward']:.1f} +/- {metrics['std_reward']:.1f}")
    print(f"  Mean length:  {metrics['mean_episode_length']:.0f}")
    print(f"  Min/Max:      {metrics['min_reward']:.0f} / {metrics['max_reward']:.0f}")

    # --- 6. Quick inference demo ---
    print("\n=== Inference Demo (5 episodes) ===\n")
    raw_env = gym.make("CartPole-v1")
    for ep in range(5):
        obs, _ = raw_env.reset()
        total_reward = 0
        done = False
        while not done:
            # Trainer.predict works on raw observations
            action, _ = trainer.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = raw_env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        print(f"  Episode {ep + 1}: reward={total_reward:.0f}")

    raw_env.close()
    print(f"\nModel saved to: {save_path}/final_model.zip")


if __name__ == "__main__":
    main()
