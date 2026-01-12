#!/usr/bin/env python3
"""
Simple test script to understand log_interval behavior with MLflow.

This script runs quick experiments with different log_interval values
to see when training metrics vs eval metrics are logged.
"""

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from src.rl_trading_lab.utils.mlflow_logger import MLflowOutputFormat


def test_log_interval(log_interval: int, total_timesteps: int = 25000):
    """Test a specific log_interval value."""

    print(f"\n{'='*60}")
    print(f"Testing log_interval={log_interval}")
    print(f"{'='*60}")

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(f"test-log-interval")

    with mlflow.start_run(run_name=f"log_interval_{log_interval}"):
        # Create environments
        env = make_vec_env("CartPole-v1", n_envs=1)
        eval_env = make_vec_env("CartPole-v1", n_envs=1)

        # Create model with n_steps=2048 (like your PPO config)
        model = PPO("MlpPolicy", env, n_steps=2048, verbose=0)

        # Setup logger with MLflow
        logger = configure(None, ["stdout"])
        logger.output_formats.append(MLflowOutputFormat())
        model.set_logger(logger)

        # Eval every 5000 steps (like your config)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=5000,
            n_eval_episodes=3,
            verbose=0
        )

        print(f"n_steps: {model.n_steps}")
        print(f"Expected training log every: {log_interval * model.n_steps} steps")
        print(f"Eval callback runs every: 5000 steps")
        print(f"\nTraining for {total_timesteps} timesteps...")

        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=log_interval,
            progress_bar=True
        )

        run_id = mlflow.active_run().info.run_id

        # Analyze what was logged
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metrics = run.data.metrics

        print(f"\n{'='*60}")
        print(f"Results for log_interval={log_interval}")
        print(f"{'='*60}")

        # Check rollout metrics
        if 'rollout/ep_rew_mean' in metrics:
            history = client.get_metric_history(run_id, 'rollout/ep_rew_mean')
            steps = [h.step for h in history]
            if len(steps) > 1:
                avg_freq = (steps[-1] - steps[0]) / (len(steps) - 1)
                print(f"\n✓ rollout/ep_rew_mean:")
                print(f"  - Logged {len(steps)} times")
                print(f"  - Steps: {steps}")
                print(f"  - Average frequency: {avg_freq:.0f} steps")
            else:
                print(f"\n✗ rollout/ep_rew_mean: Only logged {len(steps)} time(s)")
        else:
            print(f"\n✗ rollout/ep_rew_mean: NOT logged")

        # Check eval metrics
        if 'eval/mean_reward' in metrics:
            history = client.get_metric_history(run_id, 'eval/mean_reward')
            steps = [h.step for h in history]
            if len(steps) > 1:
                avg_freq = (steps[-1] - steps[0]) / (len(steps) - 1)
                print(f"\n✓ eval/mean_reward:")
                print(f"  - Logged {len(steps)} times")
                print(f"  - Steps: {steps}")
                print(f"  - Average frequency: {avg_freq:.0f} steps")
            else:
                print(f"\n✓ eval/mean_reward: Logged {len(steps)} time(s)")
        else:
            print(f"\n✗ eval/mean_reward: NOT logged")

        # Check train metrics
        if 'train/policy_gradient_loss' in metrics:
            history = client.get_metric_history(run_id, 'train/policy_gradient_loss')
            steps = [h.step for h in history]
            if len(steps) > 1:
                avg_freq = (steps[-1] - steps[0]) / (len(steps) - 1)
                print(f"\n✓ train/policy_gradient_loss:")
                print(f"  - Logged {len(steps)} times")
                print(f"  - Steps: {steps}")
                print(f"  - Average frequency: {avg_freq:.0f} steps")

        print(f"\n{'='*60}\n")

        return run_id


def main():
    """Run tests with different log_interval values."""

    print("\n" + "="*60)
    print("MLflow Logging Frequency Test")
    print("="*60)
    print("\nThis test will help understand when training metrics are logged")
    print("compared to eval metrics with different log_interval values.\n")

    # Test different log_interval values
    total_timesteps = 25000

    print("\nRunning 3 experiments...")
    print(f"Total timesteps per experiment: {total_timesteps}")
    print(f"n_steps (rollout size): 2048")
    print(f"eval_freq: 5000")

    run_ids = {}

    # Test 1: Current setting
    run_ids['log_interval_10'] = test_log_interval(10, total_timesteps)

    # Test 2: Proposed fix
    run_ids['log_interval_1'] = test_log_interval(1, total_timesteps)

    # Test 3: Middle ground
    run_ids['log_interval_2'] = test_log_interval(2, total_timesteps)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nYou can view the results in MLflow UI:")
    print(f"  http://localhost:5001")
    print(f"\nRun IDs:")
    for name, run_id in run_ids.items():
        print(f"  {name}: {run_id}")

    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("""
For your use case with eval_freq=5000:
- log_interval=10 → Training metrics every ~20k steps (too infrequent!)
- log_interval=2  → Training metrics every ~4k steps (good balance)
- log_interval=1  → Training metrics every ~2k steps (most frequent)

Recommended: log_interval=2 for a good balance between
frequency and performance overhead.
    """)


if __name__ == "__main__":
    main()
