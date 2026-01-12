#!/usr/bin/env python3
"""
Quick test to verify MLflow logging works with simplified SB3 logger integration.
This tests that training and eval metrics are logged without the MLflowCallback.
"""

import mlflow
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from rl_trading_lab.utils.mlflow_logger import MLflowOutputFormat

def test_mlflow_logging():
    """Test that SB3 logger with MLflowOutputFormat logs to MLflow."""

    # Set MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("test-simplified-logging")

    with mlflow.start_run(run_name="quick-test"):
        print("✓ MLflow run started")

        # Create simple environment
        env = make_vec_env("CartPole-v1", n_envs=1)
        eval_env = make_vec_env("CartPole-v1", n_envs=1)

        # Create model
        model = PPO("MlpPolicy", env, verbose=0)
        print("✓ Model created")

        # Setup SB3 logger with MLflow integration
        logger = configure(None, ["stdout"])
        logger.output_formats.append(MLflowOutputFormat())
        model.set_logger(logger)
        print("✓ Logger configured with MLflowOutputFormat")

        # Create eval callback (will use the logger too)
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=500,
            n_eval_episodes=3,
            verbose=0
        )

        # Train for a short time
        print("\n🚀 Starting training (5000 steps)...")
        model.learn(total_timesteps=5000, callback=eval_callback, log_interval=1)

        print("\n✅ Training completed!")
        print(f"📊 MLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"🔗 View in UI: mlflow ui --port 5000")

        # Show what was logged
        client = mlflow.tracking.MlflowClient()
        run_id = mlflow.active_run().info.run_id
        metrics = client.get_run(run_id).data.metrics

        print(f"\n📈 Metrics logged to MLflow: {len(metrics)} total")
        print("Key metrics:")
        for key in sorted(metrics.keys())[:10]:
            print(f"  - {key}: {metrics[key]}")

        if len(metrics) > 10:
            print(f"  ... and {len(metrics) - 10} more")

if __name__ == "__main__":
    test_mlflow_logging()
