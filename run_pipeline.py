#!/usr/bin/env python
"""
Run Kedro data pipeline and then train RL agent.
This ensures fresh data for each training run.
"""

import subprocess
import sys
from pathlib import Path


def run_kedro_pipeline():
    """Run Kedro pipeline to prepare data"""
    print("🔄 Running Kedro data pipeline...")
    kedro_dir = Path("../kedro-crypto-ind")

    if not kedro_dir.exists():
        print(f"✗ Kedro project not found at {kedro_dir}")
        return False

    # Run Kedro pipeline
    result = subprocess.run(
        ["kedro", "run", "--pipeline", "feature_engineering"],
        cwd=kedro_dir,
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Kedro pipeline completed successfully")
        return True
    else:
        print(f"✗ Kedro pipeline failed: {result.stderr}")
        return False


def train_rl_agent(args):
    """Train RL agent with provided arguments"""
    print("🚀 Training RL agent...")

    # Build command
    cmd = ["python", "experiments/train.py"] + args

    # Run training
    result = subprocess.run(cmd)

    return result.returncode == 0


if __name__ == "__main__":
    # Run Kedro pipeline
    if "--skip-kedro" not in sys.argv:
        success = run_kedro_pipeline()
        if not success:
            print("⚠️  Kedro pipeline failed, using existing data")

    # Remove --skip-kedro from args
    args = [arg for arg in sys.argv[1:] if arg != "--skip-kedro"]

    # Train RL agent
    success = train_rl_agent(args)

    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed")
        sys.exit(1)
