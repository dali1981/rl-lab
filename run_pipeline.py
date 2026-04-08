#!/usr/bin/env python
"""
Run Kedro data pipeline and then train RL agent.
This ensures fresh data for each training run.

Canonical training orchestration is delegated to `experiments/train.py`,
which routes through `TrainAgentUseCase`.
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
    """Train RL agent through the canonical training entrypoint."""
    print("Running canonical training entrypoint...")

    # Build command
    train_script = Path(__file__).parent / "experiments" / "train.py"
    cmd = [sys.executable, str(train_script)] + args

    # Run training
    result = subprocess.run(cmd)

    return result.returncode == 0


if __name__ == "__main__":
    help_requested = "--help" in sys.argv or "-h" in sys.argv

    # Run Kedro pipeline
    if "--skip-kedro" not in sys.argv and not help_requested:
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
