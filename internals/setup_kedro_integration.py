#!/usr/bin/env python
"""
Setup script for Kedro integration.
This helps connect your Kedro data pipeline with the RL training lab.
"""

import sys
from pathlib import Path
import yaml
import shutil


def check_kedro_project():
    """Check if Kedro project exists"""
    kedro_path = Path("../kedro-crypto-ind")
    if kedro_path.exists():
        print(f"✓ Found Kedro project at: {kedro_path.resolve()}")
        return kedro_path
    else:
        print(f"✗ Kedro project not found at: {kedro_path}")
        print("  You can still use the example data in ../tools/examples/")
        return None


def check_data_files():
    """Check available data files"""
    print("\n📊 Available Data Files:")

    # Check example data
    example_data = Path("../tools/examples/btcusdt_fractional_indicators.parquet")
    if example_data.exists():
        print(f"✓ Example data: {example_data}")
        size_mb = example_data.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ Example data not found: {example_data}")

    # Check Kedro output
    kedro_data = Path("../kedro-crypto-ind/data/08_reporting/ml_ready_features.parquet")
    if kedro_data.exists():
        print(f"✓ Kedro output: {kedro_data}")
        size_mb = kedro_data.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ Kedro output not found: {kedro_data}")
        print("  Run: cd ../kedro-crypto-ind && kedro run")

    return example_data.exists() or kedro_data.exists()


def update_config_data_path():
    """Update config with correct data path"""
    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return

    # Determine which data path to use
    example_data = Path("../tools/examples/btcusdt_fractional_indicators.parquet")
    kedro_data = Path("../kedro-crypto-ind/data/08_reporting/ml_ready_features.parquet")

    if kedro_data.exists():
        data_path = str(kedro_data)
        print(f"\n✓ Using Kedro pipeline output: {data_path}")
    elif example_data.exists():
        data_path = str(example_data)
        print(f"\n✓ Using example data: {data_path}")
    else:
        print("\n✗ No data files found!")
        return

    # Update config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['data']['train_data_path'] = data_path

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Updated config with data path: {data_path}")


def create_kedro_runner():
    """Create a script to run Kedro pipeline and train RL"""
    script_content = '''#!/usr/bin/env python
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
'''

    script_path = Path("run_pipeline.py")
    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)  # Make executable
    print(f"\n✓ Created pipeline runner: {script_path}")
    print("  Usage: python run_pipeline.py [training args]")
    print("  Example: python run_pipeline.py agent=ppo env.reward_type=sharpe")


def setup_mlflow():
    """Initialize MLflow for experiment tracking"""
    print("\n🔬 Setting up MLflow...")

    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(exist_ok=True)

    print("✓ MLflow directory created")
    print("\nTo start MLflow UI:")
    print("  mlflow ui --port 5000")
    print("  Then open: http://localhost:5000")


def main():
    """Main setup function"""
    print("🚀 RL Trading Lab - Kedro Integration Setup\n")
    print("=" * 50)

    # Check Kedro project
    kedro_path = check_kedro_project()

    # Check data files
    has_data = check_data_files()

    if not has_data:
        print("\n⚠️  No data files found!")
        print("\nTo create data:")
        print("  1. Run Kedro pipeline: cd ../kedro-crypto-ind && kedro run")
        print("  2. Or create example data: cd ../tools && python examples/06_complete_pipeline.py")
        sys.exit(1)

    # Update config
    update_config_data_path()

    # Create runner script
    create_kedro_runner()

    # Setup MLflow
    setup_mlflow()

    print("\n" + "=" * 50)
    print("✅ Setup Complete!\n")
    print("Next steps:")
    print("  1. Install dependencies: uv sync")
    print("  2. Train your first agent: uv run python experiments/train.py")
    print("  3. Or run full pipeline: uv run python run_pipeline.py")
    print("\nHappy trading! 🚀📈")


if __name__ == "__main__":
    main()