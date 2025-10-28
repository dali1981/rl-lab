#!/usr/bin/env python
"""
Diagnose saved model issues.

Usage:
    uv run python diagnose_model.py path/to/model.zip
"""

import sys
import zipfile
from pathlib import Path
import pickle

def diagnose_model(model_path):
    """Diagnose what's in a saved SB3 model"""
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return

    print(f"🔍 Diagnosing: {model_path}")
    print("=" * 60)

    # Check if it's a zip file
    if not zipfile.is_zipfile(model_path):
        print("❌ Not a valid zip file")
        return

    print("✓ Valid zip file")

    # List contents
    with zipfile.ZipFile(model_path, 'r') as zf:
        print(f"\n📦 Contents ({len(zf.namelist())} files):")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"  - {name} ({info.file_size} bytes)")

        # Try to read data
        print("\n📄 Reading data...")
        try:
            with zf.open('data') as f:
                data = pickle.load(f)

            print(f"✓ Successfully loaded data pickle")
            print(f"\n🔑 Keys in data:")
            for key in data.keys():
                print(f"  - {key}: {type(data[key])}")

            # Check policy class
            if 'policy_class' in data:
                policy_class = data['policy_class']
                print(f"\n🎯 Policy class:")
                print(f"  Module: {policy_class.__module__}")
                print(f"  Name: {policy_class.__name__}")

                # Check if it's a custom policy
                if 'rl_trading_lab' in policy_class.__module__:
                    print(f"  ⚠️  Custom policy detected!")
                    print(f"  💡 Make sure to import before loading:")
                    print(f"     from rl_trading_lab.models import {policy_class.__name__}")

            # Check observation space
            if 'observation_space' in data:
                obs_space = data['observation_space']
                print(f"\n👁️  Observation space:")
                print(f"  Type: {type(obs_space).__name__}")
                print(f"  Shape: {obs_space.shape if hasattr(obs_space, 'shape') else 'N/A'}")

            # Check action space
            if 'action_space' in data:
                action_space = data['action_space']
                print(f"\n🎮 Action space:")
                print(f"  Type: {type(action_space).__name__}")
                if hasattr(action_space, 'n'):
                    print(f"  N actions: {action_space.n}")

            # Check if verbose flag exists
            if 'verbose' in data:
                print(f"\n🔊 Verbose: {data['verbose']}")

        except Exception as e:
            print(f"❌ Could not load data: {e}")
            print(f"   Error type: {type(e).__name__}")

            if "persistent_load" in str(e):
                print("\n💡 This usually means:")
                print("   - Custom policy/feature extractor not imported")
                print("   - Version mismatch in PyTorch/SB3")
                print("\n🔧 Try:")
                print("   1. Import custom classes before loading")
                print("   2. Load with custom_objects parameter")
                print("   3. Check PyTorch version compatibility")

    print("\n" + "=" * 60)

    # Check for normalization stats
    norm_paths = [
        model_path.parent / 'vecnormalize.pkl',
        str(model_path).replace('.zip', '_vecnormalize.pkl'),
    ]

    print("\n🔍 Checking for normalization stats:")
    for norm_path in norm_paths:
        if Path(norm_path).exists():
            print(f"  ✓ Found: {norm_path}")
        else:
            print(f"  ✗ Not found: {norm_path}")

    print("\n" + "=" * 60)
    print("✅ Diagnosis complete")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python diagnose_model.py path/to/model.zip")
        print("\nSearching for models in checkpoints/...")

        checkpoint_dir = Path("../checkpoints")
        if checkpoint_dir.exists():
            models = list(checkpoint_dir.glob("**/*.zip"))
            if models:
                print(f"\nFound {len(models)} models:")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
                print("\nRun with a model path to diagnose it.")
            else:
                print("No models found in checkpoints/")
        else:
            print("checkpoints/ directory not found")

        sys.exit(1)

    model_path = sys.argv[1]
    diagnose_model(model_path)
