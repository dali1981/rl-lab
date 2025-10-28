#!/usr/bin/env python
"""Test that model loading works as expected in the notebook"""

from pathlib import Path
from stable_baselines3 import PPO
from rl_trading_lab.models import TransformerActorCriticPolicy

# This mimics the notebook cell 10
checkpoint_dir = Path("../checkpoints")

# Find best models
best_models = list(checkpoint_dir.glob("*/best_model"))
if best_models:
    best_models.sort(key=lambda p: p.stat().st_mtime)
    model_path = best_models[-1] / "best_model.zip"
    print(f"Found model: {model_path}")

    # Load it
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded successfully!")
        print(f"  Policy type: {type(model.policy).__name__}")
        print(f"  Device: {model.device}")
        print("\n✅ Notebook will work!")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        print("\n❌ Notebook will NOT work")
else:
    print("No models found")
