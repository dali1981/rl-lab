# CheckpointManager - Robust Model Saving & Loading

## Overview

**CheckpointManager** is a unified system for saving and loading RL model checkpoints with automatic metadata tracking, VecNormalize stats management, and custom policy support.

## Problem Solved

### Before CheckpointManager ❌

**Saving:**
- `best_model.zip` saved WITHOUT VecNormalize stats → unusable
- No metadata about policy type, versions, config
- Periodic checkpoints OK but loading was fragile

**Loading:**
- Custom policies (Transformer) required manual imports → "persistent_load" errors
- VecNormalize stats path mismatches
- No version compatibility checking
- Confusing error messages

### After CheckpointManager ✅

**Saving:**
- **Automatic metadata** with every checkpoint (policy, versions, config)
- **VecNormalize stats** always saved with best_model
- **Future-proof** with version tracking

**Loading:**
- **One-line loading**: `manager.load_best_model(env)`
- **Auto-imports** custom policies (Transformer, extractors)
- **Robust path matching** for VecNormalize
- **Backward compatible** with old checkpoints
- **Clear error messages** with recovery suggestions

## Files Created

1. **`src/rl_trading_lab/utils/checkpoint_manager.py`** - Core CheckpointManager class
2. **`src/rl_trading_lab/utils/custom_callbacks.py`** - Enhanced callbacks
   - `CheckpointManagerCallback` - Saves metadata with periodic checkpoints
   - `BestModelCallback` - Fixes best_model VecNormalize bug
3. **`test_checkpoint_manager.py`** - Validation tests

## Files Modified

1. **`src/rl_trading_lab/agents/sb3_agents.py`**
   - Uses new callbacks for training
   - `load()` method uses CheckpointManager
2. **`notebooks/debug_episode.ipynb`**
   - Simplified loading with CheckpointManager
3. **`src/rl_trading_lab/utils/__init__.py`**
   - Exports CheckpointManager

## Usage

### Training (Automatic)

Checkpoints are now saved with metadata automatically:

```python
# Train as usual - checkpoints now include metadata
uv run python experiments/train.py agent=ppo_transformer

# Creates:
# checkpoints/PPO_transformer_xxx/
#   best_model/
#     best_model.zip                    # Model weights
#     best_model.metadata.json          # NEW: Metadata
#     vecnormalize.pkl                  # NEW: VecNormalize stats (was missing!)
#   checkpoints/
#     rl_model_10000_steps.zip
#     rl_model_10000_steps.metadata.json   # NEW
#     rl_model_vecnormalize_10000_steps.pkl
```

### Loading in Notebooks

**Simple - Load Best Model:**
```python
from pathlib import Path
from rl_trading_lab.utils import CheckpointManager

# Create test environment
test_env = TradingEnv(...)

# Load best model (ONE LINE!)
manager = CheckpointManager(Path("checkpoints/PPO_transformer_xxx"))
model, vec_env = manager.load_best_model(test_env)

# Make predictions
obs = vec_env.reset()
action, _ = model.predict(obs)
```

**Find Latest Automatically:**
```python
from rl_trading_lab.utils import CheckpointManager

# Find latest training run
checkpoint_dir = Path("checkpoints")
runs = sorted(checkpoint_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
latest_run = runs[0]

# Load it
manager = CheckpointManager(latest_run)
model, vec_env = manager.load_best_model(test_env, verbose=1)
```

**Load Specific Checkpoint:**
```python
# Load a specific periodic checkpoint
checkpoint_path = Path("checkpoints/PPO_xxx/checkpoints/rl_model_50000_steps.zip")
model, vec_env = manager.load_checkpoint(checkpoint_path, test_env)
```

**List All Checkpoints:**
```python
manager = CheckpointManager(Path("checkpoints/PPO_xxx"))
checkpoints = manager.list_checkpoints()

for ckpt in checkpoints:
    print(f"{ckpt['path'].name} - Type: {ckpt['type']}")
    if ckpt['metadata']:
        print(f"  Policy: {ckpt['metadata']['policy_class']['name']}")
        print(f"  Created: {ckpt['metadata']['timestamp']}")
```

### Loading in Scripts

```python
from rl_trading_lab.utils import CheckpointManager

# Simple loading
manager = CheckpointManager()
model, vec_env = manager.load_checkpoint(
    path=Path("checkpoints/PPO_xxx/best_model/best_model.zip"),
    env=test_env,
    verbose=1
)

# Model and vec_env are ready to use
obs = vec_env.reset()
action, _ = model.predict(obs, deterministic=True)
```

## Metadata Structure

Each checkpoint now includes a `.metadata.json` file:

```json
{
  "version": "1.0",
  "timestamp": "2025-10-27T21:30:00",
  "algorithm": "PPO",
  "policy_class": {
    "module": "rl_trading_lab.models.transformer_policy",
    "name": "TransformerActorCriticPolicy"
  },
  "feature_extractor": {
    "module": "rl_trading_lab.models.transformer_policy",
    "name": "TransformerFeatureExtractor"
  },
  "observation_space": {
    "shape": [84],
    "dtype": "float32"
  },
  "action_space": {
    "type": "Discrete",
    "n": 3
  },
  "versions": {
    "stable_baselines3": "2.7.0",
    "torch": "2.9.0",
    "numpy": "2.3.4",
    "python": "3.12.10"
  },
  "custom": {
    "agent_config": "PPO"
  }
}
```

## Key Features

### 1. Automatic Custom Policy Loading

**Transformer policies load automatically:**
```python
# No manual imports needed!
model, vec_env = manager.load_best_model(env)
# ✓ TransformerActorCriticPolicy imported automatically
# ✓ TransformerFeatureExtractor imported automatically
```

### 2. VecNormalize Stats Management

**Always loads normalization stats:**
```python
# Tries multiple locations:
# 1. best_model/vecnormalize.pkl
# 2. checkpoints/rl_model_vecnormalize_{steps}_steps.pkl
# 3. Same directory as model with _vecnormalize suffix
```

### 3. Backward Compatibility

**Works with old checkpoints (no metadata):**
```python
# Old checkpoint without metadata still loads
manager.load_checkpoint(old_checkpoint_path, env)
# ⚠️  Warning: No metadata found, attempting to load without it
# ✓ Still works! (with fallback to common custom classes)
```

### 4. Version Tracking

**Know what created each checkpoint:**
- SB3 version
- PyTorch version
- Python version
- Policy class and module
- Timestamp

### 5. Robust Error Handling

**Clear error messages:**
```
❌ Failed to load model: TransformerActorCriticPolicy not found

💡 Solution:
  - Model uses custom policy: rl_trading_lab.models.TransformerActorCriticPolicy
  - This is now auto-imported, but check that the module exists
  - Try: from rl_trading_lab.models import TransformerActorCriticPolicy
```

## Testing

**Run the test suite:**
```bash
uv run python test_checkpoint_manager.py
```

**Tests:**
- ✅ Load old checkpoints (no metadata)
- ✅ Load best_model
- ✅ Load periodic checkpoints
- ✅ Make predictions
- ✅ List all checkpoints
- ✅ Custom policies (Transformer)
- ✅ VecNormalize stats loading

## Migration from Old Checkpoints

**Old checkpoints work immediately** - no migration needed!

CheckpointManager automatically:
- Detects missing metadata
- Imports common custom classes
- Finds VecNormalize stats in multiple locations
- Emits warnings about potential issues

**New checkpoints will have full metadata** starting from your next training run.

## Troubleshooting

### Issue: "No metadata found"
**Cause**: Old checkpoint (pre-CheckpointManager)
**Solution**: Works automatically with fallback, just a warning

### Issue: "VecNormalize stats not loaded"
**Cause**: best_model from old training run
**Solution**:
- Use a periodic checkpoint instead: `rl_model_10000_steps.zip`
- Or re-train to get fixed best_model with stats

### Issue: "Custom policy not found"
**Cause**: Policy class was renamed/moved
**Solution**: Metadata has the old import path, update it manually or re-train

### Issue: Model from different SB3 version
**Cause**: Version incompatibility
**Solution**: Check metadata versions, consider re-training

## Best Practices

1. **Use `load_best_model()` in notebooks** - simplest API
2. **Check verbosity** - Use `verbose=1` to see what's loaded
3. **List checkpoints first** - See what's available before loading
4. **Keep metadata** - Don't delete `.metadata.json` files
5. **Use latest checkpoint** - Sort by modification time for most recent

## Example: Full Workflow

```python
from pathlib import Path
from rl_trading_lab.utils import CheckpointManager
from rl_trading_lab.environment.trading_env import TradingEnv

# 1. Create environment
env = TradingEnv(df=test_data, ...)

# 2. Find latest training run
runs = sorted(Path("checkpoints").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
latest = runs[0]
print(f"Loading from: {latest.name}")

# 3. Load with CheckpointManager
manager = CheckpointManager(latest)
model, vec_env = manager.load_best_model(env, verbose=1)

# 4. Run episode
obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

print(f"Episode return: {info['total_return']}")
```

## Summary

**CheckpointManager fixes all checkpoint loading issues:**

✅ **Saves VecNormalize with best_model** (was missing!)
✅ **Auto-imports custom policies** (no more persistent_load errors)
✅ **Tracks metadata** (policy, versions, config)
✅ **One-line loading** (simple API)
✅ **Backward compatible** (works with old checkpoints)
✅ **Robust error handling** (clear messages)
✅ **Future-proof** (version tracking)

**Your models now load reliably every time!** 🎉

---

**Status**: ✅ Fully implemented and tested
**Branch**: `model/transformer`
**Last Updated**: October 27, 2025
