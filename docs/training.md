# Training & Checkpoints

This document covers the training workflow and checkpoint management system.

---

## Training Workflow

```
1. Load Data       → Historical market data (Parquet)
2. Create Env      → Gym-compatible trading environment
3. Initialize Agent → PPO, A2C, DQN, or SAC
4. Train           → Rollouts, optimization, evaluation
5. Log Metrics     → MLflow tracking
6. Save Checkpoint → Model + config + normalizer
```

---

## Running Training

Use the canonical training and evaluation commands from [docs/commands.md](commands.md).

---

## Checkpoint System

### What's Saved

Each checkpoint includes:

| File | Contents |
|------|----------|
| `model.zip` | Trained model weights |
| `vecnormalize.pkl` | Observation normalizer state |
| `metadata.json` | Training configuration |

### Self-Contained Checkpoints

A key design feature: **checkpoints embed their complete training configuration**.

This enables:
- Deployment without MLflow server
- Feature validation before inference
- Exact reproduction of training setup

### Checkpoint Structure

```
checkpoints/
└── PPO_returns_20251105_221747/
    ├── best_model/
    │   ├── model.zip
    │   ├── vecnormalize.pkl
    │   └── metadata.json
    └── checkpoint_50000/
        ├── model.zip
        └── vecnormalize.pkl
```

---

## Loading Checkpoints

### Discover Available Models

```python
from pathlib import Path
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

models = CheckpointManager.discover_all_checkpoints(Path("checkpoints"))
for model in models:
    print(f"{model['model_type']}: {model['checkpoint_dir'].name}")
```

### Load for Inference

```python
checkpoint_path = Path("checkpoints/PPO_returns_*/best_model/best_model.zip")
config = CheckpointManager.get_training_config(checkpoint_path)

print(f"Features: {config['observation']['input_features']}")
print(f"Reward: {config['env']['environment_params']['reward_type']}")
```

---

## Training Callbacks

Built-in callbacks for monitoring:

| Callback | Purpose |
|----------|---------|
| `EvalCallback` | Periodic evaluation on held-out data |
| `CheckpointCallback` | Save model at intervals |
| `MLflowCallback` | Log metrics to MLflow |
| `TradingMetricsCallback` | Log trading-specific metrics |

---

## Evaluation Metrics

### RL Metrics

- Episode reward (mean, std)
- Episode length
- Policy loss, value loss

### Trading Metrics

- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Trade count

---

## Monitoring Training

### MLflow UI

Use the `MLflow UI` row in [docs/commands.md](commands.md).

### TensorBoard

TensorBoard usage is optional and environment-specific; prefer the canonical observability command surfaces in [docs/commands.md](commands.md).

---

## Best Practices

1. **Start small**: Use the `Smoke test` row in [docs/commands.md](commands.md) to verify setup
2. **Track everything**: MLflow logs all experiments automatically
3. **Compare algorithms**: Try PPO, A2C, DQN to find best fit
4. **Use validation**: Always evaluate on held-out data
5. **Save frequently**: Checkpoints enable resuming and analysis
