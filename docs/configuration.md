# Configuration & Experiments

All hyperparameters and settings are managed through **Hydra** configuration files, enabling reproducible experiments without code changes.

---

## Configuration Structure

```
configs/
├── config.yaml           # Main entry point
├── agent/                # RL algorithm configs
│   ├── ppo.yaml
│   ├── a2c.yaml
│   ├── dqn.yaml
│   └── sac.yaml
├── env/                  # Environment configs
│   ├── returns.yaml      # Returns-based reward
│   ├── sharpe.yaml       # Sharpe ratio reward
│   └── sortino.yaml      # Sortino ratio reward
├── observation/          # Feature selection
└── training/             # Training parameters
```

---

## Running Experiments

### Basic Training

```bash
uv run python experiments/train.py
```

### Override Parameters

```bash
# Change algorithm
uv run python experiments/train.py agent=a2c

# Change reward function
uv run python experiments/train.py env=sharpe

# Override specific values
uv run python experiments/train.py \
  agent.hyperparameters.learning_rate=0.001 \
  training.total_timesteps=500000
```

---

## Key Configuration Options

### Agent Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent` | Algorithm selection | `ppo` |
| `agent.hyperparameters.learning_rate` | Learning rate | `0.0003` |
| `agent.hyperparameters.batch_size` | Batch size | `64` |
| `agent.hyperparameters.n_steps` | Steps per update | `2048` |

### Environment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `env.environment_params.reward_type` | Reward function | `returns` |
| `env.environment_params.commission_rate` | Trading cost | `0.001` |
| `env.environment_params.initial_balance` | Starting capital | `10000` |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `training.total_timesteps` | Total training steps | `100000` |
| `training.eval_freq` | Evaluation frequency | `10000` |
| `training.checkpoint_freq` | Checkpoint frequency | `10000` |

---

## Experiment Tracking

All experiments are automatically logged to **MLflow**:

```bash
# View experiments
mlflow ui --port 5000
```

Tracked items:
- Hyperparameters
- Training metrics (reward, loss, etc.)
- Trading metrics (Sharpe, returns, drawdown)
- Model checkpoints

---

## Example Configurations

### High-Frequency Training

```bash
uv run python experiments/train.py \
  agent=ppo \
  training.total_timesteps=1000000 \
  agent.hyperparameters.n_steps=512
```

### Risk-Adjusted Optimization

```bash
uv run python experiments/train.py \
  env=sharpe \
  env.environment_params.commission_rate=0.002
```

### Quick Test Run

```bash
uv run python experiments/train.py \
  training.total_timesteps=1000
```
