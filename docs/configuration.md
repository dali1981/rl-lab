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

Use the canonical CLI contract in [docs/commands.md](commands.md).

This document focuses on parameter semantics; command invocation examples are centralized in the command matrix.

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
| `training.save_freq` | Checkpoint frequency | `10000` |

---

## Experiment Tracking

All experiments are automatically logged to **MLflow**.

Use the `MLflow UI` row in [docs/commands.md](commands.md) for the canonical launch command.

Tracked items:
- Hyperparameters
- Training metrics (reward, loss, etc.)
- Trading metrics (Sharpe, returns, drawdown)
- Model checkpoints

---

## Example Override Sets

These are override fragments to apply to canonical commands from [docs/commands.md](commands.md):

- High-frequency training:
  - `agent=ppo`
  - `training.total_timesteps=1000000`
  - `agent.hyperparameters.n_steps=512`
- Risk-adjusted optimization:
  - `env=sharpe`
  - `env.environment_params.commission_rate=0.002`
- Quick test run:
  - `training.total_timesteps=1000`
