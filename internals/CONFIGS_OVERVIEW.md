# Configuration Overview

Complete reference of all available configurations in the system.

## Quick Links

- 📖 [Full Configuration Guide](docs/CONFIGURATION_GUIDE.md)
- ⚡ [Quick Reference](CONFIG_QUICK_REFERENCE.md)
- 🔄 [Refactor Summary](REFACTOR_SUMMARY.md)

---

## Observation Configs (`configs/observation/`)

Controls **what the RL agent observes** (its observation space).

| Config | Features | Description | Command |
|--------|----------|-------------|---------|
| **default** | 6 | Price + Volume + Z-scored Indicators | `observation=default` |
| **minimal** | 2 | Just close + volume | `observation=minimal` |
| **price_only** | 4 | OHLC only | `observation=price_only` |
| **with_volume** | 5 | OHLCV | `observation=with_volume` |
| **indicators_only** | 4+ | Technical indicators only | `observation=indicators_only` |
| **with_returns** | 6 | Price + Volume + Returns | `observation=with_returns` (requires `feature_engineering=returns`) |

📄 **Details**: See `configs/observation/README.md`

---

## Feature Engineering Configs (`configs/feature_engineering/`)

Controls **runtime feature creation** (optional - prefer data pipeline).

| Config | Status | Creates | Use Case | Command |
|--------|--------|---------|----------|---------|
| **none** | Disabled | Nothing | Production (recommended) | `feature_engineering=none` |
| **simple_returns** | Enabled | returns_1 | Minimal testing | `feature_engineering=simple_returns` |
| **returns** | Enabled | returns_1/5/20, log_returns | Quick prototyping | `feature_engineering=returns` |
| **full** | Enabled | Returns + log_returns + rolling | Experimentation (slow) | `feature_engineering=full` |

📄 **Details**: See `configs/feature_engineering/README.md`

⚠️ **Recommendation**: Use `none` and create features in your data pipeline (Kedro).

---

## Environment Configs (`configs/env/`)

Controls **environment requirements and trading parameters**.

### Key Settings

```yaml
# Data requirements
price_column: "close"         # Column for trade execution
required_columns:             # Must exist in data
  - "close"
  - "timestamp"

# Trading parameters
environment_params:
  lookback_window: 20
  initial_balance: 10_000
  commission_rate: 0.00
  slippage_rate: 0.00
  reward_type: "returns"     # returns, sharpe, pnl
  discrete_actions: true
  randomize_start: true
  min_episode_length: 100
  hold_closes_position: true
```

### Override Examples

```bash
# Use different price column
uv run python experiments/train.py env.price_column=vwap

# Change reward type
uv run python experiments/train.py env.environment_params.reward_type=sharpe

# Adjust lookback window
uv run python experiments/train.py env.environment_params.lookback_window=50
```

---

## Agent Configs (`configs/agent/`)

Controls **RL algorithm and hyperparameters**.

| Config | Algorithm | Description | Command |
|--------|-----------|-------------|---------|
| **ppo** | PPO | Proximal Policy Optimization (default) | `agent=ppo` |
| **a2c** | A2C | Advantage Actor Critic | `agent=a2c` |
| **dqn** | DQN | Deep Q-Network | `agent=dqn` |

Each agent config includes recommended hyperparameters and training settings.

---

## Common Combinations

### 1. Production Setup (Recommended)

All features pre-computed in data pipeline:

```bash
uv run python experiments/train.py \
  observation=default \
  feature_engineering=none \
  agent=ppo
```

### 2. Quick Prototype

Minimal features with runtime returns:

```bash
uv run python experiments/train.py \
  observation=minimal \
  feature_engineering=simple_returns \
  agent=ppo
```

### 3. Indicator-Based Trading

```bash
uv run python experiments/train.py \
  observation=indicators_only \
  feature_engineering=none \
  agent=a2c
```

### 4. Full Experimentation

```bash
uv run python experiments/train.py \
  observation=with_returns \
  feature_engineering=returns \
  env.environment_params.reward_type=sharpe \
  agent=ppo
```

### 5. Custom Everything

```bash
uv run python experiments/train.py \
  'observation.input_features=["close","volume","ratio_sma_5_close_zscore"]' \
  feature_engineering=none \
  env.price_column=close \
  env.environment_params.reward_type=returns \
  agent=dqn
```

---

## Configuration Testing

### Test Config Loads

```bash
# Test observation configs
uv run python -c "
from rl_trading_lab.config import load_config
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = str(Path.cwd() / 'configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(
        config_name='config',
        overrides=['observation=minimal', 'feature_engineering=none']
    )
    config = load_config(cfg)
    print('✓ Config valid')
"
```

### Test Data Processing

```bash
# Test full pipeline
uv run python -c "
from rl_trading_lab.config import load_config
from rl_trading_lab.utils.data_processor import DataProcessor
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = str(Path.cwd() / 'configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config')
    config = load_config(cfg)

processor = DataProcessor(
    data_path=config.data.train_data_path,
    observation_config=config.observation,
    feature_engineering_config=config.feature_engineering,
)

train_df, val_df, test_df, obs_features = processor.process()
print(f'✓ Data processed: {len(train_df)} rows, {len(obs_features)} features')
"
```

---

## Config Status Summary

### ✅ Active Configs (Use These)

```
configs/
├── observation/              # What agent observes
│   ├── README.md
│   ├── default.yaml          ✓ Recommended
│   ├── minimal.yaml
│   ├── price_only.yaml
│   ├── with_volume.yaml
│   ├── indicators_only.yaml
│   └── with_returns.yaml
│
├── feature_engineering/      # Runtime feature creation
│   ├── README.md
│   ├── none.yaml             ✓ Recommended
│   ├── simple_returns.yaml
│   ├── returns.yaml
│   └── full.yaml
│
├── env/
│   └── default.yaml
│
├── agent/
│   ├── ppo.yaml              ✓ Default
│   ├── a2c.yaml
│   └── dqn.yaml
│
└── config.yaml               # Main config
```

### ⚠️ Deprecated Configs (Reference Only)

```
configs/
└── features/                 # DEPRECATED - see README_DEPRECATED.md
    ├── README_DEPRECATED.md
    ├── fractional_indicators.yaml
    └── minimal_example.yaml
```

---

## Architecture Recap

### Data Flow

```
┌────────────────────────────────────────────────┐
│ 1. Load Raw Data                               │
│    All columns from parquet                    │
└────────────────┬───────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────┐
│ 2. Feature Engineering (optional)              │
│    Create returns, log_returns, rolling stats  │
│    Controlled by: feature_engineering/*.yaml   │
└────────────────┬───────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────┐
│ 3. Validate Requirements                       │
│    Check: env.required_columns exist           │
│    Check: env.price_column exists              │
│    Check: observation features exist           │
└────────────────┬───────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────┐
│ 4. Create Environment                          │
│    Environment: Has access to ALL columns      │
│    - Uses env.price_column for trading         │
│    Agent: Sees only observation.input_features │
│    - Makes decisions based on these features   │
└────────────────────────────────────────────────┘
```

### Key Concept: Separation

**Environment vs Agent**:
- **Environment** gets full dataframe (ALL columns)
- **Agent** sees only `observation.input_features`
- Environment uses `env.price_column` for trading
- Agent observation space is independent

**Example**:
```yaml
# observation/minimal.yaml
input_features: ["volume"]  # Agent only sees volume

# env/default.yaml
price_column: "close"        # Environment uses close for trading
```

This works! Environment can still trade even if agent doesn't observe price.

---

## Validation

The system validates:

1. ✅ **Required columns** (`env.required_columns`) - Must exist for environment
2. ✅ **Price column** (`env.price_column`) - Must exist for trading
3. ✅ **Observation features** (`observation.input_features`) - Must exist in data (if `validate_features: true`)

All validation happens at startup with clear error messages.

---

## Next Steps

1. **Read**: [Full Configuration Guide](docs/CONFIGURATION_GUIDE.md)
2. **Quick Start**: [Quick Reference](CONFIG_QUICK_REFERENCE.md)
3. **Observation Details**: `configs/observation/README.md`
4. **Feature Engineering Details**: `configs/feature_engineering/README.md`
5. **Migration**: `configs/features/README_DEPRECATED.md` (if using old configs)

---

## Questions?

- Full architecture explanation: `docs/CONFIGURATION_GUIDE.md`
- Refactoring details: `REFACTOR_SUMMARY.md`
- Quick examples: `CONFIG_QUICK_REFERENCE.md`
