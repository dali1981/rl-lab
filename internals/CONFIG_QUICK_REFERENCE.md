# Configuration Quick Reference

## TL;DR

Three configs, three purposes:

| Config | Purpose | What it controls |
|--------|---------|------------------|
| **observation/** | What agent sees | Agent's observation space |
| **feature_engineering/** | Create features | Runtime feature creation (optional) |
| **env/** | Environment needs | Price column, required columns, trading params |

---

## Quick Start

### Default Setup (Recommended)

All features pre-computed in your data pipeline:

```bash
uv run python experiments/train.py
```

**Configs**:
- observation/default.yaml - Selects features for agent
- feature_engineering/none.yaml - No runtime engineering
- env/default.yaml - Uses 'close' for trading

---

## Common Scenarios

### 1. Change Observation Features

```bash
# Via command line
uv run python experiments/train.py \
  'observation.input_features=["close","volume","ratio_sma_5_close_zscore"]'
```

Or create `configs/observation/minimal.yaml`:
```yaml
input_features:
  - "close"
  - "volume"
  - "ratio_sma_5_close_zscore"
```

Then:
```bash
uv run python experiments/train.py observation=minimal
```

### 2. Use Different Price Column

```bash
# Use VWAP instead of close (must exist in data!)
uv run python experiments/train.py env.price_column=vwap
```

### 3. Create Returns Features at Runtime

```bash
# Enable feature engineering
uv run python experiments/train.py feature_engineering=returns
```

Then update observation config to use them:
```yaml
# observation/with_returns.yaml
input_features:
  - "close"
  - "returns_1"    # Created by feature_engineering
  - "returns_5"    # Created by feature_engineering
  - "log_returns"  # Created by feature_engineering
```

---

## Key Concepts

### Observation vs Data

```
┌─────────────────────────────────────────┐
│  DataFrame (Full Data)                   │
│  ├── close                               │
│  ├── open                                │
│  ├── high                                │
│  ├── low                                 │
│  ├── volume                              │
│  ├── ratio_sma_5_close_zscore           │
│  ├── ratio_sma_20_close_zscore          │
│  └── ...14 columns total...             │
└─────────────────────────────────────────┘
          │
          │  Environment has access to ALL columns
          │  Uses env.price_column for trading
          │
          ↓
┌─────────────────────────────────────────┐
│  Observation Space (What Agent Sees)    │
│  ├── close                               │
│  ├── volume                              │
│  ├── ratio_sma_5_close_zscore           │
│  └── ...6 features from observation config
└─────────────────────────────────────────┘
```

**Key Point**: Environment gets full dataframe, agent sees only observation features.

### Price Column

- Environment needs a price for trading (execute trades, calculate P&L)
- Configured via `env.price_column` (default: "close")
- Independent of observation features
- Agent doesn't need to observe it (but can)

Example:
```yaml
# env/default.yaml
price_column: "close"  # Used for trading

# observation/default.yaml
input_features:
  - "volume"           # Agent doesn't see close price
  - "returns_1"        # Only derived features

# Still works! Environment uses 'close' for trading
```

---

## Validation

Three levels of validation:

1. **Required columns** (env.required_columns) - Must exist for environment to function
2. **Price column** (env.price_column) - Must exist for trading
3. **Observation features** (if validate_features=true) - Must exist in data

Validation happens at startup with clear error messages.

---

## When to Use Feature Engineering

| Use Data Pipeline | Use Runtime Engineering |
|-------------------|------------------------|
| ✓ Production | ✓ Quick prototyping |
| ✓ Complex features | ✓ Testing parameters |
| ✓ Reusable | ✓ One-off experiments |
| ✓ Kedro/Airflow | ✗ Production |

**Recommendation**: Use Kedro for feature creation, use observation config to select features.

---

## File Locations

```
configs/
├── config.yaml                    # Main config
│
├── observation/
│   └── default.yaml              # What agent observes
│
├── feature_engineering/
│   ├── none.yaml                 # No engineering (default)
│   └── returns.yaml              # Example: create returns
│
└── env/
    └── default.yaml              # Environment requirements
```

---

## Need Help?

- **Full guide**: `docs/CONFIGURATION_GUIDE.md`
- **Refactor summary**: `REFACTOR_SUMMARY.md`
- **Test config**: `uv run python -c "from rl_trading_lab.config import load_config; ..."`

---

## Examples

```bash
# 1. Default (pre-computed features)
uv run python experiments/train.py

# 2. Minimal observation space
uv run python experiments/train.py \
  'observation.input_features=["close","volume"]'

# 3. With runtime feature engineering
uv run python experiments/train.py \
  feature_engineering=returns

# 4. Custom price column
uv run python experiments/train.py \
  env.price_column=vwap

# 5. Different reward type
uv run python experiments/train.py \
  env.environment_params.reward_type=sharpe

# 6. Combine multiple overrides
uv run python experiments/train.py \
  observation=minimal \
  feature_engineering=returns \
  env.price_column=close \
  agent=a2c
```
