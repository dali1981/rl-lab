# Configuration Guide

## Architecture Overview

The RL Trading Lab configuration system cleanly separates three concerns:

1. **Observation Space** (`configs/observation/`) - What the RL agent observes
2. **Feature Engineering** (`configs/feature_engineering/`) - Creating new features (optional)
3. **Environment Requirements** (`configs/env/`) - What the environment needs to function

This separation ensures:
- Clear understanding of what each config does
- Environment always has data it needs (price column, required columns)
- Agent observation space is explicitly defined
- Feature engineering is optional and separate

---

## Configuration Structure

```
configs/
├── config.yaml              # Main config with defaults
├── env/
│   └── default.yaml         # Environment config (price_column, required_columns, params)
├── observation/
│   └── default.yaml         # What agent observes
├── feature_engineering/
│   ├── none.yaml            # No feature engineering (default)
│   └── returns.yaml         # Example: create returns features
├── agent/
│   ├── ppo.yaml
│   ├── a2c.yaml
│   └── dqn.yaml
└── data/
    └── ...
```

---

## 1. Observation Space Configuration

**Location**: `configs/observation/default.yaml`

**Purpose**: Controls which features the RL agent sees in its observation space.

```yaml
# Observation Space Configuration
input_features:
  - "close"
  - "volume"
  - "ratio_sma_5_close_zscore"
  - "ratio_sma_20_close_zscore"

validate_features: true
log_all_features: true
```

**Key Points**:
- Features MUST exist in your data
- These are NOT created at runtime (unless you enable feature_engineering)
- This defines the agent's observation space only
- Environment can access any column from the data

---

## 2. Environment Configuration

**Location**: `configs/env/default.yaml`

**Purpose**: Controls environment requirements and trading parameters.

```yaml
# Environment data requirements
price_column: "close"  # Column for trade execution
required_columns:      # Must exist in data
  - "close"
  - "timestamp"

environment_params:
  lookback_window: 20
  initial_balance: 10_000
  commission_rate: 0.00
  slippage_rate: 0.00
  reward_type: "returns"  # returns, sharpe, or pnl
  discrete_actions: true
  randomize_start: true
  min_episode_length: 100
  hold_closes_position: true
```

**Key Points**:
- `price_column`: What price to use for P&L calculation (typically "close")
- `required_columns`: Columns environment needs (validated at startup)
- Environment has access to full dataframe regardless of observation features

---

## 3. Feature Engineering Configuration

**Location**: `configs/feature_engineering/`

**Purpose**: Create new features at runtime (optional - most users create features in data pipeline).

### Option A: No Feature Engineering (Default)

```yaml
# configs/feature_engineering/none.yaml
enabled: false
```

### Option B: Create Features at Runtime

```yaml
# configs/feature_engineering/returns.yaml
enabled: true

# Returns features
add_returns: true
return_periods: [1, 5, 20]

# Log returns
add_log_returns: true

# Rolling statistics (expensive)
add_rolling_stats: false
rolling_window: 20
rolling_stats: ["mean", "std"]

# Missing values handling
missing_values:
  strategy: "forward_fill"  # forward_fill, interpolate, drop
  initial_fill: 0.0
```

**When to use feature engineering**:
- ✓ Quick prototyping without modifying data pipeline
- ✓ Testing different return periods or rolling windows
- ✗ Production (use Kedro or your data pipeline instead)

---

## Data Flow

```
┌─────────────────────┐
│  1. Load Raw Data   │  All columns from parquet file
│     (DataProcessor) │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  2. Engineer        │  Optional: Add returns, log_returns, etc.
│     Features        │  (if feature_engineering.enabled=true)
│                     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  3. Validate        │  Check env.required_columns exist
│     Requirements    │  Check price_column exists
│                     │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  4. Create          │  - Full df passed to environment
│     Environment     │  - observation.input_features for obs space
│                     │  - env.price_column for trading
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Environment:       │
│  - Has full df      │  Can access ANY column (close, volume, etc.)
│  - Uses price_col   │  For trade execution and P&L
│                     │
│  Agent:             │
│  - Sees only        │  observation.input_features
│    obs features     │  (subset of df columns)
└─────────────────────┘
```

---

## Common Use Cases

### Use Case 1: Simple - Pre-computed Features

All features created in your data pipeline (e.g., Kedro).

```bash
# Use default configs
uv run python experiments/train.py
```

**Configs used**:
- `observation/default.yaml` - Selects which features agent sees
- `feature_engineering/none.yaml` - No runtime feature creation
- `env/default.yaml` - Uses "close" for trading

### Use Case 2: Runtime Feature Engineering

Create returns features at runtime for quick testing.

```bash
# Enable feature engineering
uv run python experiments/train.py feature_engineering=returns
```

**Configs used**:
- `observation/default.yaml` - Agent observation space
- `feature_engineering/returns.yaml` - Creates returns_1, returns_5, etc.
- `env/default.yaml` - Environment settings

### Use Case 3: Custom Observation Features

Override which features the agent observes.

```bash
# Specify features via command line
uv run python experiments/train.py \
  'observation.input_features=["close","volume","ratio_sma_5_close_zscore"]'
```

### Use Case 4: Different Price Column

Use a different price column (e.g., VWAP instead of close).

```bash
# Use VWAP for trading (must exist in your data!)
uv run python experiments/train.py env.price_column=vwap
```

---

## FAQs

### Q: If "close" is not in observation.input_features, how does the environment trade?

**A**: The environment receives the FULL dataframe (with ALL columns). The `observation.input_features` only controls what the **agent** sees, not what the **environment** can access. The environment uses `env.price_column` (typically "close") for trading, regardless of what's in the observation space.

### Q: When should I use feature_engineering vs data pipeline?

**A**:
- **Data pipeline** (Kedro): Production use, complex features, reusable across projects
- **Runtime engineering**: Quick prototyping, testing different parameters

### Q: Can I use engineered features in observation space?

**A**: Yes, if `feature_engineering.enabled=true`, you can add engineered features to your observation space:

```yaml
# observation/default.yaml
input_features:
  - "close"
  - "returns_1"      # Created by feature_engineering
  - "log_returns"    # Created by feature_engineering
```

But you must enable feature engineering first! Otherwise these columns won't exist.

### Q: What's validated?

**A**:
1. `env.required_columns` - Must exist in data
2. `env.price_column` - Must exist in data
3. `observation.input_features` - Must exist in data (if validate_features=true)

---

## Migration from Old Config

If you have old configs using `features.yaml`:

**Old structure**:
```yaml
# configs/features/fractional_indicators.yaml
input_features: null
price_features: [...]
technical_indicators: [...]
feature_engineering: {...}
```

**New structure**:
```yaml
# configs/observation/default.yaml
input_features: ["close", "volume", ...]

# configs/feature_engineering/none.yaml
enabled: false

# configs/env/default.yaml
price_column: "close"
required_columns: ["close", "timestamp"]
```

**Benefits**:
- ✓ Clearer separation of concerns
- ✓ Explicit price column configuration
- ✓ No confusing dual-mode logic
- ✓ Easier to understand and maintain

---

## Examples

See `configs/observation/` and `configs/feature_engineering/` for example configurations.

Test your config:
```bash
# Validate config loads
uv run python -c "
from rl_trading_lab.config import load_config
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = str(Path.cwd() / 'configs')
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name='config')
    config = load_config(cfg)
    print('✓ Config valid')
    print(f'Obs features: {len(config.observation.input_features)}')
    print(f'Price column: {config.env.price_column}')
"
```
