# Observation Space Configurations

This directory contains configs that define **what the RL agent observes** (its observation space).

## Purpose

The observation config specifies which features from your data should be included in the agent's observation space. These features must already exist in your data file - they are NOT created here.

## Available Configs

| Config | Description | Features | Use Case |
|--------|-------------|----------|----------|
| **default.yaml** | Price + Volume + Indicators | 6 features | Recommended starting point |
| **minimal.yaml** | Just price and volume | 2 features | Simplest setup |
| **price_only.yaml** | OHLC only | 4 features | Price action trading |
| **with_volume.yaml** | OHLCV | 5 features | Standard market data |
| **indicators_only.yaml** | Technical indicators only | 4+ features | Pure indicator-based |
| **with_returns.yaml** | Price + Returns | 6 features | Requires feature_engineering=returns |

## Usage

### Use a pre-defined config

```bash
# Default config
uv run python experiments/train.py

# Minimal observation space
uv run python experiments/train.py observation=minimal

# Price only
uv run python experiments/train.py observation=price_only

# With returns (requires feature engineering)
uv run python experiments/train.py \
  observation=with_returns \
  feature_engineering=returns
```

### Override via command line

```bash
# Specify features directly
uv run python experiments/train.py \
  'observation.input_features=["close","volume","ratio_sma_5_close_zscore"]'
```

### Create your own config

Create `configs/observation/my_features.yaml`:

```yaml
# My Custom Observation Space
input_features:
  - "close"
  - "volume"
  - "my_custom_feature"  # Must exist in your data!

validate_features: true
log_all_features: true
```

Then use it:
```bash
uv run python experiments/train.py observation=my_features
```

## Important Notes

### Features Must Exist in Data

All features in `input_features` must exist in your data file. The observation config does NOT create features - it only selects them.

**Example error if feature doesn't exist:**
```
ValueError: Features specified in observation.input_features not found in data:
  Missing: ['my_feature']
  Available columns: ['close', 'open', 'high', 'low', 'volume', ...]
```

### Environment vs Agent

The **environment** has access to ALL columns in the dataframe:
- Uses `env.price_column` for trading (default: "close")
- Can access any column for calculations

The **agent** only sees features in `observation.input_features`:
- This is the agent's observation space
- The agent makes decisions based only on these features

**Example:**
```yaml
# observation/minimal.yaml
input_features:
  - "volume"  # Agent ONLY sees volume

# env/default.yaml
price_column: "close"  # Environment still uses 'close' for trading
```

This works! The agent sees only volume, but the environment can still execute trades using the close price.

## Observation Space Size

The observation space size is:
```
obs_size = len(input_features) * lookback_window + 4
```

Where:
- `input_features`: Features specified in this config
- `lookback_window`: From env config (default: 20)
- `+4`: Position info (position, entry_price, pnl, cash_pct)

**Example:**
- 6 features × 20 lookback = 120
- +4 position features = 124 total

## Validation

When `validate_features: true` (recommended):
- Checks that all specified features exist in data
- Provides clear error messages if missing
- Happens at startup before training begins

## Feature Engineering

If you need to create features at runtime (not recommended for production):

1. Enable feature engineering:
   ```bash
   uv run python experiments/train.py feature_engineering=returns
   ```

2. Reference the created features in your observation config:
   ```yaml
   input_features:
     - "close"
     - "returns_1"    # Created by feature_engineering
     - "log_returns"  # Created by feature_engineering
   ```

**Recommended:** Create features in your data pipeline (Kedro) instead of at runtime.

## Examples

### Example 1: Simple OHLCV Trading

```bash
uv run python experiments/train.py observation=with_volume
```

### Example 2: Indicator-Based Trading

```bash
uv run python experiments/train.py observation=indicators_only
```

### Example 3: Returns-Based Trading

```bash
uv run python experiments/train.py \
  observation=with_returns \
  feature_engineering=returns
```

### Example 4: Custom Feature Selection

```bash
uv run python experiments/train.py \
  'observation.input_features=["close","volume","ratio_sma_5_close_zscore","ratio_sma_20_close_zscore"]'
```

## See Also

- **Full guide**: `docs/CONFIGURATION_GUIDE.md`
- **Quick reference**: `CONFIG_QUICK_REFERENCE.md`
- **Feature engineering**: `configs/feature_engineering/README.md`
