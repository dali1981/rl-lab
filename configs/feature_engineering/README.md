# Feature Engineering Configurations

This directory contains configs for **runtime feature engineering** (creating new features on-the-fly).

⚠️ **Note**: For production, prefer creating features in your data pipeline (Kedro) rather than at runtime.

## Purpose

Feature engineering configs control whether to create new features at runtime, such as:
- Returns (percentage returns)
- Log returns
- Rolling statistics (mean, std, min, max)

## When to Use

| Use Runtime Engineering | Use Data Pipeline |
|------------------------|-------------------|
| ✓ Quick prototyping | ✓ Production |
| ✓ Testing parameters | ✓ Complex features |
| ✓ One-off experiments | ✓ Reusable across projects |
| ✗ Production | ✓ Better performance |

**Recommendation**: Use `none.yaml` (default) and create features in Kedro.

## Available Configs

| Config | Description | Creates | Use Case |
|--------|-------------|---------|----------|
| **none.yaml** | No feature engineering (default) | Nothing | Production (features in data pipeline) |
| **simple_returns.yaml** | Single-period returns | returns_1 | Minimal testing |
| **returns.yaml** | Multiple returns periods | returns_1, returns_5, returns_20, log_returns | Quick prototyping |
| **full.yaml** | All features including rolling | Returns + log_returns + rolling stats | Experimentation only (slow) |

## Usage

### Default: No Feature Engineering (Recommended)

```bash
# Uses feature_engineering=none (default)
uv run python experiments/train.py
```

All features come from your data file.

### Enable Returns Engineering

```bash
# Create returns features at runtime
uv run python experiments/train.py feature_engineering=returns
```

This creates: `returns_1`, `returns_5`, `returns_20`, `log_returns`

You must reference these in your observation config:
```yaml
# observation/with_returns.yaml
input_features:
  - "close"
  - "returns_1"    # Created by feature_engineering
  - "returns_5"
  - "log_returns"
```

### Full Feature Engineering (Testing Only)

```bash
# WARNING: Slow! Creates many features including rolling stats
uv run python experiments/train.py feature_engineering=full
```

## Feature Engineering Options

### Returns Features

```yaml
add_returns: true
return_periods: [1, 5, 20]
```

Creates:
- `returns_1`: 1-period percentage returns
- `returns_5`: 5-period percentage returns
- `returns_20`: 20-period percentage returns

Formula: `(price[t] - price[t-n]) / price[t-n]`

### Log Returns

```yaml
add_log_returns: true
```

Creates:
- `log_returns`: Natural log returns

Formula: `ln(price[t] / price[t-1])`

### Rolling Statistics

```yaml
add_rolling_stats: true
rolling_window: 20
rolling_stats: ["mean", "std", "min", "max"]
```

Creates:
- `rolling_mean_20`: 20-period moving average
- `rolling_std_20`: 20-period standard deviation
- `rolling_min_20`: 20-period minimum
- `rolling_max_20`: 20-period maximum

⚠️ **Warning**: Rolling statistics are computationally expensive. Use in data pipeline instead.

### Missing Values Handling

```yaml
missing_values:
  strategy: "forward_fill"  # forward_fill, interpolate, drop
  initial_fill: 0.0
```

Options:
- `forward_fill`: Fill NaN with previous value (default)
- `interpolate`: Linear interpolation
- `drop`: Drop rows with NaN
- `initial_fill`: Value to use for NaN at start of data

## Examples

### Example 1: No Engineering (Production)

```bash
# Default - all features from data pipeline
uv run python experiments/train.py feature_engineering=none

# Use observation config that references only existing features
uv run python experiments/train.py \
  observation=default \
  feature_engineering=none
```

### Example 2: Simple Returns Testing

```bash
# Create just returns_1 for quick test
uv run python experiments/train.py \
  feature_engineering=simple_returns \
  'observation.input_features=["close","returns_1"]'
```

### Example 3: Multiple Returns Periods

```bash
# Create multiple return periods
uv run python experiments/train.py \
  observation=with_returns \
  feature_engineering=returns
```

### Example 4: Custom Engineering

Create `configs/feature_engineering/my_features.yaml`:
```yaml
enabled: true

add_returns: true
return_periods: [1, 2, 3]  # Custom periods

add_log_returns: false  # Disable log returns

missing_values:
  strategy: "interpolate"  # Custom strategy
  initial_fill: 0.0
```

Then use:
```bash
uv run python experiments/train.py feature_engineering=my_features
```

## Architecture

### Data Flow

```
Load Raw Data
    ↓
Feature Engineering (if enabled)
    ├── Add returns_1, returns_5, returns_20
    ├── Add log_returns
    ├── Add rolling statistics (if enabled)
    └── Handle missing values
    ↓
DataFrame with engineered features
    ↓
Observation Selection
    ↓
Select features for agent observation space
```

### Created Features

Features created here are:
1. Added to the dataframe
2. Available for observation config selection
3. Available to environment (like all columns)

**Example:**
```yaml
# feature_engineering/returns.yaml
enabled: true
add_returns: true
return_periods: [1, 5]

# observation/my_config.yaml
input_features:
  - "close"
  - "returns_1"  # ✓ Available (created by engineering)
  - "returns_5"  # ✓ Available (created by engineering)
```

## Performance

### Runtime Cost

| Feature Type | Cost | Recommendation |
|--------------|------|----------------|
| Returns | Low | OK for prototyping |
| Log Returns | Low | OK for prototyping |
| Rolling Stats | **High** | Use data pipeline |

Rolling statistics require computation over sliding windows and significantly slow down data loading.

### When to Avoid

Avoid runtime feature engineering when:
- Running production training
- Training on large datasets
- Creating complex features
- Using rolling statistics

Instead: Create features in Kedro pipeline once, reuse many times.

## Validation

The feature engineering config validates:
1. At least one feature type enabled if `enabled: true`
2. Return periods are valid integers
3. Rolling window is positive
4. Missing value strategy is valid

## Troubleshooting

### "Feature returns_1 not found in data"

**Problem**: Observation config references engineered features but engineering is disabled.

**Solution**: Enable feature engineering:
```bash
uv run python experiments/train.py feature_engineering=returns
```

### "Feature engineering is enabled but no features selected"

**Problem**: `enabled: true` but all feature types disabled.

**Solution**: Enable at least one feature type:
```yaml
enabled: true
add_returns: true  # Enable this
```

### Slow data loading

**Problem**: Using rolling statistics.

**Solution**:
1. Disable rolling stats: `add_rolling_stats: false`
2. Or create features in data pipeline instead

## See Also

- **Full guide**: `docs/CONFIGURATION_GUIDE.md`
- **Quick reference**: `CONFIG_QUICK_REFERENCE.md`
- **Observation configs**: `configs/observation/README.md`
