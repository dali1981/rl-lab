# Configuration Refactoring Summary

## Problem

The original `features.yaml` configuration mixed three separate concerns:
1. Feature engineering (creating returns, log_returns, etc.)
2. Observation space selection (what the agent sees)
3. Environment requirements (implied, not explicit)

This caused confusion:
- Unclear what `features.yaml` actually does
- Is it computing features or selecting them? Both!
- How does the environment get `close` price if it's not in `input_features`?
- No way to configure price column (hardcoded)

## Solution

Split into three clean configs with single responsibilities:

### 1. Observation Config (`configs/observation/`)
**Purpose**: What the RL agent observes
**Controls**: Agent's observation space only

```yaml
input_features:
  - "close"
  - "volume"
  - "ratio_sma_5_close_zscore"
```

### 2. Feature Engineering Config (`configs/feature_engineering/`)
**Purpose**: Create new features at runtime (optional)
**Controls**: Feature creation only

```yaml
enabled: true
add_returns: true
return_periods: [1, 5, 20]
```

### 3. Environment Config (`configs/env/`)
**Purpose**: Environment requirements and parameters
**Controls**: What environment needs to function

```yaml
price_column: "close"
required_columns: ["close", "timestamp"]
environment_params: {...}
```

## Key Changes

### Config Schema

**Created**:
- `src/rl_trading_lab/config/observation.py` - ObservationConfig
- `src/rl_trading_lab/config/feature_engineering.py` - FeatureEngineeringConfig
- Updated `src/rl_trading_lab/config/env.py` - Added price_column, required_columns
- Updated `src/rl_trading_lab/config/main.py` - RootConfig uses new configs

**Config Files**:
- `configs/observation/default.yaml` - Default observation config
- `configs/feature_engineering/none.yaml` - No feature engineering
- `configs/feature_engineering/returns.yaml` - Example with returns
- Updated `configs/env/default.yaml` - Added price_column and required_columns

### Code Changes

**Created**:
- `src/rl_trading_lab/utils/data_processor.py` - Clean data processing pipeline
  - Separates: load → engineer → select → split
  - Clear single-responsibility methods

**Updated**:
- `src/rl_trading_lab/environment/trading_env.py`:
  - Added `price_column` parameter
  - Validates price column exists
  - Uses `self.df[self.price_column]` instead of hardcoded `['close']`

- `experiments/train.py`:
  - Uses DataProcessor instead of TradingDataLoader
  - Passes price_column to TradingEnv
  - Validates required columns exist

### Documentation

**Created**:
- `docs/CONFIGURATION_GUIDE.md` - Comprehensive guide
- `REFACTOR_SUMMARY.md` - This file

## Benefits

### 1. Clear Separation of Concerns
- ✓ Each config has single responsibility
- ✓ Easy to understand what each config does
- ✓ No more dual-mode confusing logic

### 2. Explicit Requirements
- ✓ Price column is configurable, not hardcoded
- ✓ Required columns are validated
- ✓ Clear error messages when validation fails

### 3. Better Architecture
- ✓ Environment gets full dataframe
- ✓ Agent observation space is separate
- ✓ Feature engineering is optional

### 4. More Maintainable
- ✓ Easier to extend (add new engineering options)
- ✓ Easier to test (each component isolated)
- ✓ Better documentation

## Architecture

### Old Flow
```
features.yaml → TradingDataLoader
  ├── Select OR build features (confusing!)
  ├── Validate (sometimes)
  └── Return df + feature_names

TradingEnv:
  - Hardcoded df['close'] for trading
  - features_to_use for observation space
```

### New Flow
```
observation.yaml + feature_engineering.yaml + env.yaml
  ↓
DataProcessor:
  1. Load raw data (all columns)
  2. Engineer features (if enabled)
  3. Validate requirements
  4. Select observation features
  5. Split data
  ↓
TradingEnv:
  - Full df available
  - Uses config.env.price_column for trading
  - Uses observation_features for agent obs space
```

## Testing

All functionality verified:
```bash
# Test config loads
✓ Config structure validated
✓ Observation features: 6
✓ Feature engineering: disabled
✓ Price column: close

# Test data processing
✓ Data processor loads data
✓ Feature engineering creates new columns
✓ Observation features selected correctly

# Test environment
✓ Environment uses configurable price_column
✓ Observation space correct size
✓ All features validated
```

## Backward Compatibility

Old `features.yaml` configs are deprecated but kept for reference:
- `configs/features/fractional_indicators.yaml` - Kept as example
- `configs/features/minimal_example.yaml` - Kept as example

To migrate:
1. Move `input_features` → `configs/observation/default.yaml`
2. Move `feature_engineering` → `configs/feature_engineering/returns.yaml`
3. Update `configs/env/default.yaml` with price_column and required_columns

## Files Changed

### Created
- `src/rl_trading_lab/config/observation.py`
- `src/rl_trading_lab/config/feature_engineering.py`
- `src/rl_trading_lab/utils/data_processor.py`
- `configs/observation/default.yaml`
- `configs/feature_engineering/none.yaml`
- `configs/feature_engineering/returns.yaml`
- `docs/CONFIGURATION_GUIDE.md`
- `REFACTOR_SUMMARY.md`

### Modified
- `src/rl_trading_lab/config/env.py` - Added price_column, required_columns
- `src/rl_trading_lab/config/main.py` - Updated RootConfig
- `src/rl_trading_lab/config/__init__.py` - Updated exports
- `src/rl_trading_lab/environment/trading_env.py` - Configurable price_column
- `experiments/train.py` - Uses DataProcessor, passes price_column
- `configs/config.yaml` - Uses observation and feature_engineering
- `configs/env/default.yaml` - Added price_column and required_columns

### Deprecated (kept for reference)
- `src/rl_trading_lab/config/features.py` - Use observation.py and feature_engineering.py
- `src/rl_trading_lab/utils/data_loader.py` - Use data_processor.py
- `configs/features/*.yaml` - Use observation/ and feature_engineering/

## Next Steps

1. Update any custom scripts to use new config structure
2. Update notebooks to use DataProcessor
3. Delete deprecated files after confirming everything works
4. Update any documentation or tutorials

## Questions?

See `docs/CONFIGURATION_GUIDE.md` for detailed usage guide and FAQs.
