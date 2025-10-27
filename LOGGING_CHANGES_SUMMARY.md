# Logging Changes Summary

## What Changed

### Trade logs moved from INFO to DEBUG level

**Modified file:** `src/rl_trading_lab/environment/trading_env.py`

**Lines changed:**
- Line 303: `logger.info(...)` → `logger.debug(...)` (Trade execution)
- Line 348: `logger.info(...)` → `logger.debug(...)` (Position closing)
- Line 360: `logger.info(...)` → `logger.debug(...)` (Episode end closing)

**Line 120 unchanged:** Environment initialization remains at INFO level (only happens once).

---

## Impact

### Before (INFO level):
```
[INFO] - TradingEnv initialized: randomize_start=True
[INFO] - Trade #1: LONG 0.0809 @ $121690.38
[INFO] - Position closed: P&L=$3.99, Commission=$0.00, Net=$3.99
[INFO] - Trade #2: SHORT 0.0809 @ $121712.63
[INFO] - Position closed: P&L=$0.89, Commission=$0.00, Net=$0.89
... (1000+ lines during training)
```
**Problem:** Console flooded with trade logs during training

### After (INFO level - DEFAULT):
```
[INFO] - TradingEnv initialized: randomize_start=True
Training PPO Agent...
Eval num_timesteps=5000, episode_reward=-150.23 +/- 45.12
✓ Training completed
```
**Result:** ✓ Clean, readable output

### After (DEBUG level enabled):
```
[INFO] - TradingEnv initialized: randomize_start=True
[DEBUG] - Trade #1: LONG 0.0809 @ $121690.38
[DEBUG] - Position closed: P&L=$3.99
...
```
**Result:** ✓ Detailed logs available when needed for debugging

---

## How to Use

### Default Operation (Clean Output)
```bash
python experiments/train.py
```
No change needed - trade logs are now hidden by default.

### Enable Trade Logs for Debugging
```bash
# Temporary (single run)
PYTHONLOGLEVEL=DEBUG python experiments/train.py

# Persistent (current session)
export PYTHONLOGLEVEL=DEBUG
python experiments/train.py
```

### Disable All Logs Except Errors
```bash
PYTHONLOGLEVEL=ERROR python experiments/train.py
```

---

## Testing

Verified with test script `test_logging_levels.py`:

```bash
# Run test (default INFO level)
uv run python test_logging_levels.py

# Shows:
# - INFO: Only environment initialization
# - DEBUG: Not shown (trade logs hidden) ✓

# Run test with DEBUG enabled
PYTHONLOGLEVEL=DEBUG uv run python test_logging_levels.py

# Shows:
# - INFO: Environment initialization
# - DEBUG: All trade logs visible ✓
```

**All unit tests still pass:**
```bash
uv run pytest tests/test_trading_env.py -v
# 19 passed in 0.39s ✓
```

---

## Documentation

Created comprehensive guide: `docs/LOGGING_GUIDE.md`

**Contents:**
- Quick reference for common use cases
- Log level explanations
- 4 methods to control logging (environment variable, code, per-module, config file)
- Troubleshooting guide
- Examples by use case

---

## Backward Compatibility

✓ **No breaking changes**
- Default behavior is cleaner (less noise)
- All functionality preserved
- Can still access trade logs via DEBUG level
- No impact on training, metrics, or callbacks

---

## Files Created/Modified

### Modified:
- ✅ `src/rl_trading_lab/environment/trading_env.py` (3 lines changed)

### Created:
- ✅ `docs/LOGGING_GUIDE.md` (comprehensive logging reference)
- ✅ `test_logging_levels.py` (demonstration script)
- ✅ `LOGGING_CHANGES_SUMMARY.md` (this file)

---

## Quick Reference

| Log Level | Shows Trade Logs? | Use Case |
|-----------|-------------------|----------|
| DEBUG | ✓ Yes | Debugging trade logic |
| **INFO** (default) | ✗ No | **Normal training** ✓ |
| WARNING | ✗ No | Production runs |
| ERROR | ✗ No | Silent operation |

**Most common commands:**

```bash
# Clean output (default)
python experiments/train.py

# Debug mode
PYTHONLOGLEVEL=DEBUG python experiments/train.py

# Silent mode
PYTHONLOGLEVEL=ERROR python experiments/train.py
```
