# Logging Configuration Guide

## Quick Reference

### Run with default (clean) output
```bash
python experiments/train.py
```
No trade-by-trade logs, only important info.

### Enable detailed trade logs for debugging
```bash
PYTHONLOGLEVEL=DEBUG python experiments/train.py
```
Shows every trade execution and position change.

### Suppress everything except errors
```bash
PYTHONLOGLEVEL=ERROR python experiments/train.py
```
Silent operation, only shows errors.

---

## Log Levels Explained

| Level | Priority | What You'll See | Use When |
|-------|----------|-----------------|----------|
| **DEBUG** | 10 | Everything - every trade, position change, calculation | Debugging trade logic, investigating issues |
| **INFO** | 20 | Important events - training progress, evaluations, final metrics | **Normal training (default)** ✓ |
| **WARNING** | 30 | Potential issues - deprecated features, unusual behavior | Production runs |
| **ERROR** | 40 | Serious problems - failures, exceptions | CI/CD, automated runs |
| **CRITICAL** | 50 | System failures - unrecoverable errors | N/A |

---

## Method 1: Environment Variable (Easiest)

### Set for current terminal session
```bash
export PYTHONLOGLEVEL=DEBUG
python experiments/train.py
python experiments/train.py  # Still DEBUG
```

### Set for single command
```bash
PYTHONLOGLEVEL=INFO python experiments/train.py
```

### Unset
```bash
unset PYTHONLOGLEVEL
```

---

## Method 2: Modify train.py Directly

Edit `experiments/train.py` line 35:

### For clean output (default):
```python
logging.basicConfig(level=logging.INFO)  # Current setting
```

### For detailed debugging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### For minimal output:
```python
logging.basicConfig(level=logging.WARNING)
```

---

## Method 3: Per-Module Control (Advanced)

Add after line 36 in `experiments/train.py`:

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Control specific modules independently
logging.getLogger('rl_trading_lab.environment.trading_env').setLevel(logging.WARNING)  # Hide all trade logs
logging.getLogger('rl_trading_lab.agents').setLevel(logging.DEBUG)  # Show agent details
logging.getLogger('rl_trading_lab.utils.callbacks').setLevel(logging.INFO)  # Normal callbacks
```

### Common Combinations:

**Training Mode (clean output):**
```python
logging.basicConfig(level=logging.INFO)
logging.getLogger('rl_trading_lab.environment.trading_env').setLevel(logging.INFO)
```

**Debug Trading Logic:**
```python
logging.basicConfig(level=logging.INFO)
logging.getLogger('rl_trading_lab.environment.trading_env').setLevel(logging.DEBUG)  # Trade details
```

**Debug Agent Only:**
```python
logging.basicConfig(level=logging.INFO)
logging.getLogger('rl_trading_lab.environment.trading_env').setLevel(logging.WARNING)  # Hide trades
logging.getLogger('rl_trading_lab.agents').setLevel(logging.DEBUG)  # Show agent
```

---

## Method 4: Logging Config File (Production)

Create `configs/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  simple:
    format: '[%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout

  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/training.log

root:
  level: INFO
  handlers: [console, file]

loggers:
  rl_trading_lab.environment.trading_env:
    level: WARNING  # Hide trade logs from console
    handlers: [file]  # But save to file
    propagate: false

  rl_trading_lab.agents:
    level: INFO
    propagate: true
```

Then in `train.py`:
```python
import logging.config
import yaml

with open('configs/logging.yaml') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)
```

---

## What Gets Logged at Each Level

### DEBUG Level (Most Verbose)
```
[DEBUG] rl_trading_lab.environment.trading_env: Trade #1: LONG 0.0809 @ $121690.38
[DEBUG] rl_trading_lab.environment.trading_env: Closing position: current=0.0809, signal=-1.0
[DEBUG] rl_trading_lab.environment.trading_env: Position closed: P&L=$3.99, Commission=$0.00, Net=$3.99
[DEBUG] rl_trading_lab.environment.trading_env: Opening position: signal=-1.0, price=121712.63
[DEBUG] rl_trading_lab.environment.trading_env: Trade #2: SHORT 0.0809 @ $121712.63
... (1000s of lines during training)
```

### INFO Level (Default, Clean)
```
[INFO] rl_trading_lab.environment.trading_env: TradingEnv initialized: randomize_start=True, hold_closes_position=True
Training PPO Agent...
Eval num_timesteps=5000, episode_reward=-150.23 +/- 45.12
Eval num_timesteps=10000, episode_reward=-120.45 +/- 38.76
✓ Training completed
✓ Backtest completed
  Steps: 321
  Final Return: -6.17%
  Total Trades: 2
```

### WARNING Level (Minimal)
```
Only warnings and errors
```

---

## Troubleshooting

### Q: I changed the level but still see trade logs
**A:** Make sure you're not using cached Python bytecode:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
python experiments/train.py
```

### Q: I want trade logs in a file but not on console
**A:** Use Method 4 (logging config file) with different handlers for different modules.

### Q: How do I log to both console and file?
**A:** Use Method 4 and add both handlers:
```python
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
  file:
    class: logging.FileHandler
    level: DEBUG
    filename: logs/debug.log

root:
  handlers: [console, file]
```

### Q: Environment variable not working
**A:** Check if it's set:
```bash
echo $PYTHONLOGLEVEL
```

If empty, set it:
```bash
export PYTHONLOGLEVEL=DEBUG
```

---

## Examples by Use Case

### 1. Normal Training (Clean Console)
```bash
# Default - just run it
python experiments/train.py
```

### 2. Debugging Trade Logic
```bash
# See every trade
PYTHONLOGLEVEL=DEBUG python experiments/train.py > debug.log 2>&1
# Then review: less debug.log
```

### 3. Production Run (Silent)
```bash
# Only errors
PYTHONLOGLEVEL=ERROR python experiments/train.py
```

### 4. Debugging Specific Issue
```python
# In train.py, after line 36:
logging.getLogger('rl_trading_lab.environment.trading_env').setLevel(logging.DEBUG)
# Run normally
```

### 5. Save Everything to File
```bash
# Console stays clean, file gets everything
PYTHONLOGLEVEL=INFO python experiments/train.py 2>&1 | tee training.log
```

---

## Summary

**Most Common Usage:**

```bash
# ✓ Clean output (default)
python experiments/train.py

# ✓ Debug a specific run
PYTHONLOGLEVEL=DEBUG python experiments/train.py

# ✓ Silent operation
PYTHONLOGLEVEL=ERROR python experiments/train.py
```

**Quick Reference Card:**

| You Want... | Command |
|-------------|---------|
| Clean training output | `python experiments/train.py` |
| See all trade details | `PYTHONLOGLEVEL=DEBUG python experiments/train.py` |
| Only errors | `PYTHONLOGLEVEL=ERROR python experiments/train.py` |
| Save to file | `python experiments/train.py > training.log 2>&1` |
