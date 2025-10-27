# Fixes Applied to Training/Evaluation Disparity

**Date**: 2025-10-27
**Status**: ✅ Changes Applied, Ready for Testing

## Summary

Applied critical fixes to resolve observation normalization mismatch and invalid Sharpe calculation that were causing misleading evaluation metrics.

---

## Changes Made

### 1. ✅ Added VecNormalize Wrapper for Test Environment

**File**: `experiments/train.py`

**Added function** `wrap_test_env_for_evaluation()`:
```python
def wrap_test_env_for_evaluation(test_env, agent):
    """
    Wrap test environment to match training setup.

    CRITICAL: Copies normalization statistics from training environment
    so test observations have same scale as training.
    """
    test_vec_env = DummyVecEnv([lambda e=test_env: e])
    test_vec_env = VecNormalize(
        test_vec_env,
        norm_obs=True,       # Normalize observations (CRITICAL)
        norm_reward=False,   # Don't normalize rewards during eval
        clip_obs=10.0,
        training=False,      # Don't update running statistics
    )

    # Copy normalization statistics from training
    test_vec_env.obs_rms = agent.env.obs_rms
    test_vec_env.ret_rms = agent.env.ret_rms

    return test_vec_env
```

**Impact**: Test environment now uses same observation normalization as training!

---

### 2. ✅ Updated `evaluate_final_performance()`

**File**: `experiments/train.py:204-236`

**Changes**:
- Now wraps test_env before evaluation
- Added debugging output for reward statistics
- Ensures consistent observation scaling

**Before**:
```python
metrics = agent.evaluate(env=test_env, ...)  # Raw env - WRONG!
```

**After**:
```python
test_env_wrapped = wrap_test_env_for_evaluation(test_env, agent)
metrics = agent.evaluate(env=test_env_wrapped, ...)  # Properly wrapped!
```

---

### 3. ✅ Fixed Backtest Sharpe Calculation

**File**: `experiments/train.py:239-338`

**Major Changes**:

#### A. Wrapped Test Environment
```python
# Now uses wrapped environment instead of raw
test_env_wrapped = wrap_test_env_for_evaluation(test_env, agent)
```

#### B. Fixed Sharpe Calculation

**Before** (WRONG - Sharpe of Sharpe ratios):
```python
sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
# rewards are already Sharpe approximations, this is meaningless!
```

**After** (CORRECT - Sharpe from actual returns):
```python
# Track actual portfolio returns
step_returns = []
for each step:
    step_return = (balances[-1] - balances[-2]) / balances[-2]
    step_returns.append(step_return)

# Calculate Sharpe from returns
sharpe = returns_array.mean() / (returns_array.std() + 1e-8) * np.sqrt(252)
```

#### C. Added Trade Frequency Metric
```python
trade_frequency = total_trades / len(actions)
console.print(f"  Trade Frequency: {trade_frequency:.1%}")
```

#### D. Enhanced Debugging
```python
console.print(f"[yellow]Debug: Reward statistics[/yellow]")
console.print(f"  Mean reward: {np.mean(rewards):.4f}")
console.print(f"  Std reward: {np.std(rewards):.4f}")
console.print(f"  Min/Max reward: {np.min(rewards):.4f} / {np.max(rewards):.4f}")
```

---

### 4. ✅ Added Debugging for std_reward=0 Investigation

**File**: `src/agents/sb3_agents.py:237-272`

**Added**:
```python
# Debugging: Print individual episode rewards to investigate std_reward=0
logger.info(f"DEBUG - Individual episode rewards: {episode_rewards}")
logger.info(f"DEBUG - Individual episode lengths: {episode_lengths}")
logger.info(f"DEBUG - Reward unique values: {np.unique(episode_rewards)}")

# Additional debugging metrics
metrics["min_reward"] = float(np.min(episode_rewards))
metrics["max_reward"] = float(np.max(episode_rewards))
```

**Purpose**:
- See individual episode rewards (will show if all identical)
- Identify if deterministic behavior causing std=0
- Check for any variance in episode outcomes

---

## Expected Changes in Next Run

### Before (Previous Run)

| Metric | Training Eval | Test (Broken) | Issue |
|--------|---------------|---------------|-------|
| sharpe | -6.99 | +2.92 | Invalid calculation |
| mean_reward | -310 | +61.13 | Obs normalization mismatch |
| std_reward | N/A | 0.0 | All episodes identical (suspicious) |

### After (Next Run - Expected)

| Metric | Training Eval | Test (Fixed) | Status |
|--------|---------------|--------------|--------|
| sharpe | -6.99 | ~-6 to -8 | ✅ Comparable (same calculation) |
| mean_reward | -310 | ~-200 to -400 | ✅ Comparable (same normalization) |
| std_reward | N/A | > 0 | ✅ Should have variance now |
| trade_frequency | N/A | Will show % | ✅ New metric |

**Key Point**: Test performance should now be **comparable** to validation performance, not better!

---

## What We'll Learn from Debug Output

### If std_reward still = 0:
```
DEBUG - Individual episode rewards: [61.13, 61.13, 61.13, ...]
DEBUG - Reward unique values: [61.13]
```
**Meaning**: Episodes are deterministic - agent always gets same reward
**Possible causes**:
- Environment always resets to same state
- Agent's deterministic policy always takes same actions
- Test data has no variance

### If std_reward > 0:
```
DEBUG - Individual episode rewards: [50.2, 72.1, 45.8, ...]
DEBUG - Reward unique values: [45.8, 50.2, 55.3, 72.1, ...]
```
**Meaning**: Episodes have variance - this is expected!
**Previous std=0 was likely a bug or measurement artifact**

---

## How Observations Are Now Normalized

### Training

```
Raw Obs → DummyVecEnv → VecNormalize → Agent
                         ↓
                    Learns stats:
                    - obs_mean = 12.5
                    - obs_std = 3.2
```

### Test (NEW - Fixed)

```
Raw Obs → DummyVecEnv → VecNormalize → Agent
                         ↓
                    Uses SAME stats:
                    - obs_mean = 12.5  (copied from training!)
                    - obs_std = 3.2    (copied from training!)
```

**Result**: Agent sees observations on same scale in both training and test!

---

## Debugging Workflow

### Step 1: Run Training
```bash
python experiments/train.py training.total_timesteps=100000
```

### Step 2: Check Console Output

Look for these new messages:

```
[yellow]Wrapping test environment with VecNormalize...[/yellow]
[green]✓[/green] Copied observation normalization stats from training

Evaluating Final Performance...
[yellow]Debug: Episode rewards stats[/yellow]
  Mean: -250.34
  Std: 45.67  ← Should NOT be 0!

Running Backtest...
[yellow]Wrapping test environment with VecNormalize...[/yellow]
[green]✓[/green] Backtest completed
  Steps: 1090
  Final Return: -25.3%
  Sharpe Ratio (from returns): -6.45  ← Should match eval sharpe!
  Total Trades: 1090
  Trade Frequency: 100.0%  ← Still overtrading!

[yellow]Debug: Reward statistics[/yellow]
  Mean reward: 0.0234
  Std reward: 0.156  ← Should have variance
  Min/Max reward: -2.34 / 3.12
```

### Step 3: Check Logs

In the log file, look for:
```
[INFO] DEBUG - Individual episode rewards: [array([-250.34]), array([-220.12]), ...]
[INFO] DEBUG - Individual episode lengths: [array([1090]), array([1090]), ...]
[INFO] DEBUG - Reward unique values: [-250.34 -220.12 -280.45 ...]
```

---

## Next Steps

1. **Run test**:
   ```bash
   python experiments/train.py training.total_timesteps=100000
   ```

2. **Verify fixes**:
   - ✅ Test Sharpe should match validation Sharpe (~-7)
   - ✅ std_reward should be > 0
   - ✅ Observations properly normalized (see console message)
   - ✅ Sharpe calculated from returns, not Sharpe-of-Sharpe

3. **Address overtrading** (if trade_frequency = 100%):
   - Add transaction cost penalty to reward
   - See `docs/debugging/sharpe_reward_analysis.md` for solutions

4. **Improve reward function** (if performance still poor):
   - Switch to returns-based reward
   - Or implement hybrid risk-adjusted reward
   - See solutions in `sharpe_reward_analysis.md`

---

## Files Modified

1. `experiments/train.py`:
   - Added `wrap_test_env_for_evaluation()` function
   - Updated `evaluate_final_performance()`
   - Completely rewrote `run_backtest()`

2. `src/agents/sb3_agents.py`:
   - Added debugging to `evaluate()` method
   - Added min/max reward metrics

---

## Rollback Instructions

If these changes cause issues:

```bash
git diff experiments/train.py
git checkout -- experiments/train.py
git checkout -- src/agents/sb3_agents.py
```

Or manually remove:
- `wrap_test_env_for_evaluation()` function
- Debug print statements
- Revert to using raw `test_env` instead of `test_env_wrapped`

---

## References

- VecNormalize docs: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
- Related analysis: `docs/debugging/train_eval_disparity.md`
- Reward fixes: `docs/debugging/sharpe_reward_analysis.md`

**Last Updated**: 2025-10-27