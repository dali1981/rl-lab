# Training vs Evaluation Performance Disparity Analysis

**Date**: 2025-10-27
**Run**: `PPO_sharpe_20251027_073352`
**Status**: 🟡 Measurement Inconsistency (Not Model Issue)

## Observed Disparity

| Metric | Training Eval (`eval/`) | Final Test/Backtest | Difference | Status |
|--------|------------------------|---------------------|------------|--------|
| **mean_reward** | -310 | +61.13 | 🔴 371 point gap | Huge! |
| **sharpe** | -6.99 | +2.92 | 🔴 9.91 point gap | Massive! |
| **total_return** | -30.2% | +1.43% | 🔴 31.6% gap | Critical! |
| **episode_length** | 1,400 | 1,090 | Different data? | Normal |

**Initial Impression**: Agent performs terribly during training but excellently on test?
**Reality**: **Measurement inconsistency**, not actual performance difference!

---

## Root Cause: VecNormalize Wrapper Mismatch

### The Code Flow

#### 1. Training Setup (`src/agents/sb3_agents.py:70-97`)

```python
# Training environment
self.env = DummyVecEnv([train_env_func])
self.env = VecNormalize(
    self.env,
    norm_obs=True,
    norm_reward=True,  # ← Normalizes rewards!
    clip_reward=10.0,
    gamma=0.99,
)

# Evaluation environment (used during training)
if eval_env is not None:
    self.eval_env = DummyVecEnv([eval_env_func])
    self.eval_env = VecNormalize(
        self.eval_env,
        norm_obs=True,
        norm_reward=False,  # ← Raw rewards
        training=False,
    )
```

#### 2. Training Evaluation (`experiments/train.py:146-152`)

```python
# Uses self.eval_env (wrapped with VecNormalize)
metrics = agent.train(
    eval_freq=cfg.training.eval_freq,
    n_eval_episodes=cfg.training.n_eval_episodes,
)
```

**Result**: `eval/` metrics use VecNormalize wrapper (but with `norm_reward=False`)

#### 3. Final Evaluation (`experiments/train.py:163-167`)

```python
# Uses raw test_env (NO VecNormalize wrapper!)
metrics = agent.evaluate(
    env=test_env,  # ← Raw environment!
    n_episodes=n_episodes,
    deterministic=True,
)
```

**Result**: Final test metrics use **completely different environment setup**

#### 4. Backtest (`experiments/train.py:189-234`)

```python
# Also uses raw test_env
obs, _ = test_env.reset()
while not done:
    action, _ = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
```

**Result**: Backtest also uses raw environment

---

## The Critical Difference

### During Training (eval/ metrics)

```
TradingEnv (eval_env)
    ↓
DummyVecEnv (vectorize)
    ↓
VecNormalize (norm_obs=True, norm_reward=False, training=False)
    ↓
Agent observes: Normalized observations
    ↓
Reported rewards: Raw (but from different data split)
```

**Wait!** Even though `norm_reward=False`, there's still a problem...

### During Test/Backtest

```
TradingEnv (test_env)
    ↓
NO WRAPPING
    ↓
Agent observes: Raw observations (MISMATCH!)
    ↓
Reported rewards: Raw
```

---

## The Real Problems

### Problem 1: Observation Normalization Mismatch 🔴 CRITICAL

**Training**:
- Agent learns with **normalized observations** from VecNormalize
- VecNormalize maintains running mean/std of observations
- All observations scaled to ~N(0, 1)

**Test/Backtest**:
- Agent receives **raw, unnormalized observations**
- Observations have completely different scale
- Example: Feature value 100.0 → normalized to 0.5 during training
- But during test, agent sees 100.0 directly!

**This is like training a model in meters, then testing in kilometers!**

### Problem 2: Different Data Splits

| Split | Used In | Episode Length | Performance |
|-------|---------|----------------|-------------|
| **Validation** | `eval/` during training | 1,400 steps | Sharpe -6.99 |
| **Test** | Final eval + backtest | 1,090 steps | Sharpe +2.92 |

The validation and test data are from **different time periods**:
- Validation: 20% of data (likely middle chronological period)
- Test: 10% of data (likely end chronological period)

**Market conditions could be completely different!**

### Problem 3: Rewards May Still Be Normalized in Training

Even though `norm_reward=False` for eval_env, during actual training:
- `self.env` (training) uses `norm_reward=True`
- Agent is trained on normalized rewards
- Value function predicts normalized returns
- During eval, raw rewards might not align with value predictions

---

## Why This Explains Everything

### The -310 vs +61.13 Reward Mystery

**Hypothesis 1: VecNormalize is normalizing despite norm_reward=False**
- Possible bug or edge case
- Need to verify with debugging

**Hypothesis 2: Different reward calculation**
- Training eval uses rolling Sharpe (from environment)
- Backtest calculates Sharpe differently (line 213)

Let's check the backtest Sharpe calculation:
```python
# experiments/train.py:213
sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)
```

This is **NOT** the environment's Sharpe reward! This is calculated from the collected rewards (which are Sharpe values from the environment).

**So we're taking the Sharpe of Sharpe rewards!** This is nonsensical.

### The Sharpe -6.99 vs +2.92 Mystery

**Training `eval/sharpe: -6.99`**:
- Comes from `info['sharpe']` in the environment
- Calculated from actual returns during the episode
- Represents true trading performance
- On validation data (different market regime)

**Backtest `sharpe: 2.92`**:
- Calculated as: `mean(sharpe_rewards) / std(sharpe_rewards) * sqrt(252)`
- This is **Sharpe of Sharpe ratios** (meaningless!)
- Not comparable to training sharpe
- On test data (different market regime)

---

## The Actual Performance

Let's look at what we can trust:

### Training Evaluation (eval/)

```
sharpe: -6.99
total_return: -30.2%
max_drawdown: 30.1%
```

**Interpretation**: Agent performs **terribly** on validation data.

### Test Evaluation

```
mean_reward: 61.13  (this is mean of Sharpe rewards, ~0.61 average Sharpe per step)
std_reward: 0.0     (🚨 RED FLAG - all episodes identical?)
episode_length: 1090
```

**The std_reward = 0.0 is suspicious!** All 10 test episodes gave identical rewards?

### Backtest

```
Final Return: 1.43%
Sharpe: 2.92  (meaningless - Sharpe of Sharpe rewards)
Total Trades: 1090
```

**1,090 trades in 1,090 steps = trading EVERY step!**
This is the overtrading problem identified earlier.

---

## The Truth

Based on this analysis, here's what's **actually** happening:

1. ✅ **Agent learned something** (explained_variance = 0.853)
2. ❌ **Performs poorly on validation data** (Sharpe -6.99, return -30.2%)
3. ⚠️ **Test metrics are misleading**:
   - Observation normalization mismatch
   - Different data distribution
   - Sharpe calculation is wrong (Sharpe of Sharpe)
   - 100% trade frequency is concerning
4. ❓ **True test performance unknown** due to measurement issues

---

## Critical Issues to Fix

### 🔴 Priority 1: Observation Normalization Mismatch

**Problem**: Test environment doesn't use VecNormalize, but agent expects normalized obs.

**Fix Options**:

**Option A**: Wrap test_env with VecNormalize
```python
# experiments/train.py:249
test_env_wrapped = DummyVecEnv([lambda: test_env])
test_env_wrapped = VecNormalize(
    test_env_wrapped,
    norm_obs=True,
    norm_reward=False,
    training=False,  # Don't update statistics
)
# Copy normalization statistics from training
test_env_wrapped.obs_rms = agent.env.obs_rms  # Use training stats
```

**Option B**: Remove VecNormalize entirely
```python
# src/agents/sb3_agents.py
# Comment out VecNormalize wrapper
# Test if training still works
```

### 🔴 Priority 2: Fix Backtest Sharpe Calculation

**Problem**: Currently calculates Sharpe of Sharpe rewards (meaningless).

**Fix**:
```python
# experiments/train.py:213
# Instead of:
# sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252)

# Calculate Sharpe from actual returns:
returns = np.array([(balances[i] - balances[i-1]) / balances[i-1]
                    for i in range(1, len(balances))])
sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
```

### 🟡 Priority 3: Investigate std_reward = 0.0

**Problem**: All test episodes have identical rewards (suspicious).

**Debug**:
```python
# Print episode rewards to see what's happening
print(f"Episode rewards: {episode_rewards}")
```

### 🟡 Priority 4: Reduce Overtrading

**Problem**: Agent trades every single step (1,090 trades in 1,090 steps).

**Fix**: Add transaction cost to reward (already mentioned in other doc).

---

## Corrected Evaluation Protocol

### Step 1: Prepare Test Environment Properly

```python
def create_test_env_for_eval(agent, test_env):
    """Wrap test env to match training setup"""
    # Wrap in DummyVecEnv
    test_vec_env = DummyVecEnv([lambda: test_env])

    # Wrap in VecNormalize with training statistics
    test_vec_env = VecNormalize(
        test_vec_env,
        norm_obs=True,
        norm_reward=False,
        training=False,
    )

    # CRITICAL: Copy normalization statistics from training
    test_vec_env.obs_rms = agent.env.obs_rms
    test_vec_env.ret_rms = agent.env.ret_rms

    return test_vec_env
```

### Step 2: Run Evaluation

```python
# Use properly wrapped environment
test_env_wrapped = create_test_env_for_eval(agent, test_env)
metrics = agent.evaluate(test_env_wrapped, n_episodes=10)
```

### Step 3: Run Backtest with Correct Metrics

```python
# Use wrapped environment
# Calculate Sharpe from returns, not from Sharpe rewards
```

---

## Expected Results After Fixes

Once observation normalization is fixed, we expect:

**If model is truly bad**:
- Test Sharpe will match validation Sharpe (~-7)
- Confirms model is losing money

**If model is actually good**:
- Test Sharpe will be positive (>0)
- Current +2.92 might be real (once calculated correctly)

**Most likely**:
- Test will still be poor due to overtrading
- Need to fix reward function (transaction costs)

---

## Action Items

- [ ] **Fix observation normalization mismatch**
  - [ ] Option A: Wrap test env with VecNormalize + copy stats
  - [ ] Option B: Remove VecNormalize entirely and retrain
- [ ] **Fix backtest Sharpe calculation**
  - [ ] Calculate from returns, not Sharpe rewards
- [ ] **Debug std_reward = 0.0**
  - [ ] Print individual episode rewards
  - [ ] Verify environment is resetting properly
- [ ] **Investigate overtrading**
  - [ ] Why 100% trade frequency?
  - [ ] Add transaction cost penalty to reward
- [ ] **Verify data splits**
  - [ ] Check if test data has different characteristics
  - [ ] Consider walk-forward validation
- [ ] **Re-run evaluation properly**
  - [ ] Compare apples-to-apples with same environment setup

---

## Key Learnings

1. **VecNormalize creates measurement complexity**: Hard to compare wrapped vs unwrapped envs
2. **Observation normalization is critical**: Agent can't perform on different scale
3. **Sharpe of Sharpe is meaningless**: Don't nest statistical calculations
4. **std_reward = 0.0 is a red flag**: Something is deterministic or broken
5. **100% trade frequency suggests poor reward signal**: Need explicit cost penalty

---

## References

- VecNormalize docs: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
- Evaluation docs: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#basic-usage-training-saving-loading

**Last Updated**: 2025-10-27
