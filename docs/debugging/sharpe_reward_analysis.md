# Sharpe Ratio Reward Problem Analysis

**Date**: 2025-10-27
**Run**: `PPO_sharpe_20251027_073352`
**Status**: 🔴 Critical Issue Identified

## Executive Summary

Training shows **explained variance increasing** (value function learning successfully) but **Sharpe ratio stuck at -8** (agent losing money consistently). This indicates a fundamental disconnect between what the agent optimizes and actual trading performance.

## Symptoms

| Metric | Observed | Expected | Status |
|--------|----------|----------|--------|
| `explained_variance` | ↑ Increasing steadily | > 0.5 | ✅ Good |
| `value_loss` | Stabilizing | < 1000 | ✅ Good |
| `sharpe` | -8 (stuck) | > 0 | 🔴 **CRITICAL** |
| `entropy_loss` | > 0 | > 0 | ✅ Good |
| Trading performance | Consistent losses | Profitable | 🔴 **CRITICAL** |

**Interpretation**: The agent is successfully learning to predict values, but optimizing for the **wrong objective**.

---

## Root Cause Analysis

### Problem 1: Rolling Window Sharpe vs Episode Sharpe ⚠️ CRITICAL

**What the agent optimizes** (`src/environment/trading_env.py:287-292`):
```python
elif self.reward_type == "sharpe":
    # Simple Sharpe approximation
    if len(self.history['returns']) > 1:
        returns_array = np.array(self.history['returns'][-20:])  # Last 20 returns
        reward = returns_array.mean() / (returns_array.std() + 1e-8)
    else:
        reward = returns
```

**What you measure** (`src/environment/trading_env.py:330`):
```python
info['sharpe'] = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)
```

#### The Problem

| Aspect | Agent's View (Reward) | Your View (Metric) |
|--------|----------------------|-------------------|
| **Window** | Last 20 steps (sliding) | Full episode |
| **Time horizon** | Short-term | Long-term |
| **Annualization** | None | × √252 |
| **Signal** | Non-stationary | Stationary |

**Example Scenario**:
```
Step 1-1000:  Poor trades, Sharpe = -2.0
Step 1001-1020: Good trades, Sharpe = +1.5  ← Agent sees this!
Episode Sharpe: -1.8                        ← You see this!
```

The agent optimizes the rolling 20-step window, **completely ignoring** long-term consequences.

#### Why This Causes Sharpe = -8

1. Agent focuses on short-term patterns
2. Ignores transaction costs compounding over episode
3. May overfit to recent noise
4. No incentive to maintain consistency

---

### Problem 2: VecNormalize Reward Masking ⚠️ CRITICAL

**Implementation** (`src/agents/sb3_agents.py:76-82`):
```python
self.env = VecNormalize(
    self.env,
    norm_obs=True,
    norm_reward=True,   # ← Normalizes rewards!
    clip_obs=10.0,
    clip_reward=10.0,
    gamma=0.99,
)
```

#### How Reward Normalization Works

VecNormalize maintains running statistics:
```python
normalized_reward = (reward - mean) / (std + eps)
```

**Example**:
| Step | True Sharpe | Running Mean | Running Std | Normalized Reward |
|------|-------------|--------------|-------------|-------------------|
| 100 | -0.5 | -0.2 | 0.8 | -0.375 |
| 200 | -0.8 | -0.3 | 0.9 | -0.556 |
| 300 | +0.2 | -0.25 | 0.85 | +0.529 |

#### The Problem

1. **Agent never sees true Sharpe values**
2. **Normalized rewards can flip sign**: Bad Sharpe → positive reward if worse than average
3. **Non-stationary**: Mean/std change during training
4. **Disconnects reward from goal**: Agent optimizes normalized signal, not actual Sharpe

This is why explained variance improves (predicting normalized rewards) while true Sharpe stays terrible.

---

### Problem 3: Non-Stationary Reward Signal ⚠️ HIGH

The rolling 20-step Sharpe calculation creates a **non-stationary** reward signal:

```python
# Step 100: Uses returns[80:100]
# Step 101: Uses returns[81:101]  ← Different distribution!
```

**Impact**:
- Reward for same action changes based on history
- Violates Markov property
- Makes value function approximation harder
- Creates spurious correlations

**Example**:
```
Action: Buy
Context 1: After 19 gains  → Sharpe reward ≈ +2.0 (high mean)
Context 2: After 19 losses → Sharpe reward ≈ -1.5 (low mean)
Same action, wildly different rewards!
```

---

### Problem 4: Transaction Cost Bleeding ⚠️ HIGH

**Costs** (`src/environment/trading_env.py:227-231, 254-255`):
```python
# Entry
execution_price = current_price * (1 + self.slippage_rate * np.sign(signal))
commission = trade_value * self.commission_rate  # 0.1%

# Exit
execution_price = current_price * (1 - self.slippage_rate * np.sign(self.position.size))
commission = trade_value * self.commission_rate  # 0.1%
```

**Total Round-Trip Cost**:
- Commission: 0.1% × 2 = **0.2%**
- Slippage: 0.05% × 2 = **0.1%**
- **Total: 0.3% per round trip**

**Why This Matters**:

If agent trades every 10 steps in a 7,770-step episode:
- Trades: 777 round trips
- Cost: 777 × 0.3% = **233% of capital**
- Sharpe = -8 suggests **massive overtrading**

The reward signal doesn't explicitly penalize trading frequency, so agent may trade excessively.

---

### Problem 5: Position Sizing Inconsistency ⚠️ MEDIUM

**Current Implementation** (`src/environment/trading_env.py:223`):
```python
max_position_value = self.balance * self.max_position_pct
```

**Issue**: `self.balance` is cash only, doesn't account for:
- Capital tied up in existing positions
- Unrealized P&L
- Could lead to overleveraging if position is profitable

**Better approach**:
```python
max_position_value = self._get_portfolio_value() * self.max_position_pct
```

---

## Why Explained Variance Increases

**Explained Variance** measures: _"How well does the value function predict returns?"_

```python
explained_var = 1 - Var(returns - value_pred) / Var(returns)
```

**Increasing explained variance means**:
- ✅ Value function accurately predicts **normalized rolling Sharpe rewards**
- ✅ Neural network is learning patterns
- ✅ Optimization is working

**BUT**:
- ❌ It's predicting the **wrong target** (normalized, rolling)
- ❌ Good at predicting ≠ Good at trading
- ❌ "Successfully" learning a bad policy

**Analogy**: Student perfectly memorizes wrong answers → high test confidence, failing grade.

---

## Solution Approaches

### Approach 1: Simple Returns Reward ⭐ Recommended for Quick Fix

**Change**:
```yaml
# configs/env/default.yaml
reward_type: returns  # Instead of "sharpe"
```

**Pros**:
- ✅ Stationary reward signal
- ✅ No normalization artifacts
- ✅ Clear, interpretable signal
- ✅ One line change
- ✅ Can still measure Sharpe separately

**Cons**:
- ❌ Doesn't explicitly optimize Sharpe
- ❌ Doesn't penalize volatility
- ❌ May learn risky behavior

**Code Changes**: None needed, just config!

---

### Approach 2: Fixed Sharpe Reward

**Changes Required**:

1. **Use full-episode Sharpe** (`src/environment/trading_env.py:278`):
```python
elif self.reward_type == "sharpe":
    # Use full episode history, not rolling window
    if len(self.history['returns']) > 1:
        returns_array = np.array(self.history['returns'])  # All returns
        reward = returns_array.mean() / (returns_array.std() + 1e-8)
    else:
        reward = returns
```

2. **Disable reward normalization** (`src/agents/sb3_agents.py:78`):
```python
self.env = VecNormalize(
    self.env,
    norm_obs=True,
    norm_reward=False,  # ← Changed to False
    clip_obs=10.0,
    clip_reward=10.0,
)
```

**Pros**:
- ✅ Aligns reward with goal
- ✅ Agent sees true Sharpe values
- ✅ Directly optimizes what you measure

**Cons**:
- ❌ Still non-stationary (but less)
- ❌ Reward changes even if action unchanged
- ❌ May be slow to learn (long horizon)

---

### Approach 3: Hybrid Reward (Risk-Adjusted Returns) ⭐ Recommended for Best Performance

**Implementation**:

Add new reward type (`src/environment/trading_env.py:278`):
```python
elif self.reward_type == "risk_adjusted":
    returns = (current_value - prev_value) / prev_value

    # Base reward: returns
    reward = returns

    # Penalty for volatility
    if len(self.history['returns']) > 10:
        returns_array = np.array(self.history['returns'][-50:])
        volatility_penalty = returns_array.std() * 0.5
        reward -= volatility_penalty

    # Penalty for trading (transaction costs)
    if action != 0:  # If not holding
        reward -= 0.003  # 0.3% cost

    # Clip to reasonable range
    reward = np.clip(reward, -1.0, 1.0)
```

**Config**:
```yaml
# configs/env/default.yaml
reward_type: risk_adjusted
```

**Pros**:
- ✅ Stationary base reward (returns)
- ✅ Explicitly penalizes volatility
- ✅ Accounts for transaction costs
- ✅ More stable learning signal
- ✅ Empirically works well

**Cons**:
- ❌ Requires code changes
- ❌ Hyperparameter tuning (penalty weights)
- ❌ More complex

---

## Recommended Action Plan

### Phase 1: Quick Test (1 hour)

1. **Test with simple returns**:
   ```bash
   # configs/env/default.yaml
   reward_type: returns
   ```

2. **Run short training** (10k steps):
   ```bash
   python experiments/train.py training.total_timesteps=10000
   ```

3. **Check if Sharpe improves**:
   - If Sharpe > 0: Continue training
   - If Sharpe still < 0: Environment or data issue

### Phase 2: Implement Hybrid Reward (2-3 hours)

1. Add `risk_adjusted` reward type to `trading_env.py`
2. Test with multiple penalty weight combinations
3. Monitor both Sharpe and raw returns

### Phase 3: Advanced Tuning (if needed)

1. Experiment with entropy coefficient (`ent_coef`)
2. Adjust clipping ranges
3. Try different network architectures

---

## Key Insights

1. **VecNormalize is double-edged**: Great for stability, hides true rewards
2. **Rolling Sharpe ≠ Episode Sharpe**: Optimize what you measure
3. **Transaction costs matter**: Explicitly include in reward
4. **Explained variance misleading**: Can predict wrong thing perfectly
5. **Sharpe is hard to optimize**: Consider simpler reward first

---

## References

- **VecNormalize docs**: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
- **Reward engineering**: Ng et al. "Policy Invariance Under Reward Transformations"
- **RL for trading**: Mosavi & Vaezipour "Reinforcement Learning in Stock Trading"

---

## Next Steps

- [ ] Decide on approach (1, 2, or 3)
- [ ] Implement chosen solution
- [ ] Run ablation study (with/without VecNormalize)
- [ ] Monitor Sharpe ratio during training
- [ ] Document findings

**Last Updated**: 2025-10-27
