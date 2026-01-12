# Breaking the 0.0186 Reward Plateau

## 📊 The Problem

Your DQN agent consistently achieves ~0.0186 mean reward and refuses to improve further.

### Root Causes:

1. **Premature Exploration Decay**
   - Exploration ends at step 29,092 (10% of training)
   - For 90% of training, agent only explores 5% of the time
   - Found a "safe" strategy early and stopped looking for better ones

2. **Local Optimum**
   - Agent learned: "holding/safe trading = small consistent gains"
   - Never explored: "aggressive trading = potential for higher gains"
   - Result: Better than random (0.0186 vs ~0.001), but not optimal

3. **Limited Market Opportunity**
   - Data has mean step return of 0.0007% (nearly flat)
   - Max single-step move: ±0.16%
   - Agent needs to compound small edges over many steps

---

## 🚀 Solutions (Ranked by Impact)

### **Solution 1: Fix Exploration Schedule** ⭐⭐⭐⭐⭐
**Impact: HIGH | Effort: LOW**

The #1 problem is your agent stopped exploring too early.

**What to do:**
```bash
# Use the new aggressive exploration config
python experiments/train.py agent=dqn_aggressive
```

**Key changes in `configs/agent/dqn_aggressive.yaml`:**
- `exploration_fraction: 0.3` (was 0.1) - Explore for 30% of training
- `exploration_final_eps: 0.2` (was 0.05) - Keep exploring 20% forever
- `total_timesteps: 500_000` (was 200k) - More time to learn
- Larger network: `[512, 512]` - More capacity for complex strategies

**Expected result:** Agent continues exploring and finds better policies

---

### **Solution 2: Modify Reward Structure** ⭐⭐⭐⭐
**Impact: HIGH | Effort: MEDIUM**

Current reward (returns) might not incentivize profitable trading enough.

**Option A: Try Sharpe Ratio Rewards**
```bash
python experiments/train.py agent=dqn_aggressive env=aggressive_rewards env.environment_params.reward_type=sharpe
```

**Why:** Sharpe ratio rewards consistent profitable trades, not just any positive return.

**Option B: Add Reward Shaping**
Modify `trading_env.py:387` to add bonuses:
```python
def _calculate_reward(self, prev_value: float, current_value: float) -> float:
    returns = (current_value - prev_value) / prev_value

    # Base reward
    reward = returns

    # BONUS: Reward for taking positions (not just holding)
    if abs(self.position.size) > 0.01:
        reward += 0.0001  # Small bonus for being in market

    # BONUS: Extra reward for profitable trades
    if returns > 0.001:  # More than 0.1% gain
        reward += returns * 0.5  # 50% bonus for good trades

    return np.clip(reward, -10.0, 10.0)
```

---

### **Solution 3: Use PPO Instead of DQN** ⭐⭐⭐⭐
**Impact: MEDIUM-HIGH | Effort: LOW**

DQN can get stuck in local optima. PPO explores more naturally.

```bash
python experiments/train.py agent=ppo
```

**Why PPO is better for trading:**
- Continuous exploration through stochastic policy
- Better at handling sparse rewards
- More stable training (trust region updates)

---

### **Solution 4: Relax Risk Constraints** ⭐⭐⭐
**Impact: MEDIUM | Effort: LOW**

Your environment terminates episodes too aggressively, preventing exploration of risky strategies.

**Edit `src/rl_trading_lab/environment/trading_env.py:409`:**

```python
def _is_terminated(self) -> bool:
    """Check if episode should terminate"""
    # BEFORE: Terminate if balance too low
    # if self.balance < self.initial_balance * 0.2:  # Lost 80%
    #     return True

    # AFTER: More lenient
    if self.balance < self.initial_balance * 0.1:  # Lost 90%
        return True

    # BEFORE: 30% drawdown
    # if drawdown > 0.3:  # 30% drawdown
    #     return True

    # AFTER: 50% drawdown
    if drawdown > 0.5:  # 50% drawdown
        return True

    return False
```

**Why:** Allows agent to learn from riskier strategies without premature termination.

---

### **Solution 5: Feature Engineering** ⭐⭐⭐
**Impact: MEDIUM | Effort: MEDIUM**

Agent might not have enough information to make better decisions.

**Add predictive features:**
```bash
# Edit configs/feature_engineering/full.yaml
# Or create new feature engineering config with:
# - Volatility indicators (ATR, Bollinger Bands)
# - Momentum indicators (RSI, MACD, Stochastic)
# - Volume indicators (OBV, VWAP divergence)
# - Market microstructure (bid-ask spread, order imbalance)

python experiments/train.py \
  agent=dqn_aggressive \
  feature_engineering=full \
  observation=with_returns
```

---

### **Solution 6: Curriculum Learning** ⭐⭐
**Impact: MEDIUM | Effort: HIGH**

Train agent on progressively harder tasks.

**Curriculum:**
1. Phase 1: Train on high-volatility periods (easier to profit)
2. Phase 2: Train on mixed volatility
3. Phase 3: Train on all data

**Implementation:**
```python
# Filter data by volatility
high_vol_data = df[df['close'].rolling(20).std() > threshold]
```

---

### **Solution 7: Ensemble & Risk-Seeking Agents** ⭐⭐
**Impact: LOW-MEDIUM | Effort: HIGH**

Train multiple agents with different risk profiles and combine them.

```bash
# Train conservative agent (current)
python experiments/train.py agent=dqn

# Train aggressive agent
python experiments/train.py agent=dqn_aggressive env=aggressive_rewards

# Combine strategies based on market conditions
```

---

## 🎯 Recommended Action Plan

### **Week 1: Quick Wins**
1. ✅ **Train with better exploration** (Solution 1)
   ```bash
   python experiments/train.py agent=dqn_aggressive
   ```

2. ✅ **Try PPO** (Solution 3)
   ```bash
   python experiments/train.py agent=ppo training.total_timesteps=500000
   ```

### **Week 2: Reward Engineering**
3. ✅ **Test Sharpe rewards** (Solution 2A)
   ```bash
   python experiments/train.py \
     agent=dqn_aggressive \
     env.environment_params.reward_type=sharpe
   ```

4. ✅ **Relax risk constraints** (Solution 4)
   - Edit trading_env.py as shown above
   - Retrain with new constraints

### **Week 3: Advanced Optimization**
5. ✅ **Feature engineering** (Solution 5)
6. ⏭️  **Curriculum learning** (if time permits)

---

## 📈 Expected Results

| Solution | Expected Reward | Expected Return/Episode |
|----------|-----------------|-------------------------|
| Current (baseline) | 0.0186 | ~1.86% |
| Better exploration | 0.03-0.05 | 3-5% |
| PPO | 0.04-0.07 | 4-7% |
| Sharpe rewards | 0.03-0.06 | 3-6% |
| All combined | 0.08+ | 8%+ |

---

## 🔍 How to Monitor Progress

**During training, watch for:**
1. **Episode rewards increasing** beyond 0.02
2. **Trade frequency** should be reasonable (not all holds, not all trades)
3. **Sharpe ratio** improving (in MLflow metrics)

**In MLflow UI:**
```bash
mlflow ui
# Navigate to: http://localhost:5000
# Compare: eval/mean_reward across experiments
```

**Key metrics to track:**
- `eval/mean_reward` - Should increase beyond 0.0186
- `eval/sharpe` - Should be positive and growing
- `train/exploration_rate` - Should stay higher longer
- `backtest/final_return` - Ultimate test of performance

---

## 🐛 Debugging

If agent still plateaus:

1. **Check action distribution**
   ```bash
   python analyze_agent_behavior.py
   ```
   - Should NOT be >70% hold actions
   - Should see reasonable mix of buy/sell/hold

2. **Inspect episode trajectories**
   - Are trades profitable?
   - Is agent entering/exiting at good times?
   - Plot episode balances to visualize performance

3. **Verify data quality**
   - Check for look-ahead bias
   - Ensure features are properly normalized
   - Validate no NaN/inf values

---

## 💡 Pro Tips

1. **Start simple:** Test one solution at a time to isolate impact
2. **Monitor closely:** Use MLflow to track every experiment
3. **Be patient:** RL training is noisy - run multiple seeds
4. **Validate carefully:** Test on held-out data before trusting results
5. **Document everything:** Keep notes on what works and what doesn't

---

## 🚨 Important Warnings

1. **Overfitting risk:** More exploration ≠ always better
   - Can lead to overfitting on training data
   - Always validate on test set

2. **Survivorship bias:** Relaxing risk limits can hide bad strategies
   - Agent might take huge risks for short-term gains
   - Monitor max drawdown carefully

3. **Reality check:** 8%+ per episode might not be realistic
   - Crypto markets are efficient
   - Be skeptical of "too good to be true" results

---

Good luck breaking that plateau! 🚀
