# HOW IT WORKS - Visual Explanation

## The Key Concept

**ENVIRONMENT and AGENT are SEPARATE.**

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR DATA FILE                          │
│                     (All 14 columns exist)                      │
├─────────────────────────────────────────────────────────────────┤
│ close │ open │ high │ low │ volume │ ratio_sma_5_close_zscore │ ... │
├─────────────────────────────────────────────────────────────────┤
│ 50000 │ 49950│ 50100│ 49900│  1000  │         1.23             │ ... │
└─────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┴───────────────┐
                │                              │
                ↓                              ↓
┌───────────────────────────────┐  ┌───────────────────────────────┐
│       ENVIRONMENT             │  │        AGENT                  │
│                               │  │                               │
│ • Gets FULL dataframe         │  │ • Gets ONLY input_features    │
│   (ALL 14 columns)            │  │   (specified in observation/) │
│                               │  │                               │
│ • Uses env.price_column       │  │ • Makes decisions based on    │
│   for trading                 │  │   ONLY these features         │
│   (default: "close")          │  │                               │
│                               │  │ Example observation:          │
│ • Calculates P&L              │  │   [ratio_sma_5_close_zscore,  │
│ • Executes trades             │  │    ratio_sma_20_close_zscore, │
│ • Has access to ANY column    │  │    ratio_range_close_zscore,  │
│                               │  │    fracdiff_0.4_zscore]       │
└───────────────────────────────┘  └───────────────────────────────┘
```

---

## Real Example from Your Config

### Your Current Setup:

```yaml
# configs/observation/default.yaml
input_features:
  # - "close"     # COMMENTED OUT
  # - "volume"    # COMMENTED OUT
  - "ratio_sma_5_close_zscore"
  - "ratio_sma_20_close_zscore"
  - "ratio_range_close_zscore"
  - "fracdiff_0.4_zscore"
```

```yaml
# configs/env/default.yaml
price_column: "close"  # Environment uses this for trading
```

### What Happens:

```
AGENT:
  Sees: [ratio_sma_5_close_zscore, ratio_sma_20_close_zscore,
         ratio_range_close_zscore, fracdiff_0.4_zscore]

  Does NOT see: close, volume, open, high, low

  Makes trading decisions based ONLY on the 4 indicators

ENVIRONMENT:
  Has: ALL 14 columns (close, open, high, low, volume, all indicators, timestamp)

  Uses: close for trade execution and P&L calculation

  Can access: ANY column from the dataframe
```

### Result:

✅ **Agent makes decisions using ONLY indicators**

✅ **Environment trades using close price**

✅ **They are SEPARATE and both work perfectly**

---

## Why This Matters

### Traditional (Bad) Approach:

```
Agent observation = Everything the environment needs
```

Problem: Agent sees too much, training is harder

### Our (Good) Approach:

```
Agent observation = What agent needs to make decisions
Environment data = What environment needs to function
```

Benefit:
- Agent sees only relevant features (faster training, better generalization)
- Environment has all data it needs (accurate P&L, flexible trading)

---

## Test It Yourself

```bash
# Run with your current config (indicators only for agent)
uv run python experiments/train.py

# The agent will observe ONLY the 4 indicators
# The environment will trade using "close" price
# And it will work perfectly
```

---

## The Three Configs Explained

### 1. `observation/default.yaml` - Agent's Brain

```yaml
input_features:  # What the AGENT sees
  - "ratio_sma_5_close_zscore"
  - "ratio_sma_20_close_zscore"
  # ... indicators only
```

**Purpose**: Define agent observation space

**Used by**: Agent's neural network input

**Does NOT affect**: Environment's data access

### 2. `env/default.yaml` - Environment's Needs

```yaml
price_column: "close"        # What price to use for trading
required_columns:            # What must exist in data
  - "close"
  - "timestamp"
```

**Purpose**: Define environment requirements

**Used by**: Environment for trading, P&L, validation

**Does NOT affect**: Agent's observation space

### 3. `feature_engineering/none.yaml` - Data Transformation

```yaml
enabled: false  # Don't create new columns at runtime
```

**Purpose**: Whether to create new features (returns, log_returns)

**Used by**: Data loading pipeline

**Affects**: What columns exist in the dataframe

---

## Complete Flow

```
1. Load Data
   └→ DataFrame with all columns from parquet file

2. Feature Engineering (if enabled)
   └→ Add new columns (returns_1, log_returns, etc.)

3. Validate
   ├→ Check env.required_columns exist
   └→ Check observation.input_features exist

4. Create Environment
   ├→ Environment receives: Full dataframe (ALL columns)
   └→ Environment uses: env.price_column for trading

5. Create Agent
   ├→ Agent receives: observation.input_features subset
   └→ Agent makes decisions: Based on observed features only

6. Training Loop
   ├→ Agent observes: input_features
   ├→ Agent decides: Buy/Sell/Hold based on indicators
   ├→ Environment executes: Trade using price_column
   └→ Environment calculates: Reward based on P&L
```

---

## FAQ

**Q: If agent doesn't observe close price, how does it know the price?**

A: **It doesn't need to!** Agent learns patterns in the indicators. The indicators already contain price information (they're derived from price). Agent makes decisions based on indicator patterns, not raw price.

**Q: Can agent observe close AND indicators?**

A: **Yes!** Just add close to input_features:
```yaml
input_features:
  - "close"  # Add this
  - "ratio_sma_5_close_zscore"
  - "ratio_sma_20_close_zscore"
```

**Q: Can environment use different price column?**

A: **Yes!** Just change price_column:
```yaml
# env/default.yaml
price_column: "vwap"  # Use VWAP instead of close
```

**Q: Does this affect P&L calculation?**

A: **No!** Environment always calculates P&L correctly using price_column, regardless of what agent observes.

---

## Summary

- **observation.input_features** → What agent sees (agent's brain input)
- **env.price_column** → What environment uses for trading
- **env.required_columns** → What must exist in data
- **feature_engineering.enabled** → Whether to create new columns

**They are SEPARATE. They don't affect each other.**

**Your config is correct. It works.**
