# Execution Patterns (Optional)

This document covers the live trading integration, provided as a **demonstration of system integration patterns**.

> **Note**: Live trading components exist to demonstrate how trained models can be deployed with proper safety controls. They are not intended for unattended production use.

---

## Architecture

```
Exchange WebSocket
       │
       ▼
┌──────────────┐
│ StreamConsumer│  ← Real-time tick data
└──────────────┘
       │
       ▼
┌──────────────┐
│ BarProcessor │  ← Dollar volume bars
└──────────────┘
       │
       ▼
┌──────────────┐
│FeatureComputer│  ← Technical indicators
└──────────────┘
       │
       ▼
┌──────────────┐
│ InferenceEngine│  ← Model predictions
└──────────────┘
       │
       ▼
┌──────────────┐
│ SafetyGuard  │  ← Risk checks
└──────────────┘
       │
       ▼
┌──────────────┐
│ OrderExecutor│  ← Trade execution
└──────────────┘
```

---

## Safety Controls

### Multi-Layer Protection

| Layer | Control | Purpose |
|-------|---------|---------|
| **Circuit Breaker** | Max drawdown limit | Stop trading at 20% loss |
| **Rate Limiting** | Max trades/hour | Prevent overtrading |
| **Position Limits** | Max capital % | Limit exposure |
| **Loss Limits** | Consecutive losses | Stop after N losses |
| **Balance Check** | Minimum balance | Ensure sufficient capital |

### Configuration

```python
SafetyConfig(
    max_drawdown_pct=0.20,      # Stop at 20% drawdown
    max_trades_per_hour=10,     # Rate limit
    max_consecutive_losses=5,   # Loss limit
    min_balance_usd=100.0,      # Minimum balance
)
```

---

## Feature Consistency

A critical requirement: **live features must match training features exactly**.

The system ensures this by:
1. Loading feature statistics from training checkpoint
2. Using same indicator periods and normalization
3. Validating feature dimensions before inference

```python
# Features computed live match training data
expected_features = config['observation']['input_features']
if live_features != expected_features:
    raise ValueError("Feature mismatch")
```

---

## Deployment Workflow

```
1. Train Model      → Develop and validate strategy
2. Validate         → Test with historical data
3. Paper Trade      → Run on testnet (no real money)
4. Monitor          → Observe behavior for 24-48 hours
5. Small Scale      → Deploy with minimal capital
6. Scale Up         → Gradually increase exposure
```

---

## Validation

Before any live deployment, use the `Pre-live validation surface` command from [docs/commands.md](commands.md).

This verifies:
- Data loading works
- Features compute correctly
- Model predictions run
- No errors in pipeline

---

## Monitoring

### Real-Time Dashboard

The system includes a terminal dashboard showing:
- Portfolio value and P&L
- Active positions
- Model predictions
- Safety status
- Recent trades

### Trade Logging

All trades are logged to SQLite for analysis:
- Entry/exit prices
- P&L per trade
- Commission costs
- Timestamps

---

## Limitations

This execution layer is **demonstrative**, not production-grade:

- Single exchange support (Binance)
- Market orders only
- No advanced order types
- Limited error recovery
- Requires manual monitoring

For production trading systems, additional engineering is required around:
- High availability
- Disaster recovery
- Regulatory compliance
- Advanced order management
