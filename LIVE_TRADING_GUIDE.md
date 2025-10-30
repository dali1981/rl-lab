# Live Trading Guide

Complete guide for running the RL Trading Lab live trading system on Binance testnet.

## Quick Start

### 1. Prerequisites

**Get Binance Testnet API Keys:**
1. Visit https://testnet.binance.vision/
2. Create an account (no KYC required)
3. Generate API keys

**Set Environment Variables:**
```bash
export BINANCE_TESTNET_KEY="your_testnet_key_here"
export BINANCE_TESTNET_SECRET="your_testnet_secret_here"
```

**Install Dependencies:**
```bash
cd /Users/mohamedali/trading_project/rl-trading-lab
uv sync
```

### 2. Validate the System

Before going live, validate the entire pipeline with historical data:

```bash
uv run python experiments/validate_live.py \
  --symbol BTCUSDT \
  --days 1 \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --vecnormalize checkpoints/PPO_returns_20251028_143659/vecnormalize.pkl
```

This will:
- Load recent data from MinIO
- Create dollar volume bars
- Compute features
- Run model predictions
- Simulate trades
- Show PnL and statistics

### 3. Run Live Trading (Testnet)

**Simple run (single symbol):**
```bash
uv run python experiments/live_trading.py \
  --config configs/trading/testnet.yaml \
  --symbols BTCUSDT
```

**Multi-symbol trading:**
```bash
uv run python experiments/live_trading.py \
  --config configs/trading/testnet.yaml \
  --symbols BTCUSDT ETHUSDT BNBUSDT
```

**With custom model:**
```bash
uv run python experiments/live_trading.py \
  --config configs/trading/testnet.yaml \
  --symbols BTCUSDT \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip
```

### 4. Monitor the Dashboard

The live trading dashboard shows:
- **Portfolio Summary**: Balance, PnL, returns, drawdown
- **Active Positions**: Current positions per symbol
- **Model Predictions**: Real-time BUY/SELL/HOLD signals with confidence
- **Recent Trades**: Last 5 trades with PnL
- **Safety Status**: Circuit breaker state and violations

## Configuration

### Testnet Configuration (`configs/trading/testnet.yaml`)

Key parameters:

```yaml
# Risk Management
risk:
  initial_balance: 10000      # Starting balance (USD)
  max_position_pct: 0.95      # Max 95% in positions
  max_position_size: 1000     # Max position size per trade
  max_drawdown: 0.20          # Stop at 20% drawdown
  max_trades_per_hour: 20     # Rate limit
  min_holding_period: 60      # Prevent churning (seconds)

# Dollar Volume Bars
dollar_volume_thresholds:
  BTCUSDT: 1000000  # $1M per bar
  ETHUSDT: 500000   # $500K per bar
  default: 100000   # Default for others
```

### Live Configuration (`configs/trading/live.yaml`)

⚠️ **More conservative settings for real money:**
- Smaller position sizes
- Tighter stop-loss
- Fewer trades per hour
- Stricter drawdown limits

## Safety Features

### 1. Circuit Breaker

Automatically stops trading when:
- Drawdown exceeds threshold (default: 20%)
- Balance drops below minimum
- Too many consecutive losses
- Trade rate limits exceeded

### 2. Position Limits

- Maximum position size per trade
- Maximum % of balance in positions
- Minimum order size enforcement

### 3. Rate Limiting

- Hourly trade limit (default: 20)
- Daily trade limit (default: 100)
- Prevents API rate limit issues

### 4. Trade Validation

- Checks every order before execution
- Validates position sizes
- Ensures sufficient balance
- Prevents invalid orders

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Live Trading System                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐                    │
│  │   Binance    │      │   Dollar     │                    │
│  │   WebSocket  ├─────>│   Volume     │                    │
│  │   Streams    │      │   Bars       │                    │
│  └──────────────┘      └──────┬───────┘                    │
│                                │                            │
│                        ┌───────v────────┐                   │
│                        │   Feature      │                   │
│                        │   Computer     │                   │
│                        │  (Rolling Win) │                   │
│                        └───────┬────────┘                   │
│                                │                            │
│                        ┌───────v────────┐                   │
│                        │     Model      │                   │
│                        │   Inference    │                   │
│                        │   (PPO/A2C)    │                   │
│                        └───────┬────────┘                   │
│                                │                            │
│             ┌──────────────────┼──────────────────┐         │
│             │                  │                  │         │
│      ┌──────v──────┐   ┌──────v─────┐   ┌───────v──────┐  │
│      │   Safety    │   │   Order    │   │  Portfolio   │  │
│      │   Guard     │   │  Executor  │   │   Manager    │  │
│      └─────────────┘   └────────────┘   └──────────────┘  │
│                                                              │
│                        ┌──────────────┐                     │
│                        │  Dashboard   │                     │
│                        │   (Rich)     │                     │
│                        └──────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. StreamConsumer (`live/stream_consumer.py`)
- Connects to Binance WebSocket
- Buffers trades
- Creates dollar volume bars on-the-fly
- Triggers callbacks when bars are ready

### 2. FeatureComputer (`live/feature_computer.py`)
- Maintains rolling window of bars
- Computes indicators (SMA, fracdiff, etc.)
- Z-score normalization
- Matches training data format exactly

### 3. ModelInferenceEngine (`live/inference.py`)
- Loads trained PPO/A2C/DQN models
- Predicts BUY/SELL/HOLD actions
- Returns confidence scores
- Supports VecNormalize wrapper

### 4. OrderExecutor (`live/executor.py`)
- Translates actions into Binance orders
- Manages positions per symbol
- Calculates commissions
- Handles order failures

### 5. PortfolioManager (`live/portfolio.py`)
- Tracks multi-symbol portfolio
- Calculates PnL and returns
- Stores trade history in SQLite
- Updates current prices

### 6. SafetyGuard (`live/safety.py`)
- Circuit breaker system
- Trade rate limiting
- Drawdown monitoring
- Consecutive loss tracking

### 7. TradingDashboard (`live/dashboard.py`)
- Real-time Rich display
- Portfolio metrics
- Position tracking
- Trade history
- Prediction display

## Troubleshooting

### "Missing Binance API credentials"
Set environment variables:
```bash
export BINANCE_TESTNET_KEY="your_key"
export BINANCE_TESTNET_SECRET="your_secret"
```

### "No files in log segment" (Delta Lake error)
Data collection may have failed. Run:
```bash
cd ../dlt-starter
uv run python examples/01_run_pipeline_example.py --symbol BTCUSDT --delta
```

### "Model not found"
Check that model path in config is correct:
```yaml
models:
  BTCUSDT:
    path: "checkpoints/PPO_returns_20251028_143659/best_model"
```

### Circuit breaker triggered
Check safety guard status in dashboard. Reset if manual stop:
- The system will show "OPEN" or "MANUAL" state
- Review violations in logs
- Adjust risk parameters if needed

### WebSocket disconnects
The system auto-reconnects up to max attempts (default: 5).
Check network connectivity and Binance API status.

## Best Practices

### 1. Start Small
- Begin with testnet (no real money)
- Test with single symbol first
- Use small position sizes initially
- Monitor for at least 24 hours

### 2. Monitor Closely
- Watch the dashboard continuously
- Check trade history regularly
- Review safety guard violations
- Monitor model confidence scores

### 3. Gradual Scaling
- Start with 1 symbol → 2-3 symbols → portfolio
- Increase position sizes gradually
- Test different market conditions
- Keep detailed logs

### 4. Regular Validation
- Run validation script before each session
- Compare features with training data
- Check model predictions make sense
- Review recent PnL patterns

### 5. Emergency Stop
Press `Ctrl+C` to gracefully shutdown. The system will:
- Stop accepting new trades
- Close WebSocket connections
- Save all data
- Print final summary

## Performance Expectations

**Latency:**
- Bar creation: Real-time (on threshold)
- Feature computation: <100ms
- Model inference: <50ms
- Order execution: <200ms
- End-to-end: ~500ms per bar

**Resource Usage:**
- CPU: ~5-10% (single symbol)
- Memory: ~200-300MB
- Network: Minimal (WebSocket stream)

## Next Steps

1. ✅ Run validation script
2. ✅ Start with testnet
3. ✅ Monitor for 24 hours
4. ✅ Review performance
5. ✅ Adjust parameters
6. ⚠️ Consider live trading (at your own risk!)

## Support

For issues or questions:
1. Check logs in console
2. Review BINANCE_TESTNET_STATUS.md
3. Run validation script for diagnostics
4. Check Binance testnet status

## Important Warnings

⚠️ **Testnet vs Live:**
- Testnet uses fake money (no risk)
- Testnet resets monthly
- Live trading uses REAL money
- Always test thoroughly on testnet first

⚠️ **Risk Disclaimer:**
- Trading involves risk of loss
- Past performance ≠ future results
- Use at your own risk
- Never invest more than you can afford to lose

⚠️ **Model Limitations:**
- Models trained on historical data
- May not perform in all market conditions
- Requires regular retraining
- Monitor performance continuously
