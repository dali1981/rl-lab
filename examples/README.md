# Live Trading Examples

This directory contains comprehensive examples for deploying trained RL models to live trading on Binance testnet.

## Quick Start

### Prerequisites

1. **Get Binance Testnet API Keys**
   - Visit https://testnet.binance.vision/
   - Create an account (no KYC required, no real money)
   - Generate API keys

2. **Set Environment Variables**
   ```bash
   export BINANCE_TESTNET_KEY="your_testnet_key_here"
   export BINANCE_TESTNET_SECRET="your_testnet_secret_here"
   ```

3. **Install Dependencies**
   ```bash
   cd /Users/mohamedali/trading_project/rl-trading-lab
   uv sync
   ```

### Step 1: Validate Your Model

**ALWAYS** validate before live trading:

```bash
# Test with 1 day of historical data
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --days 1
```

This will:
- ✓ Load recent data from MinIO
- ✓ Create dollar volume bars
- ✓ Compute features
- ✓ Test model predictions
- ✓ Simulate trades
- ✓ Show expected performance

### Step 2: Run on Testnet

Once validation passes, start live trading on testnet:

```bash
# Single symbol trading
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT \
  --balance 10000
```

### Step 3: Monitor Dashboard

The live trading system shows a real-time dashboard with:
- **Portfolio Summary**: Balance, PnL, returns, drawdown
- **Active Positions**: Current positions and entry prices
- **Model Predictions**: Real-time BUY/SELL/HOLD signals with confidence
- **Recent Trades**: Last 10 trades with PnL
- **Safety Status**: Circuit breaker state and violations

Press `Ctrl+C` to stop gracefully.

### Step 4: Analyze Results

After trading, analyze your session:

```bash
uv run python examples/live_trading_example.py analyze --db portfolio.db
```

This shows:
- Total trades and win rate
- PnL and commission
- Average win/loss
- Profit factor
- Recent trades table
- Exports results to CSV

---

## Files in This Directory

### `live_trading_example.py`

**Complete reference implementation** demonstrating:

#### Section 1: Custom Safety Guard
- Extends `SafetyGuard` with custom rules
- Trading hours restrictions (e.g., 9am-9pm UTC only)
- Symbol-specific risk limits
- Custom violation tracking

Example:
```python
from live_trading_example import CustomSafetyGuard

guard = CustomSafetyGuard(
    max_drawdown=0.20,
    max_trades_per_hour=20,
    trading_hours=(9, 21),  # Only trade 9am-9pm UTC
)
```

#### Section 2: Model Loading and Inspection
- Load PPO/A2C/DQN models
- Auto-detect VecNormalize wrapper
- Inspect model metadata
- Test predictions

Example:
```python
from live_trading_example import load_and_inspect_model

engine = load_and_inspect_model(
    model_path="checkpoints/PPO_returns_20251028_143659/best_model.zip",
    vecnormalize_path=None,  # Auto-detected if in same directory
)
```

#### Section 3: Historical Validation
- Test entire pipeline with real data
- No actual trading
- Validates: data → bars → features → predictions → simulated trades

#### Section 4: Live Trading System
- Complete orchestration of all components
- Error handling and auto-reconnection
- Real-time dashboard
- Graceful shutdown

#### Section 5: Trade Analysis
- Load trade history from SQLite
- Calculate performance metrics
- Generate statistics tables
- Export to CSV

#### Section 6: CLI Commands
Three commands available:
- `validate`: Test with historical data
- `trade`: Run live on testnet
- `analyze`: Analyze past session

### `configs/single_symbol.yaml`

Simplified configuration file showing all available options:

```yaml
# Trading setup
symbol: BTCUSDT

# Model configuration
model:
  path: "checkpoints/PPO_returns_20251028_143659/best_model"
  vecnormalize: "checkpoints/PPO_returns_20251028_143659/vecnormalize.pkl"

# Dollar volume threshold ($1M per bar)
dollar_volume_threshold: 1000000

# Risk management
risk:
  initial_balance: 10000
  max_position_size: 1000
  max_drawdown: 0.20
  max_trades_per_hour: 20

# Safety guards
safety:
  enable_circuit_breaker: true
  trading_hours_start: 9   # 9am UTC
  trading_hours_end: 21    # 9pm UTC
```

---

## Usage Examples

### Example 1: Basic Validation

Test your model with 1 day of recent data:

```bash
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT \
  --days 1
```

### Example 2: Extended Validation

Test with more data for better confidence:

```bash
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT \
  --days 7
```

### Example 3: Live Trading on Testnet

Start live trading with default settings:

```bash
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT
```

### Example 4: Custom Risk Parameters

Trade with custom balance and drawdown limit:

```bash
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT \
  --balance 50000 \
  --max-drawdown 0.15
```

### Example 5: Different Model

Try a different trained model:

```bash
uv run python examples/live_trading_example.py trade \
  --model checkpoints/A2C_returns_20251028_143912/best_model.zip \
  --symbol BTCUSDT
```

### Example 6: Post-Trading Analysis

After stopping the trading session:

```bash
uv run python examples/live_trading_example.py analyze --db portfolio.db
```

---

## Understanding the Components

### StreamConsumer
Connects to Binance WebSocket and creates dollar volume bars in real-time.

**Key concept**: Dollar volume bars adapt to market activity. During high volume, bars form faster; during low volume, bars form slower. This is superior to time-based bars.

### FeatureComputer
Maintains a rolling window of recent bars and computes features on-the-fly.

**Key concept**: Features must match training data exactly. The computer uses the same indicators, windows, and normalization as during training.

### ModelInferenceEngine
Loads your trained RL model and makes predictions.

**Key concept**: Supports VecNormalize wrapper for observation normalization. If your model was trained with VecNormalize, you must use it in live trading.

### OrderExecutor
Translates RL actions (BUY/SELL/HOLD) into actual Binance market orders.

**Key concept**: Manages position state. You can only have one position per symbol at a time (long or flat, or short or flat).

### PortfolioManager
Tracks your portfolio across multiple symbols (if trading multiple).

**Key concept**: Uses SQLite to persist trade history. You can query this database later for analysis.

### SafetyGuard
Circuit breaker system that stops trading when risk thresholds are exceeded.

**Key concept**: Multiple layers of protection:
- Drawdown limits (stop at 20% loss by default)
- Rate limits (max trades per hour/day)
- Consecutive loss limits (stop after 5 losses in a row)
- Custom rules (trading hours, volatility checks, etc.)

### TradingDashboard
Real-time Rich terminal UI showing portfolio and trading activity.

**Key concept**: Non-blocking updates. The dashboard refreshes without interrupting trading logic.

---

## Error Handling

The example demonstrates robust error handling for common scenarios:

### 1. Connection Errors

```python
# WebSocket disconnects
# System auto-reconnects up to max_reconnect_attempts (default: 5)
# If reconnection fails, system shuts down gracefully
```

### 2. Model Loading Errors

```python
# Missing model file
# Incompatible model version
# Missing VecNormalize wrapper
# All result in clear error messages with suggestions
```

### 3. Insufficient Balance

```python
# If you don't have enough cash for a trade
# Order is skipped with warning
# System continues monitoring
```

### 4. Circuit Breaker Triggered

```python
# When drawdown exceeds threshold
# All trading stops immediately
# Positions remain open (not auto-closed)
# Dashboard shows OPEN state
```

### 5. API Rate Limits

```python
# If you exceed Binance rate limits
# SafetyGuard prevents new orders
# System waits until limit resets
```

---

## Safety Best Practices

### 1. Always Validate First

**Never** skip the validation step:
```bash
# This is NON-NEGOTIABLE
uv run python examples/live_trading_example.py validate --model your_model.zip --days 1
```

### 2. Start with Testnet

Testnet uses fake money. Perfect for:
- Testing your model in real market conditions
- Learning how the system behaves
- Debugging issues
- Building confidence

Get testnet keys: https://testnet.binance.vision/

### 3. Start Small

When moving to live trading:
- Start with minimum balance ($100)
- Use maximum position limits
- Set tight stop-loss
- Trade one symbol only
- Monitor closely for 24 hours

### 4. Monitor Continuously

Watch the dashboard:
- Check predictions make sense
- Verify PnL is reasonable
- Monitor safety guard status
- Review recent trades
- Watch for violations

### 5. Understand Circuit Breakers

The system will stop trading when:
- Drawdown > 20% (configurable)
- Balance < $100 (configurable)
- 5 consecutive losses (configurable)
- Rate limits exceeded

**This is a good thing!** Better to stop early than lose everything.

### 6. Regular Retraining

Models trained on old data may not perform well in new market conditions.

Retrain regularly:
- Every month minimum
- After major market events
- When performance degrades
- When market regime changes

### 7. Diversify

Don't rely on a single:
- Model (try PPO, A2C, DQN)
- Symbol (trade multiple pairs)
- Strategy (combine with other approaches)
- Timeframe (use different bar thresholds)

---

## Troubleshooting

### Problem: "Missing Binance API credentials"

**Solution:**
```bash
export BINANCE_TESTNET_KEY="your_key"
export BINANCE_TESTNET_SECRET="your_secret"
```

Check they're set:
```bash
echo $BINANCE_TESTNET_KEY
```

### Problem: "No files in log segment" (Delta Lake error)

**Solution:**
Data collection may have failed. Run the data pipeline:
```bash
cd ../dlt-starter
uv run python examples/01_run_pipeline_example.py --symbol BTCUSDT --delta
```

### Problem: "Model not found"

**Solution:**
Check model path exists:
```bash
ls -la checkpoints/PPO_returns_20251028_143659/best_model.zip
```

Use absolute or relative path from project root.

### Problem: Circuit breaker triggered immediately

**Solution:**
Check if previous session left circuit breaker open:
```bash
# Delete old portfolio database to reset
rm portfolio.db

# Or adjust max_drawdown parameter
uv run python examples/live_trading_example.py trade --max-drawdown 0.30
```

### Problem: WebSocket keeps disconnecting

**Solution:**
- Check internet connection
- Verify Binance testnet is operational
- Increase reconnect attempts in config
- Check firewall settings

### Problem: No trades being executed

**Possible causes:**
1. Model only predicting HOLD (check with validation)
2. Safety guard preventing trades (check dashboard)
3. Insufficient balance (check portfolio)
4. Outside trading hours (if using custom safety guard)
5. Rate limits reached (check violations)

**Debug:**
```bash
# Run with debug logging
export LOG_LEVEL=DEBUG
uv run python examples/live_trading_example.py trade ...
```

---

## Advanced Usage

### Programmatic Integration

Use the example components in your own Python code:

```python
from examples.live_trading_example import (
    CustomSafetyGuard,
    load_and_inspect_model,
    LiveTradingSystem,
)

# Load model
engine = load_and_inspect_model("checkpoints/PPO.../best_model.zip")

# Create system
system = LiveTradingSystem(
    symbol="BTCUSDT",
    model_path="checkpoints/PPO.../best_model.zip",
    initial_balance=10000,
    trading_hours=(9, 21),  # Custom hours
)

# Run
import asyncio
asyncio.run(system.run())
```

### Custom Callbacks

Extend the `LiveTradingSystem` class:

```python
class MyTradingSystem(LiveTradingSystem):
    async def on_bar(self, symbol: str, bar):
        # Add custom logic before/after trading
        await super().on_bar(symbol, bar)

        # Custom notification
        if self.portfolio.get_stats()['returns'] > 0.10:
            self.send_alert("10% profit reached!")
```

### Multiple Models

Compare different models:

```python
# Terminal 1: Run PPO model
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO.../best_model.zip

# Terminal 2: Run A2C model
uv run python examples/live_trading_example.py trade \
  --model checkpoints/A2C.../best_model.zip
```

Compare results:
```python
# Analyze both databases
uv run python examples/live_trading_example.py analyze --db portfolio_ppo.db
uv run python examples/live_trading_example.py analyze --db portfolio_a2c.db
```

---

## Related Documentation

### Internal
- `../LIVE_TRADING_GUIDE.md` - Complete system documentation
- `../BINANCE_TESTNET_STATUS.md` - Project status and architecture
- `../notebooks/03_live_trading_tutorial.ipynb` - Interactive tutorial

### External
- [Binance Testnet](https://testnet.binance.vision/) - Get API keys
- [python-binance docs](https://python-binance.readthedocs.io/) - API library
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms
- [Rich documentation](https://rich.readthedocs.io/) - Terminal UI library

---

## Support

For issues or questions:
1. Check this README
2. Read `LIVE_TRADING_GUIDE.md`
3. Run validation script for diagnostics
4. Check Binance testnet status
5. Review logs in console output

---

## ⚠️ Important Warnings

### Testnet vs Live

- **Testnet**: Fake money, no risk, resets monthly
- **Live**: REAL money, REAL risk, permanent losses possible

**Always test on testnet first!**

### Risk Disclaimer

- Trading involves risk of loss
- Past performance ≠ future results
- RL models can fail in new market conditions
- Use at your own risk
- Never invest more than you can afford to lose

### Model Limitations

- Models trained on historical data
- May not perform in all market conditions
- Requires regular retraining
- No guarantee of profitability
- Monitor performance continuously

---

**Last Updated:** 2025-11-05
**Status:** Production Ready ✅
