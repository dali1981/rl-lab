# Binance Testnet Trading Implementation Status

**Project:** RL Trading Lab - Live Trading on Binance Testnet
**Branch:** `binance/testnet`
**Last Updated:** 2025-10-28
**Status:** Phase 1 Complete, Bug Fix Required

---

## 🎯 Project Goal

Deploy trained RL models (PPO, A2C, DQN) to trade live on Binance testnet using:
- Real-time WebSocket data streaming
- Dollar volume bar sampling
- Multi-symbol portfolio trading
- Full observability and safety guards

---

## ✅ Phase 1: Data Pipeline Integration (COMPLETE)

### What We Built

#### 1. **BinanceDataAdapter** (`src/rl_trading_lab/data/binance_adapter.py`)
- Loads tick data from MinIO Delta Lake storage
- Connects to `s3://binance-data/warehouse/`
- Supports date range filtering and multi-symbol loading
- Query methods: `load_symbol_data()`, `load_multiple_symbols()`, `get_available_symbols()`

**⚠️ BUG IDENTIFIED:**
- Line 45: `dataset_name: str = "binance_test.db"` should be `"binance_test"` (no `.db`)
- dlt writes to `s3://binance-data/warehouse/binance_test` but adapter looks for `.db` suffix

#### 2. **BarProcessor** (`src/rl_trading_lab/data/bar_processor.py`)
- Creates dollar volume bars from tick data
- Configurable thresholds per symbol:
  - BTCUSDT: $1,000,000 per bar
  - ETHUSDT: $500,000 per bar
  - BNBUSDT: $100,000 per bar
  - Default: $100,000 per bar
- Wraps dlt-starter's `DollarVolumeSampler` with fallback implementation
- Multi-symbol processing via `MultiSymbolBarProcessor`

#### 3. **FeaturePipeline** (`src/rl_trading_lab/data/feature_pipeline.py`)
- Engineers ML features matching training data format:
  - `ratio_sma_5_close`: SMA(5) / close ratio
  - `ratio_sma_20_close`: SMA(20) / close ratio
  - `ratio_range_close`: (high - low) / close ratio
  - `fracdiff_0.4`: Fractional differentiation (d=0.4)
  - Z-score normalized versions of all indicators
- Includes fallback implementations for fractional differentiation
- Can save/load feature statistics for consistent normalization

#### 4. **Validation Script** (`experiments/validate_data_pipeline.py`)
- Tests entire pipeline end-to-end
- Loads data → creates bars → engineers features → compares with training data
- Rich terminal output with tables and color coding

### Dependencies Added

```toml
"deltalake>=0.15.0"      # Delta Lake for MinIO data access
"boto3>=1.28.0"          # S3/MinIO access
"python-binance>=1.0.19" # Binance API for testnet trading
"typer>=0.9.0"           # CLI interface
```

### Commit

- **Hash:** `ffef7ac`
- **Message:** "Phase 1: Data Pipeline Integration for Binance Live Trading"
- **Files Changed:** 7 files, 2474 insertions

---

## 🐛 Current Issue: Path Mismatch

### Problem
The validation script fails with:
```
ERROR: Generic delta kernel error: No files in log segment
```

### Root Cause
**Data location mismatch:**
- dlt wrote data to: `s3://binance-data/warehouse/binance_test/`
- Adapter looks for: `s3://binance-data/warehouse/binance_test.db/`

**Dataset name in dlt pipeline:**
```python
# dlt-starter/examples/01_run_pipeline_example.py:194
dataset_name="binance_test"  # No .db suffix
```

**Dataset name in adapter:**
```python
# rl-trading-lab/src/rl_trading_lab/data/binance_adapter.py:45
dataset_name: str = "binance_test.db"  # Wrong - has .db suffix
```

### Fix Required
Change line 45 in `src/rl_trading_lab/data/binance_adapter.py`:
```python
# Before:
dataset_name: str = "binance_test.db",

# After:
dataset_name: str = "binance_test",
```

### Testing After Fix
```bash
cd /Users/mohamedali/trading_project/rl-trading-lab
uv run python experiments/validate_data_pipeline.py --symbol BTCUSDT --days 3
```

Expected output:
- ✓ Load 3 days of BTCUSDT trades from MinIO
- ✓ Create dollar volume bars (~100-200 bars)
- ✓ Engineer features (14 columns)
- ✓ Compare with training data distributions

---

## 📋 Remaining Implementation

### Phase 2: Real-Time Streaming System

#### Files to Create:

**1. Real-Time Data Consumer** (`src/rl_trading_lab/live/stream_consumer.py`)
- Use dlt-starter's WebSocket implementation
- Buffer trades until dollar volume threshold reached
- Create bars on-the-fly using `DollarVolumeSampler`
- Multi-symbol support with independent bar tracking

**2. Feature Computer** (`src/rl_trading_lab/live/feature_computer.py`)
- Maintain rolling window of recent bars for each symbol
- Compute indicators incrementally as new bars arrive
- Handle lookback requirements (e.g., 20-bar SMA)
- Normalize features using saved statistics from training

**3. Model Inference Engine** (`src/rl_trading_lab/live/inference.py`)
- Load trained model from checkpoints (e.g., `checkpoints/PPO_returns_20251028_143659/best_model.zip`)
- Load VecNormalize wrapper (`vecnormalize.pkl`)
- Predict actions in real-time
- Support multiple models simultaneously

### Phase 3: Binance Testnet Integration

#### Files to Create:

**1. Testnet Configuration** (`configs/trading/testnet.yaml`)
```yaml
testnet:
  api_endpoint: https://testnet.binance.vision
  api_key: ${BINANCE_TESTNET_KEY}
  api_secret: ${BINANCE_TESTNET_SECRET}
  max_position_size: 1000  # USD
  rate_limit: 1200  # requests per minute
```

**2. Order Execution Manager** (`src/rl_trading_lab/live/executor.py`)
- Connect to Binance testnet API
- Translate RL actions (BUY/SELL/HOLD) → market orders
- Position tracking per symbol
- Risk management: max position size, stop-loss
- Transaction cost accounting (commission, slippage)

**3. Portfolio Manager** (`src/rl_trading_lab/live/portfolio.py`)
- Multi-symbol portfolio tracking
- Balance allocation across symbols
- PnL calculation and reporting
- SQLite database for trade history

### Phase 4: Live Trading Runner

#### Files to Create:

**1. Live Trading Script** (`experiments/live_trading.py`)
```python
# CLI interface with symbol selection
typer CLI:
  --symbols BTCUSDT ETHUSDT BNBUSDT
  --mode testnet|paper|live
  --models-dir checkpoints/

# Main loop:
  1. Start WebSocket streams for all symbols
  2. Process bars → features → predictions → orders
  3. Update dashboard in real-time
```

**2. Configuration** (`configs/trading/live.yaml`)
```yaml
trading:
  mode: testnet
  symbols: [BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT]
  models:
    BTCUSDT: checkpoints/PPO_returns_20251028_143659/best_model
    ETHUSDT: checkpoints/A2C_returns_20251028_143912/best_model
  dollar_volume_thresholds:
    BTCUSDT: 1000000
    ETHUSDT: 500000
    default: 100000
  risk:
    max_position_pct: 0.95
    max_drawdown: 0.20
    initial_balance: 10000
```

**3. Dashboard** (`src/rl_trading_lab/live/dashboard.py`)
- Real-time PnL per symbol using Rich library
- Position tracking
- Recent trades log
- Model predictions display
- Streaming bar updates

### Phase 5: Monitoring & Safety

#### Files to Create:

**1. Safety Guards** (`src/rl_trading_lab/live/safety.py`)
- Max drawdown circuit breaker
- Position size limits
- API rate limiting
- Connection monitoring with auto-reconnect
- Emergency stop functionality

**2. Validation Script** (`experiments/validate_live.py`)
- Run on recent historical data before going live
- Simulate entire live trading pipeline
- Verify features match training data distribution
- Stress test with edge cases

---

## 🗂️ File Structure

```
rl-trading-lab/
├── src/rl_trading_lab/
│   ├── data/
│   │   ├── __init__.py                  ✅ Done
│   │   ├── binance_adapter.py           ✅ Done (needs 1-line fix)
│   │   ├── bar_processor.py             ✅ Done
│   │   └── feature_pipeline.py          ✅ Done
│   └── live/                             📝 To Create
│       ├── __init__.py
│       ├── stream_consumer.py
│       ├── feature_computer.py
│       ├── inference.py
│       ├── executor.py
│       ├── portfolio.py
│       ├── dashboard.py
│       └── safety.py
├── experiments/
│   ├── validate_data_pipeline.py        ✅ Done
│   ├── live_trading.py                  📝 To Create
│   └── validate_live.py                 📝 To Create
├── configs/trading/                      📝 To Create
│   ├── testnet.yaml
│   └── live.yaml
└── checkpoints/                          ✅ Exists
    └── [trained models...]
```

---

## 🔑 Key Design Decisions

### Dollar Volume Bars vs Time Bars
**Decision:** Use dollar volume bars (adaptive sampling)
**Rationale:**
- Better statistical properties (more normally distributed returns)
- Reduced serial correlation
- Information-driven sampling adapts to market activity
- Matches research on information-theoretic sampling

### Feature Engineering Approach
**Decision:** Exact replication of training features
**Rationale:**
- Avoid distribution shift between training and live data
- Use same indicator windows, normalization params
- Save/load feature statistics from training

### Multi-Symbol Strategy
**Decision:** Independent models per symbol OR shared model
**Rationale:**
- Separate bar tracking per symbol
- Portfolio-level risk management
- Can use different models for different symbols

### Testnet First
**Decision:** Deploy to Binance testnet before live trading
**Rationale:**
- No real capital at risk
- Test entire pipeline end-to-end
- Free, no KYC required
- Testnet resets monthly (perfect for testing)

---

## 📊 Data Pipeline Status

### MinIO Data Availability

**Last Pipeline Run:** 2025-10-28 22:52:58
- **Duration:** 1:04:49
- **Symbol:** BTCUSDT
- **Records:** ~359,000 trades
- **Date Range:** 2025-10-25 09:45:00 to recent
- **Location:** `s3://binance-data/warehouse/binance_test/BTCUSDT/`
- **Format:** Delta Lake with date partitioning

**Known Issues:**
- Had 1 timeout error during collection (recovered)
- Path mismatch bug (detailed above)

### Training Data Reference

**Location:** `/Users/mohamedali/trading_project/tools/examples/btcusdt_fractional_indicators.parquet`
- **Size:** 4.0 MB
- **Records:** 56,663 bars
- **Features:** 14 columns
  - OHLCV: timestamp, open, high, low, close, volume
  - Indicators: ratio_sma_5_close, ratio_sma_20_close, ratio_range_close, fracdiff_0.4
  - Z-scores: *_zscore versions of all indicators

### Available Models

**Location:** `/Users/mohamedali/trading_project/rl-trading-lab/checkpoints/`

**Best Models:**
- `PPO_returns_20251028_143659/best_model.zip` (2.1 MB)
- `A2C_returns_20251028_143912/best_model.zip`
- `DQN_returns_20251028_081209/best_model.zip`

**Model Contents:**
- `best_model.zip`: Trained weights
- `vecnormalize.pkl`: Observation normalization statistics
- `best_model.metadata.json`: Training configuration

---

## 🚀 Quick Start (After Bug Fix)

### 1. Fix the Bug
```bash
# Edit src/rl_trading_lab/data/binance_adapter.py line 45
# Change: dataset_name: str = "binance_test.db"
# To:     dataset_name: str = "binance_test"
```

### 2. Test Data Pipeline
```bash
cd /Users/mohamedali/trading_project/rl-trading-lab
uv run python experiments/validate_data_pipeline.py --symbol BTCUSDT --days 3
```

Expected output:
```
✓ Loaded 300,000+ trades for BTCUSDT
✓ Created 150+ bars from trades
✓ Created features: 14 total columns
✓ Distributions are within acceptable range
```

### 3. Continue with Phase 2
```bash
# Create live/ directory
mkdir -p src/rl_trading_lab/live

# Implement real-time streaming components
# (See Phase 2 details above)
```

---

## 🎯 Success Criteria

**Phase 1 (Current):**
- ✅ Load Binance data from MinIO
- ✅ Create dollar volume bars
- ✅ Engineer features matching training format
- ⚠️ Validate feature distributions (blocked by bug)

**Phase 2 (Next):**
- [ ] Stream real-time data via WebSocket
- [ ] Create bars in real-time
- [ ] Compute features in real-time
- [ ] Load and run trained models

**Phase 3 (Testnet):**
- [ ] Connect to Binance testnet
- [ ] Execute trades
- [ ] Track portfolio
- [ ] Handle errors gracefully

**Phase 4 (Production Ready):**
- [ ] Multi-symbol live trading
- [ ] Real-time dashboard
- [ ] Safety guards active
- [ ] Monitoring and alerting

---

## 🔗 Related Documentation

### Internal
- `../dlt-starter/README.md` - Data pipeline documentation
- `../dlt-starter/DELTA_LAKE_SETUP.md` - MinIO/Delta Lake setup
- `README.md` - RL Trading Lab overview

### External
- [Binance Testnet](https://testnet.binance.vision/) - Create testnet account
- [python-binance docs](https://python-binance.readthedocs.io/) - API library
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms

---

## 📝 Notes for Next Session

### Immediate Actions
1. **Fix the dataset_name bug** (1-line change)
2. **Run validation script** to confirm Phase 1 works
3. **Commit the fix** before continuing

### Then Continue With
1. **Phase 2 implementation** - Real-time streaming
2. **Test with recent data** from MinIO
3. **Create dashboard** for monitoring

### Environment Setup for Testnet
```bash
# Create .env file
echo "BINANCE_TESTNET_KEY=your_key" > .env
echo "BINANCE_TESTNET_SECRET=your_secret" >> .env

# Get keys from: https://testnet.binance.vision/
# Click "API Key" after login
```

### Performance Expectations
- **Bar creation:** ~2-3 bars per second (dollar volume threshold dependent)
- **Feature computation:** <100ms per bar
- **Model inference:** <50ms per prediction
- **Order execution:** <200ms to Binance testnet

---

## 🆘 Troubleshooting

### "No files in log segment" Error
**Cause:** Path mismatch between dlt output and adapter config
**Fix:** Change `dataset_name` to `"binance_test"` (no `.db`)

### "MinIO connection refused"
**Check:** Is MinIO running?
```bash
cd ../dlt-starter
docker-compose ps
docker-compose up -d  # if not running
```

### "Missing fracdiff function"
**Cause:** tools package not in path
**Fix:** Adapter has fallback implementation, should work automatically

### "Model file not found"
**Check:** Checkpoint path exists
```bash
ls -la checkpoints/PPO_returns_20251028_143659/best_model.zip
```

---

**Last Updated:** 2025-10-28 by Claude Code
**Next Review:** After Phase 1 bug fix and validation