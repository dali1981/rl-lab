# RL Trading Lab

A research and engineering framework for **reinforcement learning-based trading systems**, focused on reproducible experiments, custom environment design, and controlled deployment.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository demonstrates how I structure RL projects for real-world constraints:
- Custom environments tailored to trading data
- Config-driven experimentation (Hydra + MLflow)
- Transparent metrics and checkpoints
- Optional live / paper trading with safety guards

---

## What this project is for

This project is used to:
- Design custom RL environments for proprietary market data
- Train and benchmark agents using Stable-Baselines3
- Run reproducible experiments with Hydra and MLflow
- Prototype execution logic with explicit risk controls

### What this project is NOT
- Not a turnkey trading bot
- Not financial advice
- Not a promise of profitability

---

## 60-second smoke test

Run a minimal training loop to verify installation:

```bash
git clone https://github.com/dali1981/rl-lab.git
cd rl-lab
uv sync

uv run python experiments/train.py \
  training.total_timesteps=1000
```

Expected results:
- Training starts successfully
- Metrics logged to `mlruns/`
- Checkpoint written to `checkpoints/`

---

## Documentation

- [Architecture overview](docs/architecture.md)
- [Configuration & experiments](docs/CONFIGURATION_GUIDE.md)
- [Logging guide](docs/LOGGING_GUIDE.md)
- [Live trading (optional)](LIVE_TRADING_GUIDE.md)

---

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/dali1981/rl-lab.git
cd rl-lab

# Install dependencies with UV
uv sync

# Activate environment (if needed)
source .venv/bin/activate
```

### 2. Data Preparation

Your data is ready in `tools/examples/btcusdt_fractional_indicators.parquet`!

Alternatively, use the Kedro pipeline:
```bash
cd ../kedro-crypto-ind
kedro run --pipeline=feature_engineering
```

Or use the dlt pipeline for live data:
```bash
cd ../dlt-starter
uv run python examples/01_run_pipeline_example.py --symbol BTCUSDT --delta
```

### 3. Train Your First Model

```bash
# Train PPO with default settings (recommended)
uv run python experiments/train.py

# Quick test (1000 timesteps)
uv run python experiments/train.py training.total_timesteps=1000

# Train different algorithms
uv run python experiments/train.py agent=a2c
uv run python experiments/train.py agent=dqn

# Change reward function
uv run python experiments/train.py env.environment_params.reward_type=sharpe
```

### 4. View Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser to http://localhost:5000
# Compare experiments, view metrics, download models
```

---

## Training Features

### Environment Capabilities

- **Action Spaces**:
  - Discrete: Buy/Sell/Hold (default)
  - Continuous: Position sizing (coming soon)

- **Reward Functions**:
  - `returns` - Simple returns (default)
  - `sharpe` - Sharpe ratio
  - `sortino` - Sortino ratio
  - `pnl` - Profit and loss
  - `calmar` - Calmar ratio

- **Risk Management**:
  - Transaction costs (commission, slippage)
  - Position limits (max 95% of capital)
  - Episode length controls
  - One-trade mode for single-trade evaluation

- **Observation Normalization**:
  - VecNormalize for observation and reward scaling
  - Z-score normalization for features
  - Clip values to prevent outliers

### RL Algorithms (via Stable-Baselines3)

| Algorithm | Best For | Speed | Sample Efficiency |
|-----------|----------|-------|-------------------|
| **PPO** | Continuous control, stable training | Medium | Medium |
| **A2C** | Fast training, good baseline | Fast | Low |
| **DQN** | Discrete actions, sample efficient | Slow | High |
| **SAC** | Continuous actions, max entropy | Medium | High |

**Recommendation:** Start with PPO for most use cases.

### Technical Indicators

From the integrated tools module:

- **Trend**: SMA ratios (5, 20, 50, 200)
- **Volatility**: Range/Close ratio
- **Stationarity**: Fractional differentiation (d=0.4)
- **Normalization**: Rolling z-scores (no look-ahead bias)

### Reward Functions

Available reward types (configure in `env.environment_params.reward_type`):

```python
# Returns (default) - Simple percentage returns
reward = (portfolio_value - initial_balance) / initial_balance

# Sharpe Ratio - Risk-adjusted returns
reward = mean(returns) / std(returns)

# Sortino Ratio - Downside risk-adjusted
reward = mean(returns) / std(negative_returns)

# PnL - Absolute profit/loss
reward = portfolio_value - previous_value

# Calmar Ratio - Return over max drawdown
reward = total_return / max_drawdown
```

---

## Live Trading

### Live Trading Quick Start

**⚠️ IMPORTANT: Always test on testnet first!**

#### Prerequisites

1. Get Binance testnet API keys: https://testnet.binance.vision/
2. Set environment variables:
   ```bash
   export BINANCE_TESTNET_KEY="your_key"
   export BINANCE_TESTNET_SECRET="your_secret"
   ```

#### Step 1: Validate Your Model

**NEVER skip validation!**

```bash
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_*/best_model/best_model.zip \
  --days 1
```

This tests:
- ✓ Data loading from MinIO
- ✓ Dollar volume bar creation
- ✓ Feature computation
- ✓ Model predictions
- ✓ Simulated trading

#### Step 2: Run on Testnet

```bash
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO_returns_*/best_model/best_model.zip \
  --symbol BTCUSDT \
  --balance 10000
```

Features:
- Real-time dashboard with portfolio stats
- Live predictions (BUY/SELL/HOLD)
- Safety guards and circuit breakers
- Trade history logging to SQLite

#### Step 3: Analyze Results

```bash
uv run python examples/live_trading_example.py analyze \
  --db portfolio.db
```

Shows:
- Win rate and profit factor
- PnL and commission
- Trade distribution
- Exports to CSV

### Safety Features

Built-in protection:

- **Circuit Breakers**: Stop at 20% drawdown (configurable)
- **Rate Limits**: Max trades per hour/day
- **Position Limits**: Max 95% capital in positions
- **Consecutive Loss Limits**: Stop after 5 losses in a row
- **Trading Hours**: Restrict to specific hours (optional)
- **Balance Checks**: Stop if balance too low

### Deployment Workflow

```
1. Train Model → 2. Validate → 3. Test on Testnet → 4. Monitor → 5. Deploy Live
                     ↓              ↓                    ↓
              (Historical)    (Fake Money)        (Small Amount)
```

**Recommendation:** Run on testnet for 24-48 hours before going live.

---

## Checkpoint Management

### Training Configuration Embedding

**New Feature (2025-11-05):** Checkpoints now embed their complete training configuration!

Benefits:
- ✅ **Know exact features** the model expects
- ✅ **Offline capable** - No MLflow server needed
- ✅ **Feature validation** - Catch mismatches before deployment
- ✅ **Reproducibility** - Recreate exact training setup

### Model Discovery

Find all trained models:

```python
from pathlib import Path
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

# Discover all checkpoints
models = CheckpointManager.discover_all_checkpoints(Path("checkpoints"))

for model in models:
    print(f"{model['model_type']}: {model['checkpoint_dir'].name}")
    print(f"  Features: {model['observation_dim']}")
    print(f"  VecNormalize: {model['vecnormalize_path'] is not None}")
```

### Configuration Retrieval

Retrieve training config from any checkpoint:

```python
from pathlib import Path
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

# Load checkpoint config
checkpoint_path = Path("checkpoints/PPO_*/best_model/best_model.zip")
config = CheckpointManager.get_training_config(checkpoint_path)

print(f"Source: {config['source']}")  # 'embedded' (offline-capable)
print(f"Features: {config['observation']['input_features']}")
print(f"Reward: {config['env']['environment_params']['reward_type']}")

# Validate live features match training
expected_features = config['observation']['input_features']
if live_features != expected_features:
    raise ValueError(f"Feature mismatch! Expected {expected_features}")
```

Quick inspection:

```bash
# Show all available models and their configs
uv run python inspect_model.py

# Inspect specific model
uv run python inspect_model.py checkpoints/PPO_*/best_model/best_model.zip
```

---

## Project Structure

```
rl-trading-lab/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── agent/                 # PPO, A2C, DQN, SAC configs
│   │   ├── ppo.yaml
│   │   ├── a2c.yaml
│   │   └── dqn.yaml
│   ├── env/                   # Trading environment configs
│   │   ├── returns.yaml       # Returns reward
│   │   ├── sharpe.yaml        # Sharpe reward
│   │   └── sortino.yaml       # Sortino reward
│   ├── observation/           # Feature selection configs
│   └── feature_engineering/   # Indicator configs
│
├── src/rl_trading_lab/
│   ├── environment/           # Gym trading environment
│   │   ├── trading_env.py
│   │   ├── rewards.py
│   │   └── factory.py
│   ├── agents/                # RL agent wrappers
│   │   └── sb3_agents.py
│   ├── live/                  # Live trading components
│   │   ├── stream_consumer.py # WebSocket consumer
│   │   ├── inference.py       # Model predictions
│   │   ├── executor.py        # Order execution
│   │   ├── portfolio.py       # Portfolio tracking
│   │   ├── safety.py          # Risk management
│   │   └── dashboard.py       # Real-time UI
│   ├── data/                  # Data adapters
│   │   ├── binance_adapter.py # MinIO/Delta Lake
│   │   ├── bar_processor.py   # Dollar volume bars
│   │   └── feature_pipeline.py # Feature engineering
│   ├── utils/                 # Utilities
│   │   ├── checkpoint_manager.py # Model saving/loading
│   │   ├── callbacks.py       # Training callbacks
│   │   └── mlflow_logger.py   # MLflow integration
│   └── config/                # Pydantic config models
│
├── experiments/
│   ├── train.py               # Main training script
│   └── notebooks/             # Jupyter notebooks
│       └── 03_live_trading_tutorial.ipynb
│
├── examples/
│   ├── live_trading_example.py # Complete reference
│   ├── configs/               # Example configs
│   └── README.md              # Live trading guide
│
├── checkpoints/               # Saved models
├── mlruns/                    # MLflow experiments
├── logs/                      # TensorBoard logs
│
├── LIVE_TRADING_GUIDE.md      # Complete live trading docs
├── BINANCE_TESTNET_STATUS.md  # Project status
├── CHECKPOINT_CONFIG_FIX.md   # Config embedding docs
└── TESTING_GUIDE.md           # Testing instructions
```

---

## Configuration System

### Hydra Configuration

Override any config parameter via CLI:

```bash
# Change algorithm
uv run python experiments/train.py agent=a2c

# Change reward function
uv run python experiments/train.py env=sharpe

# Override specific values
uv run python experiments/train.py \
  agent.hyperparameters.learning_rate=0.001 \
  training.total_timesteps=500000 \
  env.environment_params.commission_rate=0.001
```

### Config Files

Main categories:

1. **Agent Configs** (`configs/agent/`)
   - Algorithm hyperparameters
   - Network architecture
   - Learning rates, batch sizes

2. **Environment Configs** (`configs/env/`)
   - Reward function selection
   - Transaction costs
   - VecNormalize settings

3. **Observation Configs** (`configs/observation/`)
   - Feature selection
   - Validation rules

4. **Training Configs** (`configs/training/`)
   - Total timesteps
   - Evaluation frequency
   - Checkpoint frequency

---

## Example Experiments

### Compare Reward Functions

```bash
# Sharpe ratio (risk-adjusted)
uv run python experiments/train.py \
  env=sharpe \
  experiment.run_name=sharpe_test

# Simple returns
uv run python experiments/train.py \
  env=returns \
  experiment.run_name=returns_test

# Sortino ratio (downside risk)
uv run python experiments/train.py \
  env=sortino \
  experiment.run_name=sortino_test
```

### Test Different Algorithms

```bash
# PPO (recommended)
uv run python experiments/train.py agent=ppo

# A2C (faster training)
uv run python experiments/train.py agent=a2c

# DQN (sample efficient)
uv run python experiments/train.py agent=dqn
```

### Hyperparameter Tuning

```bash
# Increase learning rate
uv run python experiments/train.py \
  agent.hyperparameters.learning_rate=0.001

# Larger network
uv run python experiments/train.py \
  agent.hyperparameters.policy_kwargs.net_arch.pi=[512,512] \
  agent.hyperparameters.policy_kwargs.net_arch.vf=[512,512]

# More training steps
uv run python experiments/train.py \
  training.total_timesteps=1000000
```

---

## Monitoring & Debugging

### Real-Time Monitoring

```bash
# Verbose training output
uv run python experiments/train.py logging.console.verbose=1

# Disable progress bar (for logging to file)
uv run python experiments/train.py logging.console.progress_bar=false
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Train with TensorBoard
uv run python experiments/train.py logging.tensorboard.enabled=true
```

View at http://localhost:6006

### MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000
```

Features:
- Compare experiment runs
- View hyperparameters and metrics
- Download trained models
- Track model lineage

View at http://localhost:5000

---

## Integration with Data Pipeline

### Using Kedro Output

```yaml
# configs/config.yaml
data:
  train_data_path: "../kedro-crypto-ind/data/08_reporting/ml_ready_features.parquet"
```

### Using dlt/MinIO Output

The system automatically loads from MinIO Delta Lake tables:

```python
from rl_trading_lab.data import BinanceDataAdapter

adapter = BinanceDataAdapter()
df = adapter.load_symbol_data(symbol="BTCUSDT", start_date=..., end_date=...)
```

---

## Live Trading Documentation

**Complete guides available:**

- **[LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md)** - Comprehensive system documentation
- **[examples/README.md](examples/README.md)** - Live trading examples and usage
- **[notebooks/03_live_trading_tutorial.ipynb](notebooks/03_live_trading_tutorial.ipynb)** - Interactive tutorial
- **[BINANCE_TESTNET_STATUS.md](BINANCE_TESTNET_STATUS.md)** - Project status and architecture

### Key Components

1. **StreamConsumer** - WebSocket connection to Binance
2. **BarProcessor** - Dollar volume bar creation
3. **FeatureComputer** - Real-time feature computation
4. **ModelInferenceEngine** - Model predictions
5. **OrderExecutor** - Order placement and management
6. **PortfolioManager** - Multi-symbol portfolio tracking
7. **SafetyGuard** - Risk management and circuit breakers
8. **TradingDashboard** - Real-time Rich UI

---

## Troubleshooting

### Missing Data

```bash
# Check data exists
ls tools/examples/btcusdt_fractional_indicators.parquet

# Create new data
cd tools
python examples/07_fractional_indicators.py
```

### GPU Support

```bash
# Install CUDA support
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Train with GPU
uv run python experiments/train.py experiment.device=cuda
```

### Memory Issues

```bash
# Reduce batch size
uv run python experiments/train.py agent.hyperparameters.batch_size=32

# Reduce buffer size (DQN only)
uv run python experiments/train.py agent.hyperparameters.buffer_size=10000
```

### Config Embedding Issues

If checkpoints don't have embedded configs:

```bash
# Validate checkpoint
uv run python validate_config_embedding.py

# View diagnostic guide
cat CHECKPOINT_CONFIG_FIX.md
```

---

## CLI Commands

### Training

```bash
# Basic training
uv run python experiments/train.py

# With specific config
uv run python experiments/train.py agent=ppo env=sharpe

# Override parameters
uv run python experiments/train.py training.total_timesteps=100000
```

### Live Trading

```bash
# Validate pipeline
uv run python examples/live_trading_example.py validate \
  --model <path> --days 1

# Run live trading
uv run python examples/live_trading_example.py trade \
  --model <path> --symbol BTCUSDT

# Analyze results
uv run python examples/live_trading_example.py analyze \
  --db portfolio.db
```

### Model Inspection

```bash
# List all models
uv run python inspect_model.py

# Inspect specific model
uv run python inspect_model.py <checkpoint-path>

# Validate config embedding
uv run python validate_config_embedding.py
```

---

## API Reference

### CheckpointManager

```python
from rl_trading_lab.utils.checkpoint_manager import CheckpointManager

# Discover models
models = CheckpointManager.discover_all_checkpoints(Path("checkpoints"))

# Load checkpoint
model, vec_env = CheckpointManager.load_checkpoint(path, env)

# Get training config
config = CheckpointManager.get_training_config(checkpoint_path)
```

### TradingEnv

```python
from rl_trading_lab.environment import TradingEnv

env = TradingEnv(
    df=dataframe,
    initial_balance=10000,
    commission_rate=0.001,
    reward_type='sharpe',
    one_trade_mode=False
)
```

### ModelInferenceEngine

```python
from rl_trading_lab.live import ModelInferenceEngine

engine = ModelInferenceEngine(
    model_path="checkpoints/.../best_model.zip",
    vecnormalize_path="checkpoints/.../vecnormalize.pkl"
)

action, confidence = engine.predict(features, deterministic=True)
```

---

## Resources

### Documentation
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Binance API](https://binance-docs.github.io/apidocs/spot/en/)

### Internal Guides
- [LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md) - Complete live trading documentation
- [CHECKPOINT_CONFIG_FIX.md](CHECKPOINT_CONFIG_FIX.md) - Config embedding implementation
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing and validation
- [examples/README.md](examples/README.md) - Live trading examples

### Related Projects
- [kedro-crypto-ind](../kedro-crypto-ind) - Data pipeline
- [dlt-starter](../dlt-starter) - Real-time data collection
- [tools](../tools) - Technical indicators

---

## Contributing

This is a personal research project, but suggestions are welcome! Areas for improvement:

- [ ] Additional reward functions
- [ ] Custom feature extractors
- [ ] Ensemble methods
- [ ] Multi-asset portfolio optimization
- [ ] Advanced order types (limit orders, stop-loss)
- [ ] Backtesting framework improvements

---

## License

MIT License - See LICENSE file for details

---

## Disclaimer

**⚠️ IMPORTANT RISK WARNING ⚠️**

- This software is for **educational and research purposes only**
- Trading cryptocurrencies involves **substantial risk of loss**
- Past performance does **NOT** indicate future results
- RL models can fail in new market conditions
- **Always test on testnet first**
- **Never invest more than you can afford to lose**
- The authors are **NOT responsible** for any financial losses

**USE AT YOUR OWN RISK**

---

## Quick Reference

### Most Common Commands

```bash
# Train model
uv run python experiments/train.py

# View results
mlflow ui

# Inspect model
uv run python inspect_model.py

# Test live feed
uv run python test_live_feed_with_config.py

# Deploy to testnet
uv run python examples/live_trading_example.py trade --model <path>
```

---

**Remember:** Start simple, test thoroughly, measure everything, and iterate based on results.

---

**Last Updated:** 2026-01-12
**Version:** 2.1.0
