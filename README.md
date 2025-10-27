# 🚀 RL Trading Lab

A **hybrid Kedro + RL** framework for cryptocurrency trading with reinforcement learning. This project combines Kedro's excellent data pipeline capabilities with a streamlined RL training environment.

## 📋 Overview

This project implements a clean separation of concerns:
- **Kedro** handles data preparation (tick data → bars → indicators → normalization)
- **RL Lab** handles agent training and experimentation
- **Hydra** manages configurations (much simpler than Kedro configs for RL)
- **MLflow** tracks experiments and results

## 🏗️ Architecture

```
Data Flow:
Raw Ticks → [Kedro Pipeline] → ML-Ready Features → [RL Lab] → Trained Models
                    ↓                                  ↓
            Dollar Volume Bars                   Trading Strategies
            Technical Indicators                 Performance Metrics
            Z-Score Normalization
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to project
cd /Users/mohamedali/trading_project/rl-trading-lab

# Install dependencies with UV
uv venv
uv sync

# Activate environment
source .venv/bin/activate  # On macOS/Linux
```

### 2. Prepare Data (Using Your Existing Pipeline)

Your data is already prepared in `tools/examples/btcusdt_fractional_indicators.parquet`!

If you want to use Kedro pipeline output instead:
```bash
cd ../kedro-crypto-ind
kedro run --pipeline=feature_engineering
# This creates data/08_reporting/ml_ready_features.parquet
```

### 3. Train Your First Agent

```bash
# Train PPO with default settings
uv run python experiments/train.py

# Train with different agent
uv run python experiments/train.py agent=a2c

# Change reward function
uv run python experiments/train.py env.reward_type=sortino

# Use z-score normalized features
uv run python experiments/train.py features.use_zscore=true
```

### 4. View Results

```bash
# Start MLflow UI
mlflow ui --port 5000

# Open browser to http://localhost:5000
```

## 🎯 Key Features

### Environment Features
- **Discrete & Continuous Actions**: Buy/Sell/Hold or position sizing
- **Multiple Reward Functions**: Sharpe, Sortino, Returns, PnL
- **Transaction Costs**: Commission and slippage modeling
- **Risk Management**: Stop-loss, take-profit, max drawdown

### Indicators Available (From Your Tools Module)
- **Price Ratios**: SMA/Close ratios (normalized trend indicators)
- **Volatility**: Range/Close ratio
- **Stationarity**: Fractional differentiation (d=0.4)
- **Z-Scores**: No look-ahead bias normalization

### RL Algorithms (via Stable-Baselines3)
- **PPO**: Best for continuous control, stable training
- **A2C**: Faster than PPO, good baseline
- **DQN**: Discrete actions only, sample efficient
- **SAC**: Continuous actions, maximum entropy

## 📁 Project Structure

```
rl-trading-lab/
├── configs/                 # Hydra configurations
│   ├── config.yaml         # Main config
│   ├── agent/             # PPO, A2C, DQN configs
│   ├── env/               # Trading environment configs
│   └── features/          # Feature engineering configs
├── src/
│   ├── environment/       # Gym trading environment
│   ├── agents/           # RL agent wrappers
│   └── utils/            # Data loading, metrics
├── experiments/
│   ├── train.py          # Main training script
│   ├── backtest.py       # Backtesting (TODO)
│   └── optimize.py       # Hyperparameter search (TODO)
├── mlruns/               # MLflow experiment tracking
└── checkpoints/          # Saved models
```

## 🧪 Example Experiments

### Experiment 1: Compare Reward Functions
```bash
# Sharpe ratio reward
uv run python experiments/train.py env.reward_type=sharpe experiment.run_name=sharpe_test

# Simple returns reward
uv run python experiments/train.py env.reward_type=returns experiment.run_name=returns_test

# PnL reward
uv run python experiments/train.py env.reward_type=pnl experiment.run_name=pnl_test
```

### Experiment 2: Feature Sets
```bash
# Use only fractional differentiation
uv run python experiments/train.py features.technical_indicators=[fracdiff_0.4]

# Use z-score normalized indicators
uv run python experiments/train.py features.use_zscore=true

# Add return features
uv run python experiments/train.py features.feature_engineering.add_returns=true
```

### Experiment 3: Agent Hyperparameters
```bash
# Increase learning rate
uv run python experiments/train.py agent.hyperparameters.learning_rate=0.001

# Larger network
uv run python experiments/train.py agent.hyperparameters.policy_kwargs.net_arch.pi=[512,512]

# More training steps
uv run python experiments/train.py training.total_timesteps=500000
```

## 🔄 Integration with Kedro

To use Kedro pipeline outputs directly:

1. Update data path in config:
```yaml
# configs/config.yaml
data:
  train_data_path: "../kedro-crypto-ind/data/08_reporting/ml_ready_features.parquet"
```

2. Or load from Kedro catalog programmatically:
```python
from src.utils.data_loader import load_kedro_catalog_data

df = load_kedro_catalog_data(
    catalog_name="ml_ready_features",
    kedro_project_path="../kedro-crypto-ind"
)
```

## 📊 Monitoring & Debugging

### Training Progress
```python
# Watch training in real-time
uv run python experiments/train.py logging.console.verbose=true

# Disable progress bar for logging
uv run python experiments/train.py logging.console.progress_bar=false
```

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir logs/

# Training with TensorBoard logging
uv run python experiments/train.py logging.tensorboard.enabled=true
```

### MLflow Tracking
- Experiments: http://localhost:5000
- Compare runs, metrics, hyperparameters
- Download trained models

## 🎓 Next Steps

1. **Add More Indicators**: Extend feature engineering
2. **Custom Reward Functions**: Implement in `src/environment/rewards.py`
3. **Ensemble Methods**: Train multiple agents and combine
4. **Live Trading**: Connect to exchange APIs
5. **Hyperparameter Optimization**: Use Optuna for systematic search

## 🐛 Troubleshooting

### Missing Data
```bash
# Check data path
ls ../tools/examples/btcusdt_fractional_indicators.parquet

# Or create new data
cd ../tools
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

# Reduce buffer size (for DQN)
uv run python experiments/train.py agent.hyperparameters.buffer_size=10000
```

## 📚 Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Your Indicators Module](../tools/README.md)

## 🎯 Why This Approach Works

1. **No Framework Lock-in**: Each tool does what it's best at
2. **Fast Iteration**: Change configs, not code
3. **Reproducible**: All experiments tracked in MLflow
4. **Scalable**: Easy to add new algorithms, features, or data sources
5. **Production Ready**: Clean path from research to deployment

---

**Happy Trading! 🚀📈**

Remember: Start simple, measure everything, and iterate based on results!