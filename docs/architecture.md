# Architecture Overview

Related policy docs:
- [Architecture rules](architecture_rules.md) (authoritative enforcement baseline)
- [Production boundary & supported modes](production_boundary.md)

The system is structured around clear separation of concerns:

```
   Market Data
        │
        ▼
┌──────────────┐
│ Environment  │  ← state, actions, reward
└──────────────┘
        │
        ▼
┌──────────────┐
│ RL Agent     │  ← PPO / DQN / SAC (SB3)
└──────────────┘
        │
        ▼
┌──────────────┐
│ Trainer      │  ← rollout, optimization
└──────────────┘
        │
        ├── Metrics → MLflow
        ├── Configs → Hydra
        └── Checkpoints

    (Optional)
        ▼
┌──────────────┐
│ Execution    │  ← paper / live trading with guards
└──────────────┘
```

---

## Design principles

- **Explicit environment modeling** - Custom Gym environments with clear state/action/reward definitions
- **Config-driven experimentation** - All hyperparameters managed through Hydra YAML configs
- **Reproducibility over ad-hoc scripts** - Every experiment is tracked and reproducible
- **Safety-first deployment** - Circuit breakers, position limits, and risk controls for live trading

### Canonical Environment Implementation

Canonical environment runtime:
- `TradingDomain` (`src/rl_trading_lab/domain/trading_domain.py`)
- `GymTradingEnvAdapter` (`src/rl_trading_lab/infrastructure/adapters/gym_adapter.py`)

Legacy path:
- `src/rl_trading_lab/environment/trading_env.py` is deprecated/read-only compatibility surface.
- No new environment features should be added to the legacy path.

---

## Component Overview

### Data Layer

```
Raw Ticks → Dollar Volume Bars → Technical Indicators → Z-Score Normalization → ML-Ready Features
```

- **Bar Processor**: Converts tick data to dollar volume bars
- **Feature Pipeline**: Computes technical indicators (SMA ratios, volatility, fractional differentiation)
- **Normalization**: Rolling z-scores without look-ahead bias

### Training Layer

| Component | Purpose |
|-----------|---------|
| `TradingDomain` + `GymTradingEnvAdapter` | Canonical environment runtime and Gymnasium integration |
| `SB3Agents` | Wrapper for Stable-Baselines3 algorithms |
| `CheckpointManager` | Model persistence with embedded configs |
| `MLflowCallback` | Experiment tracking and metrics logging |

### Execution Layer (Optional)

| Component | Purpose |
|-----------|---------|
| `StreamConsumer` | WebSocket connection to exchange |
| `FeatureComputer` | Real-time feature computation |
| `ModelInferenceEngine` | Model predictions |
| `SafetyGuard` | Risk management and circuit breakers |
| `OrderExecutor` | Order placement and management |

---

## Directory Structure

```
rl-trading-lab/
├── configs/                    # Hydra configuration files
│   ├── agent/                 # PPO, A2C, DQN, SAC configs
│   ├── env/                   # Environment configs
│   └── observation/           # Feature selection
│
├── src/rl_trading_lab/
│   ├── environment/           # Gym trading environment
│   ├── agents/                # RL agent wrappers
│   ├── live/                  # Live trading components
│   ├── data/                  # Data adapters
│   └── utils/                 # Utilities
│
├── experiments/               # Training scripts
├── checkpoints/               # Saved models
└── mlruns/                    # MLflow experiments
```

---

## Data Flow

### Training

```
1. Load historical data (Parquet)
2. Initialize TradingDomain and wrap with GymTradingEnvAdapter
3. Train agent with SB3
4. Log metrics to MLflow
5. Save checkpoint with embedded config
```

### Live Trading

```
1. Connect to exchange WebSocket
2. Accumulate ticks → Dollar bars
3. Compute features (matching training)
4. Run model inference
5. Execute order (with safety checks)
6. Update portfolio state
```

---

## Key Design Decisions

### Self-contained Checkpoints

Each model checkpoint embeds its complete training configuration. This enables:
- Offline deployment without MLflow server
- Feature validation before inference
- Exact reproduction of training setup

### Action Masking

The environment supports action masking via `sb3-contrib.MaskablePPO` to prevent invalid actions (e.g., selling when not holding).

### Reward Functions

Multiple reward functions available:
- `returns` - Simple percentage returns
- `sharpe` - Risk-adjusted (Sharpe ratio)
- `sortino` - Downside risk-adjusted
- `pnl` - Absolute profit/loss

---

## Integration Points

| System | Integration |
|--------|-------------|
| **Binance** | WebSocket streaming, REST API for orders |
| **MinIO** | Delta Lake storage for historical data |
| **MLflow** | Experiment tracking and model registry |
| **Hydra** | Configuration management |
| **TensorBoard** | Training visualization |
