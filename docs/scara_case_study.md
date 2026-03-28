# RL Trading Lab — Case Study

## Situation

Exploring reinforcement learning for systematic trading required reproducible experiment infrastructure across multiple RL agents, reward functions, and market environments. The goal was to compare PPO, A2C, DQN, and MaskablePPO across different hyperparameter configurations on crypto market data (Binance BTCUSDT), with the ability to eventually extend to equities and FX.

## Complication

Three problems blocked progress:

1. **No reproducibility.** Ad-hoc RL experiments couldn't be replicated. There was no way to compare hyperparameter sweeps or track model performance over time. An experiment that "worked yesterday" couldn't be reproduced today because the exact configuration, data split, and random seed weren't captured.

2. **Tangled architecture.** The environment logic — position management, commission calculation, slippage simulation, risk management (stop-loss, drawdown limits) — was mixed directly into the Gymnasium wrapper. This made it impossible to reuse the trading simulation for different asset classes, or to use the RL training infrastructure with non-trading environments.

3. **No safety guarantees.** Risk management was applied as post-hoc filtering instead of being enforced at the environment level. An agent could take catastrophic positions during training without any guardrails, and there was no standard way to enforce drawdown limits, consecutive loss stops, or position size caps.

## Action

Built a modular RL framework using Domain-Driven Design principles:

**Pure domain layer** (zero numpy/pandas/gym dependencies):
- `TradingDomain` orchestrator with immutable value objects (`Position`, `CompletedTrade`, `Bar`, `FeatureWindow`)
- Protocol-based pluggable services: reward calculation (returns, PnL, risk-adjusted), risk management (standard, conservative, none), position sizing (fixed percentage, Kelly Criterion)
- `MarketDataPort` interface isolating data access from domain logic

**Composable RL trainer** (reusable beyond trading):
- `Trainer` — slim 230-line orchestrator that takes any SB3 algorithm + any Gym environment
- `EnvWrapperBuilder` — Monitor/VecNormalize chain builder
- `CallbackFactory` — lazy-imported MLflow/TensorBoard integration (optional, not required)
- Demonstrated working on CartPole to prove framework independence

**Pluggable data layer:**
- `DataLoaderPort` with Parquet, CSV, and Delta Lake adapters
- `FeatureEngineeringPort` with crypto-specific and passthrough implementations
- Factory functions resolve config to the correct adapter at runtime

**Reproducible experiment infrastructure:**
- 8 Hydra agent configurations (PPO, A2C, DQN, MaskablePPO, 3 Transformer variants, DQN-aggressive)
- MLflow tracking for every metric, parameter, and artifact
- Full config YAML + git commit hash logged per run

**Live deployment pipeline:**
- Binance WebSocket real-time data streaming
- Multi-layer safety: circuit breakers, rate limits, position size caps, drawdown stops
- Rich terminal dashboard for live monitoring

## Result

- **52 trained models** across 4 agent architectures (PPO, A2C, DQN, MaskablePPO)
- **79 tracked experiments** with full metric comparison in MLflow
- **13,700 lines** of production Python — type-safe, tested, documented
- **Reusable RL module** demonstrated on standard Gym environments (CartPole)
- **8 config templates** for reproducible experiments via Hydra CLI overrides
- **Multi-layer safety** enforced at environment level: 30% max drawdown, 80% stop-loss, consecutive loss limits, minimum holding periods

## Artifacts

- MLflow dashboard showing 79 experiment runs with training curves, hyperparameters, and agent comparisons
- 5-panel episode analysis (Price, Actions, Positions, Rewards, Portfolio Value)
- CartPole reusability demo proving framework independence
- Architecture diagram showing clean domain separation
- Live trading dashboard with Rich terminal UI (Binance testnet)

## Tech Stack

Python 3.12 | Stable-Baselines3 | Gymnasium | PyTorch | MLflow | Hydra | Pydantic | Rich | Binance API | Delta Lake
