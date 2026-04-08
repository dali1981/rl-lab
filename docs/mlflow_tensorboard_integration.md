# MLflow and TensorBoard Integration

**Date**: 2025-10-27
**Status**: Historical implementation note (pre-trainer consolidation)

> **Canonical runtime note (April 9, 2026):**
> - Primary training path is `experiments/train.py` -> `runtime/training_entrypoint.py` -> `TrainAgentUseCase`.
> - Authoritative trainer surfaces are `src/rl_trading_lab/agents/trainer.py` and `src/rl_trading_lab/agents/trainer_factory.py`.
> - Any `sb3_agents.py` references in this document are legacy context, not the current primary interface.

## Overview

Integrated MLflow and TensorBoard logging with Stable-Baselines3 using the official Logger system. All training metrics are now automatically logged to both platforms simultaneously.

---

## Implementation

### 1. MLflow Output Format (`src/utils/mlflow_logger.py`)

Created custom `MLflowOutputFormat` class that extends `KVWriter`:

```python
class MLflowOutputFormat(KVWriter):
    """
    Custom logger for MLflow integration with SB3.

    Automatically logs all SB3 metrics (rollout/, train/, eval/) to MLflow.
    """

    def write(self, key_values, key_excluded, step=0):
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # Skip if excluded from MLflow
            if excluded is not None and "mlflow" in excluded:
                continue

            # Log scalar values only
            if isinstance(value, np.ScalarType) and not isinstance(value, str):
                mlflow.log_metric(key, value, step)
```

**Features**:
- Implements SB3's `KVWriter` interface
- Filters metrics based on exclusion rules
- Logs only scalar values (not strings or complex types)
- Automatically called by SB3 during training

---

### 2. Logger Setup (legacy path: `src/agents/sb3_agents.py`)

Added `_setup_logger()` method to configure both MLflow and TensorBoard:

```python
def _setup_logger(self):
    """
    Configure SB3 logger with MLflow and TensorBoard outputs.
    """
    tensorboard_log = self.config.get("tensorboard_log", None)
    output_formats = []

    # Add MLflow output if active run
    if mlflow.active_run():
        output_formats.append(MLflowOutputFormat())
        logger.info("MLflow logging enabled")

    # Configure with TensorBoard support
    if tensorboard_log:
        new_logger = configure(tensorboard_log, output_formats)
        self.agent.set_logger(new_logger)
        logger.info(f"TensorBoard logging enabled: {tensorboard_log}")
    elif output_formats:
        new_logger = Logger(folder=None, output_formats=output_formats)
        self.agent.set_logger(new_logger)
```

**How it works**:
1. Checks if MLflow run is active
2. Creates MLflowOutputFormat if yes
3. Configures SB3 logger with both TensorBoard and MLflow outputs
4. SB3 automatically writes to both platforms

---

### 3. Trading Metrics Logging (legacy path: `src/agents/sb3_agents.py`)

Updated `TradingMetricsCallback` to log trading-specific metrics:

```python
class TradingMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        # ... episode detection ...

        if "sharpe" in info:
            # Log to both MLflow and TensorBoard via SB3 logger
            if self.logger:
                self.logger.record("trading/sharpe", info["sharpe"])
                self.logger.record("trading/total_return", info["total_return"])
                self.logger.record("trading/max_drawdown", info["max_drawdown"])
                self.logger.record("trading/portfolio_value", info["portfolio_value"])

        return True
```

**Logged Metrics**:
- `trading/sharpe` - Sharpe ratio
- `trading/total_return` - Total return %
- `trading/max_drawdown` - Maximum drawdown %
- `trading/portfolio_value` - Current portfolio value

---

### 4. Manual Logging for Test Metrics (`experiments/train.py`)

Kept manual logging for post-training evaluation:

```python
# Log final test metrics to MLflow
# Note: Training metrics are automatically logged by SB3 logger integration
# These are final evaluation metrics computed after training completes
if mlflow.active_run():
    # Add test/ prefix to distinguish from training metrics
    test_metrics_prefixed = {f"test/{k}": v for k, v in test_metrics.items()}
    mlflow.log_metrics(test_metrics_prefixed)
    mlflow.log_dict(backtest_results, "backtest_results.json")

# Backtest metrics
if mlflow.active_run():
    mlflow.log_metrics({
        "backtest/final_return": final_return,
        "backtest/sharpe_ratio": sharpe,
        "backtest/num_trades": total_trades,
        "backtest/trade_frequency": trade_frequency,
    })
```

**Why manual logging?**:
- These metrics are computed AFTER training completes
- Not part of the training loop
- SB3 logger only handles training-time metrics

---

## Metrics Logged Automatically

### From SB3 (via Logger)

All standard SB3 metrics are logged automatically:

**Rollout Metrics** (`rollout/`):
- `ep_len_mean` - Average episode length
- `ep_rew_mean` - Average episode reward

**Training Metrics** (`train/`):
- `learning_rate` - Current learning rate
- `loss` - Total loss
- `policy_loss` - Policy gradient loss
- `value_loss` - Value function loss
- `entropy_loss` - Entropy regularization loss
- `approx_kl` - Approximate KL divergence
- `clip_fraction` - Fraction of clipped updates
- `explained_variance` - Explained variance of value function
- `n_updates` - Number of gradient updates

**Evaluation Metrics** (`eval/`):
- `mean_reward` - Mean reward during evaluation
- `mean_ep_length` - Mean episode length during evaluation

### Trading Metrics (via Callback)

Custom trading metrics logged via `TradingMetricsCallback`:

**Trading Metrics** (`trading/`):
- `sharpe` - Sharpe ratio from environment
- `total_return` - Total return %
- `max_drawdown` - Maximum drawdown %
- `portfolio_value` - Current portfolio value

### Test/Backtest Metrics (Manual)

Final evaluation metrics logged after training:

**Test Metrics** (`test/`):
- All metrics from `evaluate_final_performance()`
- Includes mean_reward, std_reward, min_reward, max_reward, etc.

**Backtest Metrics** (`backtest/`):
- `final_return` - Final return %
- `sharpe_ratio` - Sharpe ratio calculated from returns
- `num_trades` - Total number of trades
- `trade_frequency` - Percentage of steps with trades

---

## How to Use

### 1. Enable in Configuration

Ensure TensorBoard logging is enabled in agent config:

```yaml
# configs/agent/ppo.yaml
tensorboard_log: "./runs"
```

### 2. Enable MLflow

Ensure MLflow is configured:

```yaml
# configs/logging/default.yaml
mlflow:
  enabled: true
  tracking_uri: "./mlruns"
  experiment_name: "trading-rl"
```

### 3. Run Training

```bash
python experiments/train.py
```

### 4. View Logs

**TensorBoard**:
```bash
tensorboard --logdir=./runs
```

**MLflow**:
```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## Benefits

1. **Automatic Logging**: All SB3 metrics logged without manual intervention
2. **Dual Platform**: Same metrics logged to both MLflow and TensorBoard
3. **Organized Namespaces**: Metrics grouped by category (rollout/, train/, eval/, trading/, test/, backtest/)
4. **Official Pattern**: Follows SB3 documentation recommendations
5. **Extensible**: Easy to add new metrics via logger.record()
6. **Clean Code**: No manual mlflow.log_metrics() in training loop

---

## Architecture

```
Training Loop
    ↓
SB3 Agent (PPO/A2C/etc.)
    ↓
Logger.record()
    ↓
    ├─→ TensorBoard Writer → ./runs/
    └─→ MLflowOutputFormat → mlflow.log_metric() → ./mlruns/

TradingMetricsCallback
    ↓
self.logger.record("trading/...")
    ↓
    ├─→ TensorBoard Writer → ./runs/
    └─→ MLflowOutputFormat → mlflow.log_metric() → ./mlruns/
```

---

## Key Files Modified

1. **`src/utils/mlflow_logger.py`** - New file
   - Implements MLflowOutputFormat class

2. **`src/agents/sb3_agents.py`**
   - Added _setup_logger() method
   - Updated TradingMetricsCallback to log portfolio_value
   - Added imports for Logger and MLflowOutputFormat

3. **`experiments/train.py`**
   - Added comments explaining manual logging
   - Added test/ and backtest/ prefixes for clarity

---

## Troubleshooting

### Metrics not appearing in MLflow

**Check**:
1. Is MLflow enabled? (`cfg.logging.mlflow.enabled = true`)
2. Is there an active run? (Check console for "MLflow tracking enabled")
3. Is the metric being recorded during training? (Check TensorBoard first)

### TensorBoard not showing metrics

**Check**:
1. Is tensorboard_log set in agent config?
2. Are you looking at the correct run folder?
3. Try refreshing the TensorBoard UI

### Trading metrics missing

**Check**:
1. Are trading metrics in environment's info dict?
2. Is TradingMetricsCallback being used?
3. Check environment implementation (sharpe, total_return, etc.)

---

## Future Enhancements

Possible improvements:

1. **Add more trading metrics**:
   - Win rate
   - Profit factor
   - Average trade duration
   - Position distribution

2. **Log model artifacts**:
   - mlflow.log_artifact(model_path)
   - Save best model to MLflow

3. **Add custom plots**:
   - Equity curve
   - Drawdown chart
   - Action distribution

4. **Hyperparameter tracking**:
   - Log all hyperparameters to MLflow params
   - Track hyperparameter sweeps

---

## References

- **SB3 Logger Documentation**: https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
- **SB3 MLflow Integration**: https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html#mlflow
- **MLflow Tracking**: https://mlflow.org/docs/latest/tracking.html
- **TensorBoard**: https://www.tensorflow.org/tensorboard

---

**Last Updated**: 2025-10-27
