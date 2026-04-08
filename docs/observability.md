# Observability Contract

This document is the authoritative observability reference for RL Trading Lab.

Scope date: April 8, 2026.

## Scope

In scope:
- canonical training observability (`experiments/train.py` -> runtime -> use case)
- MLflow run structure (params, metrics, artifacts)
- pre-live/live integration observability surfaces
- failure signals and first-triage locations

Out of scope:
- new logging/metrics implementation
- alerting/dashboard infrastructure changes

## Canonical Observability Surfaces

| Surface | Source | What it provides |
|---|---|---|
| Console + Hydra run log | `experiments/train.py` with `hydra.run.dir` | Runtime lifecycle events, environment creation, checkpoint save events, completion/failure messages |
| MLflow tracking | `logging.mlflow.*` via `create_mlflow_tracker()` | Experiment run metadata, flattened params, final training metrics, run status |
| Local checkpoint artifacts | `training.save_path` via `CheckpointService` | `final_model/model.zip`, `final_model/vecnormalize.pkl`, `final_model/metadata.json`, eval/best/checkpoint files |
| Pre-live/live runtime logs | `src/rl_trading_lab/runtime/live_entrypoint.py` and `src/rl_trading_lab/live/*` | Predictions, trade decisions, portfolio/safety state, connection/execution stats |

## Key Training Log Fields

Canonical training logs use the Hydra log format:

`[timestamp][logger_name][level] - message`

Core logger namespaces to watch first:
- `rl_trading_lab.infrastructure.adapters.mlflow_tracker`
- `rl_trading_lab.application.use_cases.train_agent`
- `rl_trading_lab.application.services.environment_service`
- `rl_trading_lab.application.services.agent_service`
- `rl_trading_lab.application.services.checkpoint_service`

High-signal message examples:
- `MLflow tracker initialized: ...`
- `Started MLflow run: ...`
- `Loaded train data: <rows> rows`
- `Creating <algorithm> agent...`
- `Starting training for <timesteps> timesteps...`
- `Saved model to .../final_model/model.zip`
- `Training completed in ...s`
- `Ended MLflow run with status: FINISHED|FAILED`

## Training Metrics Contract

### Console/SB3 metrics (during training)

From Eval/rollout output blocks:
- `eval/mean_ep_length`
- `eval/mean_reward`
- `rollout/exploration_rate`
- `time/total_timesteps`

### MLflow metrics (canonical run-level)

From `TrainAgentUseCase.execute()` final metric logging:
- `training_time_seconds`
- `total_timesteps`

### MLflow params (canonical run-level)

Run parameters are flattened and include:
- base training/environment fields: `algorithm`, `policy`, `reward_type`, `total_timesteps`, `initial_balance`, `commission_rate`, `lookback_window`, `max_drawdown_pct`
- agent hyperparameters with `agent.` prefix (for example `agent.learning_rate`, `agent.batch_size`, `agent.gamma`)

## Live/Pre-Live Metrics Contract

Current live integration surfaces expose the following metric groups.

### Portfolio metrics (`PortfolioManager.get_stats()`)

- `initial_balance`, `cash_balance`, `position_value`, `total_value`
- `realized_pnl`, `unrealized_pnl`, `total_pnl`
- `returns`, `drawdown`
- `total_commission`, `total_trades`, `active_positions`
- `positions.<symbol>.*` (quantity, entry/current price, value, unrealized PnL)

### Safety metrics (`SafetyGuard.get_stats()`)

- `state`
- `balance`, `peak_balance`
- `drawdown`, `drawdown_pct`
- `consecutive_losses`
- `trades_last_hour`, `trades_today`
- `violations`, `last_violation`

### Inference metrics (`ModelInferenceEngine.get_stats()`)

- `model_type`, `model_path`
- `total_predictions`
- `action_counts`
- `action_distribution`

### Execution metrics (`OrderExecutor.get_stats()`)

- `total_orders`, `successful_orders`, `failed_orders`
- `success_rate`
- `total_commission`
- `active_positions`

## MLflow Run Structure

With file-based tracking (`logging.mlflow.tracking_uri=file:<path>`), a canonical run is structured as:

```text
<tracking_uri>/
  <experiment_id>/
    meta.yaml
    <run_id>/
      meta.yaml
      params/
      metrics/
      tags/
      artifacts/
```

Contract notes:
- `params/` contains one file per parameter key.
- `metrics/` contains one file per metric key with timestamp/value/step rows.
- `tags/` includes standard MLflow tags (run name, user, source metadata).
- `artifacts/` directory exists per run. In the current canonical training path, tracker artifact APIs are available but not invoked by `TrainAgentUseCase`, so run artifacts may be empty.
- Training artifacts are still produced locally under `training.save_path` by `CheckpointService`.

## Failure Signals And Where To Look First

| Symptom | First place to check | Typical cause |
|---|---|---|
| Command exits before run starts | CLI stderr + `.hydra/overrides.yaml` | Bad Hydra key/override or config mismatch |
| No MLflow run created | `train.log` lines from `mlflow_tracker` | MLflow disabled, bad tracking URI, or init/start failure |
| Training starts but crashes | `train.log` line `Training failed: ...` from `train_agent` | Data/schema/env/agent runtime exception |
| No model files in save path | `checkpoint_service` log lines + `training.save_path` | Callback/save path config mismatch or interrupted run |
| Run exists but missing expected metrics | MLflow run `metrics/` and `params/` files | Early crash before final metric logging |
| Live loop unstable or halted | `runtime/live_entrypoint.py` logs + safety stats | Safety guard trigger, missing credentials, inference/feature errors |

## Canonical Workflow Smoke For Observability

Use a small bounded run and inspect logs + MLflow filesystem:

```bash
uv run python experiments/train.py \
  data.train_data_path=sample_data/btcusdt_sample_10k.parquet \
  training.total_timesteps=16 \
  training.eval_freq=8 \
  training.save_freq=16 \
  training.n_eval_episodes=1 \
  logging.mlflow.enabled=true \
  logging.tensorboard.enabled=false \
  logging.console.progress_bar=false
```

Verification checklist:
- run log contains start/run/train/save/end milestones
- MLflow run has `params/` and `metrics/` populated
- local save path has final and best model artifacts
