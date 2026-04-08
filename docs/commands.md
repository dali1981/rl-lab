# Canonical Command Matrix

This document is the single source of truth for RL Trading Lab CLI commands.

Scope date: April 8, 2026.

## Contract

- Use only the commands in this matrix for user-facing docs.
- Hydra overrides must use current keys (for example `training.total_timesteps`, `data.train_data_path`).
- Deprecated legacy override variants are not part of the supported contract.

## Command Matrix

| Use case | Command | Expected result |
|---|---|---|
| Smoke test (clean install) | `uv run python experiments/train.py data.train_data_path=sample_data/btcusdt_sample_10k.parquet training.total_timesteps=16 training.eval_freq=8 training.save_freq=16 training.n_eval_episodes=1 logging.mlflow.enabled=false logging.tensorboard.enabled=false logging.console.progress_bar=false` | Training exits successfully and writes `checkpoints/final_model/model.zip` plus `checkpoints/best_model/best_model.zip`. |
| Local training | `uv run python experiments/train.py data.train_data_path=sample_data/btcusdt_sample_10k.parquet training.total_timesteps=100000` | Full local training run with checkpoints and MLflow logging according to config. |
| Evaluation (within canonical training flow) | `uv run python experiments/train.py data.train_data_path=sample_data/btcusdt_sample_10k.parquet training.total_timesteps=10000 training.eval_freq=1000 training.n_eval_episodes=5` | Periodic evaluation metrics are produced during training. |
| Canonical wrapper help | `uv run python run_pipeline.py --help` | Wrapper delegates to canonical training CLI surface and exits cleanly. |
| Pre-live validation surface | `uv run python examples/live_trading_example.py validate --help` | Validate CLI help renders and exits cleanly. |
| Paper trading surface | `uv run python examples/live_trading_example.py trade --help` | Trade CLI help renders and exits cleanly. |
| Session analysis surface | `uv run python examples/live_trading_example.py analyze --help` | Analyze CLI help renders and exits cleanly. |
| MLflow UI | `uv run mlflow ui --port 5000` | MLflow UI starts on `http://localhost:5000`. |
| Notebook inspection | `uv run jupyter lab notebooks/debug_episode.ipynb` | Jupyter Lab starts with the notebook workspace. |

## Notes

- Commands that launch interactive services (`mlflow ui`, `jupyter lab`) are long-running by design.
- Paper/live-facing commands require environment prerequisites (credentials, data, models) for full execution; `--help` checks are the canonical smoke surface in CI-style validation.
