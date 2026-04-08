# Production Boundary and Supported Modes

This document is the authoritative boundary for what RL Trading Lab supports today.

As of April 8, 2026, this repository is a research and engineering lab with controlled execution demonstrations. It is not an unattended production trading system.

## Tier definitions

| Tier | Meaning |
|---|---|
| Production-ready | Stable and supported for intended lab usage in this repository with clear operating constraints. |
| Beta/controlled | Functional but requires operator control, explicit gating, and close monitoring. |
| Internal/experimental | Useful for development, demos, and architecture exploration; not a supported production surface. |

### Internal/experimental surfaces in this repository

The `internal/experimental` tier maps to concrete, non-authoritative surfaces such as:
- demo collateral and marketing-oriented plans (for example `DEMO_PLAN.md`, `docs/upwork_demo_execution_plan.md`),
- architecture review/refactor analysis docs (for example `docs/ddd_clean_architecture_review.md`, `docs/refactor_2024-11-29.md`, `docs/rl_module_reusability_assessment.md`),
- debugging and exploratory analysis notes (for example `docs/debugging/*`, notebooks under `notebooks/`),
- convenience wrappers or exploratory orchestration paths not treated as canonical runtime contracts (for example `run_pipeline.py`).

## Supported modes

| Mode | Tier | Supported today | Gated constraints | Prerequisites |
|---|---|---|---|---|
| Research | Production-ready | Yes, for iterative research and architecture work. | No unattended automation claims; results are exploratory by default. | Python/uv environment, repository dependencies installed, local configs. |
| Offline training | Production-ready | Yes, through the canonical training path. | Requires valid local dataset path and config compatibility; command overrides must match current Hydra schema. | `uv sync`, data file available at configured path, write access for checkpoints/logs. |
| Evaluation | Production-ready | Yes, as part of canonical training/eval workflow and offline validation surfaces. | Standalone "production evaluator" service is not provided; evaluation is bounded by current scripts and checkpoint compatibility. | Trained checkpoint, compatible feature/config schema, evaluation data availability. |
| Paper trading (testnet) | Beta/controlled | Yes, via `examples/live_trading_example.py trade` on Binance testnet. | Human-in-the-loop operation required; explicit drawdown/position/rate limits required; testnet only by default policy. | Binance testnet credentials, validated model, safety guard parameters, active monitoring session. |
| Pre-live validation (historical/integration) | Beta/controlled | Yes, via historical pipeline validation (`validate`) and controlled integration checks. | Validation is pre-live gating only; passing validation is not permission for unattended live execution. | Trained model + optional VecNormalize, historical data access, feature compatibility checks. |

## Canonical runtime path

Preferred architecture direction remains:

```text
CLI -> config -> use_case -> services -> domain -> adapters
```

For current operation in this repository, canonical execution surfaces are:
- Training/evaluation orchestration: `experiments/train.py`
- Controlled integration validation and paper trading demo: `examples/live_trading_example.py`

`run_pipeline.py` is a convenience wrapper and not an authoritative production boundary definition.

## Command matrix (current repository reality)

Use these as the primary command surfaces when discussing supported modes.

### Research / Offline training / Evaluation

```bash
uv run python experiments/train.py --help
```

Typical execution (requires valid local data path in config):

```bash
uv run python experiments/train.py training.total_timesteps=1000
```

Notes:
- `trainer.max_steps` is not a valid current override key.
- `env.dataset=sample` is not a valid current override key.
- Evaluation metrics are integrated in the training workflow rather than a separate production evaluator command.

### Pre-live validation / Paper trading

```bash
uv run python examples/live_trading_example.py --help
uv run python examples/live_trading_example.py validate --help
uv run python examples/live_trading_example.py trade --help
uv run python examples/live_trading_example.py analyze --help
```

## Failure model and safety gates

Paper/live-facing modes are treated as controlled operations with explicit stop conditions.

### Failure model

Assume failures can occur in any layer:
- data ingestion gaps or schema mismatch,
- feature mismatch between training and inference,
- model/checkpoint incompatibility,
- exchange/network/API interruptions,
- execution/risk constraint violations,
- operational errors (credentials, environment, process supervision).

### Mandatory safety gates for paper/live modes

1. Pre-run validation:
- run `validate` path and confirm end-to-end pipeline execution before `trade`.

2. Feature compatibility gate:
- ensure inference features and checkpoint expectations match.

3. Risk guardrails gate:
- configure and enforce at minimum drawdown, position sizing, and trade-rate limits.

4. Credentials/environment gate:
- use testnet credentials for paper mode; never hardcode secrets; validate runtime env vars.

5. Active operator gate:
- session must be monitored by a human operator with stop authority.

6. Incident stop gate:
- halt immediately on circuit-breaker activation, repeated execution errors, or data/feature integrity violations.

## Explicitly unsupported for unattended live execution

The following are not supported as production guarantees in this repository:
- unattended live trading,
- HA/disaster recovery guarantees,
- regulatory/compliance operations,
- advanced order lifecycle management,
- SRE-grade on-call runbooks and automated recovery.

## Boundary summary

- Supported now: research, offline training/evaluation, controlled paper/pre-live-validation workflows.
- Not supported now: unattended production live trading.
- Any move from beta/controlled toward production-ready live execution requires dedicated hardening work, explicit runbooks, and expanded operational controls beyond current repository scope.
