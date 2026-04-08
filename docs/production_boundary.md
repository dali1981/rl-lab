# Production Boundary and Supported Modes

This document is the authoritative boundary for what RL Trading Lab supports today.

As of April 8, 2026, this repository is a research and engineering lab with controlled execution demonstrations. It is not an unattended production trading system.

## Status definitions

| Status | Meaning |
|---|---|
| Stable | Supported for intended lab usage in this repository with clear operating constraints. |
| Beta | Functional but requires operator control, explicit gating, and close monitoring. |
| Experimental | Useful for development, demos, and architecture exploration; not a supported production surface. |

### Experimental surfaces in this repository

The `experimental` status maps to concrete, non-authoritative surfaces such as:
- demo collateral and marketing-oriented plans (for example `DEMO_PLAN.md`, `docs/upwork_demo_execution_plan.md`),
- architecture review/refactor analysis docs (for example `docs/ddd_clean_architecture_review.md`, `docs/refactor_2024-11-29.md`, `docs/rl_module_reusability_assessment.md`),
- debugging and exploratory analysis notes (for example `docs/debugging/*`, notebooks under `notebooks/`),
- convenience wrappers or exploratory orchestration paths not treated as canonical runtime contracts (for example `run_pipeline.py`).

Note: `Experimental` is included in the status legend for full boundary definition. No currently supported mode is classified as `Experimental`; those surfaces are intentionally listed outside the supported-mode contract.

## Supported modes

### Mode overview

| Mode | Description | Status | Supported today | Intended users |
|---|---|---|---|---|
| Research | Iterative strategy and architecture exploration in lab context. | Stable | Yes | Researchers and developers |
| Offline training | Canonical model training from historical datasets. | Stable | Yes | Researchers and model developers |
| Evaluation | Checkpoint evaluation within canonical training/eval workflow. | Stable | Yes | Researchers, reviewers, and release operators |
| Paper trading (testnet) | Controlled order execution against Binance testnet with operator supervision. | Beta | Yes, via `examples/live_trading_example.py trade` | Human operator and developer during supervised sessions |
| Pre-live validation (historical/integration) | Historical/integration gating run to validate pipeline readiness before paper trading. | Beta | Yes, via `examples/live_trading_example.py validate` | Human operator and release reviewer |

### Mode operating contract

| Mode | Prerequisites | Constraints | Safety gates |
|---|---|---|---|
| Research | Python/uv environment, repository dependencies installed, local configs. | No unattended automation claims; results are exploratory by default. | N/A (non-execution mode). |
| Offline training | `uv sync`, data file available at configured path, write access for checkpoints/logs. | Requires valid dataset path and config compatibility; overrides must match current Hydra schema. | Data/config compatibility checks before long runs. |
| Evaluation | Trained checkpoint, compatible feature/config schema, evaluation data availability. | Standalone production evaluator service is not provided; evaluation is bounded by current scripts. | Feature/checkpoint compatibility gate before interpretation. |
| Paper trading (testnet) | Binance testnet credentials, validated model, safety guard parameters, active monitoring session. | Testnet only by default policy; human-in-the-loop required; no unattended operation. | Drawdown, position-size, trade-rate, and incident-stop gates required. |
| Pre-live validation (historical/integration) | Trained model + optional VecNormalize, historical data access, feature compatibility checks. | Not live execution; no exchange order placement; passing does not authorize unattended live trading. | Must pass before paper-trading sessions; block on data/feature/model mismatch. |

Paper trading vs pre-live validation distinction:
- `Pre-live validation` is historical/integration gating only and does not place live orders.
- `Paper trading` is supervised testnet execution with explicit runtime safety controls.

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

Use [docs/commands.md](commands.md) as the authoritative command source.

Mode-to-command mapping:
- Research / Offline training / Evaluation: use `Smoke test`, `Local training`, and `Evaluation` rows.
- Pre-live validation / Paper trading: use `Pre-live validation surface`, `Paper trading surface`, and `Session analysis surface` rows.
- MLflow and notebook operations: use `MLflow UI` and `Notebook inspection` rows.

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

- Supported now: stable research and offline training/evaluation workflows, plus beta paper/pre-live-validation workflows.
- Not supported now: unattended production live trading.
- Any move from beta toward stable live execution requires dedicated hardening work, explicit runbooks, and expanded operational controls beyond current repository scope.
