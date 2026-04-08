# Canonical Runtime Execution Path

This document defines the authoritative runtime execution path for RL Trading Lab.

Scope date: April 8, 2026.

## Canonical flow

The canonical architecture flow is:

```text
CLI -> config (Hydra) -> use_case -> services -> domain -> adapters
```

Equivalent Mermaid view:

```mermaid
flowchart LR
    A[CLI] --> B[Config (Hydra)]
    B --> C[Use Case]
    C --> D[Application Services]
    D --> E[Domain]
    E --> F[Adapters]
```

## What direct wiring means

Direct wiring means a top-level runnable script directly imports and wires domain, environment, agent, or infrastructure components instead of routing through application use cases/services as the orchestration boundary.

Examples of direct wiring patterns:
- runnable imports `rl_trading_lab.environment.*` and constructs env directly,
- runnable imports trainer/agent/infrastructure modules and composes pipeline inline,
- runnable bypasses use-case orchestration for core training/evaluation flow.

Why direct wiring is disallowed for canonical paths:
- blurs architecture boundaries,
- duplicates orchestration logic across scripts,
- makes enforcement and migration harder,
- increases regression risk when core flow changes.

## Entrypoint classification

All top-level entrypoints in `experiments/`, `run_pipeline.py`, and `examples/` are classified below.

| Entrypoint | Classification | Why |
|---|---|---|
| `experiments/train.py` | Canonical core runtime path | Uses Hydra config and routes orchestration through `TrainAgentUseCase` and application services. |
| `run_pipeline.py` | Canonical wrapper path | Optional Kedro pre-step wrapper that delegates training execution to canonical `experiments/train.py`. |
| `experiments/live_trading.py` | Intentionally standalone demo/integration (deprecated warning on invoke) | Optional live integration runner wiring real-time components directly; not canonical core domain/runtime path. |
| `experiments/validate_data_pipeline.py` | Intentionally standalone demo/integration (deprecated warning on invoke) | Standalone data-pipeline validation harness for operational checks. |
| `experiments/validate_live.py` | Intentionally standalone demo/integration (deprecated warning on invoke) | Standalone end-to-end live pipeline validation harness. |
| `experiments/test_one_trade_mode.py` | Intentionally standalone demo/legacy harness (deprecated warning on invoke) | Standalone script targeting legacy environment behavior verification. |
| `experiments/test_transformer.py` | Intentionally standalone demo/research harness (deprecated warning on invoke) | Standalone research harness for transformer policy compatibility checks. |
| `examples/live_trading_example.py` | Intentionally standalone demo/integration (deprecated warning on invoke) | Demonstration CLI for testnet/live validation workflows; optional integration surface. |
| `examples/gym_cartpole_example.py` | Intentionally standalone demo (deprecated warning on invoke) | Reusability showcase for trainer components in non-trading Gym environment. |

## Canonical vs non-canonical boundary

Canonical runtime path is for core training/evaluation orchestration and must converge to:

```text
CLI -> config -> use_case -> services -> domain -> adapters
```

Non-canonical paths are allowed only when explicitly demo/integration-oriented and clearly documented as optional surfaces.

Compatibility/demo entrypoints must emit `DeprecationWarning` when invoked.

## DAL-134 deprecated bypass allowlist

For DAL-134 enforcement, direct integration imports in `experiments/` are allowed only for these explicit deprecated paths:

- `experiments/live_trading.py`
- `experiments/validate_data_pipeline.py`
- `experiments/validate_live.py`
- `experiments/test_one_trade_mode.py`
- `experiments/test_transformer.py`

Any new or existing `experiments/` entrypoint outside this list that imports direct integration surfaces (`rl_trading_lab.live`, `rl_trading_lab.data`, `rl_trading_lab.infrastructure`, or legacy `environment` paths) is out of compliance.

## Reference for enforcement

DAL-127 (enforcement) should use this document as the source of truth for which entrypoints require migration.

Recommended reference line:

```text
Runtime Path Reference: docs/runtime_path.md
```

## Smoke command surfaces (current)

These command surfaces are currently valid and used for runtime-path inspection:

```bash
uv run python experiments/train.py --help
uv run python run_pipeline.py --help
uv run python experiments/live_trading.py --help
uv run python experiments/validate_data_pipeline.py --help
uv run python experiments/validate_live.py --help
uv run python examples/live_trading_example.py --help
```

Interpretation:
- Presence of command/help surface does not imply canonical status.
- Canonical status is determined by architecture flow and boundary compliance, not by script availability.
- `DeprecationWarning` on demo/integration entrypoints is expected and indicates non-canonical status by design.
