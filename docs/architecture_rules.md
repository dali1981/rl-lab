# Architecture Rules (Authoritative)

This document defines the checkable architecture rules for RL Trading Lab.

Scope date: April 8, 2026.

Use this file as the baseline for all architecture enforcement tickets in the `RL Trading Lab — Production Hardening` project.

## How to use this document

- Treat each rule as normative.
- Do not add new code that violates a rule.
- If a rule is temporarily violated by legacy code, contain it and document a migration ticket.
- Reference this file in every enforcement ticket.

Recommended ticket line:

```text
Architecture Rules Reference: docs/architecture_rules.md
```

## Rule set

### Rule 1: Domain purity

`src/rl_trading_lab/domain/` must not import framework/infrastructure libraries, including:
- `pandas`
- `numpy`
- `gymnasium`
- `stable_baselines3`
- `mlflow`
- exchange clients (for example `binance`)

Check by inspection:

```bash
rg -n "import pandas|from pandas|import numpy|from numpy|import gymnasium|from gymnasium|import stable_baselines3|from stable_baselines3|import mlflow|from mlflow|binance" src/rl_trading_lab/domain
```

Expected: no matches.

### Rule 2: Application orchestrates, adapters implement

`src/rl_trading_lab/application/` may orchestrate use cases and policies but must not implement concrete external integrations (exchange clients, storage SDK clients, MLflow concrete tracker internals).

Concrete integrations belong in `src/rl_trading_lab/infrastructure/` or explicitly optional integration surfaces.

Check by inspection:
- application code depends on ports/services/use cases.
- concrete framework/external client calls live in adapters/integration modules.

### Rule 3: Only adapters touch framework/exchange specifics

Framework boundary logic (Gym wrappers, MLflow adapter, concrete data/exchange adapters) must be implemented in:
- `src/rl_trading_lab/infrastructure/adapters/`
- optional integration surfaces (`src/rl_trading_lab/live/`, demo entrypoints)

Domain and use-case code must stay framework-agnostic.

Check by inspection:
- framework-specific classes/functions are in adapters/integration modules.
- domain/use-case modules do not directly instantiate exchange/framework clients.

### Rule 4: Top-level runnables must route through use cases (except explicit demo/integration)

Top-level runnable entrypoints must execute application orchestration and not embed new business logic in scripts.

Allowed exception:
- explicit demo/integration entrypoints clearly marked as optional (for example live/paper integration examples).

Check by inspection:
- runnable scripts (`experiments/*`, `examples/*`, `run_pipeline.py`) call services/use cases or integration orchestrators.
- no new core business rules are added directly in runnable scripts.

### Rule 5: Legacy environment path is deprecated

Legacy environment path (`src/rl_trading_lab/environment/`) is treated as transitional:
- no new product features in legacy path,
- only stability fixes or migration-enabling changes,
- net-new architecture work must target domain/application/infrastructure path.

Check by inspection:
- feature tickets do not introduce net-new capabilities in `environment/`.
- migration/enforcement tickets move behavior toward domain/use-case/adapters flow.

### Rule 6: Live trading is optional integration, not core domain

`src/rl_trading_lab/live/` and live-related scripts are optional integration surfaces.
They must not become the core domain contract.

Check by inspection:
- domain rules do not depend on live modules.
- documentation and runnable pathways distinguish core architecture from optional live integration.

### Rule 7: Ports define dependencies across boundaries

Cross-layer dependencies should be expressed via port interfaces:
- domain ports in `src/rl_trading_lab/domain/ports/`
- application ports in `src/rl_trading_lab/application/ports/`

Check by inspection:
- adapters implement ports.
- orchestration depends on abstractions where boundary crossing is required.

### Rule 8: Runtime direction remains inward

Target runtime direction for core flows:

```text
CLI -> config -> use_case -> services -> domain -> adapters
```

No new change should invert this flow by pushing external concerns into domain logic.

Check by inspection:
- new dependencies point inward for core logic.
- adapter code remains at the boundary.

## Enforcement ticket policy

All architecture enforcement tickets in this project should:
- cite this rules file,
- name the specific rule IDs being enforced (for example Rule 1, Rule 5),
- state what evidence was used (grep output, file inspection, tests if applicable),
- avoid unrelated refactors outside the selected rules.
