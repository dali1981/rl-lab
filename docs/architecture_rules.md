# Architecture Rules (Authoritative)

This document defines the checkable architecture rules for RL Trading Lab.

Scope date: April 9, 2026.

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

### Composition root note

`src/rl_trading_lab/runtime/` is the runtime composition root.
It may wire application services/use cases to infrastructure adapters for executable entrypoints.
This exception applies only to dependency assembly, not to domain/business logic placement.

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

Concrete check command (forbidden imports in application layer):

```bash
rg -n "from rl_trading_lab\\.(infrastructure|live)\\.|import rl_trading_lab\\.(infrastructure|live)\\.|from rl_trading_lab\\.data\\.binance_adapter|from rl_trading_lab\\.infrastructure\\.adapters" src/rl_trading_lab/application
```

Expected: no matches.

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

Concrete check scope:
- Core runnables (must route through application orchestration): `experiments/train.py`, `run_pipeline.py`
- Explicit optional integration/demo runnables: `experiments/live_trading.py`, `examples/live_trading_example.py`, `examples/gym_cartpole_example.py`

Concrete check commands:

```bash
rg -n "from rl_trading_lab\\.application\\.(use_cases|services)|rl_trading_lab\\.application\\.(use_cases|services)" experiments/train.py run_pipeline.py
rg -n "from rl_trading_lab\\.(domain|infrastructure)\\.|import rl_trading_lab\\.(domain|infrastructure)\\." experiments/train.py run_pipeline.py
```

Expected:
- first command has at least one match in core runnables,
- second command has no matches in core runnables.

### Rule 5: Legacy environment path is deprecated

Legacy environment path (`src/rl_trading_lab/environment/`) is treated as transitional:
- no new product features in legacy path,
- only stability fixes or migration-enabling changes,
- net-new architecture work must target domain/application/infrastructure path.

Canonical environment declaration:
- Canonical implementation is `TradingDomain` + `GymTradingEnvAdapter`:
  - `src/rl_trading_lab/domain/trading_domain.py`
  - `src/rl_trading_lab/infrastructure/adapters/gym_adapter.py`
- Legacy monolith `src/rl_trading_lab/environment/trading_env.py` is read-only compatibility surface.
- New environment behavior/features must target canonical path only.
- Legacy compatibility harnesses (for example `experiments/test_one_trade_mode.py`) must:
  - be explicitly marked deprecated,
  - emit `DeprecationWarning`,
  - and remain outside canonical command matrix surfaces.

Migration note:
- DAL-132 declares canonical path and deprecates legacy path for new development.
- DAL-140 adds runtime `DeprecationWarning` on `TradingEnv` instantiation.
- Removal milestone target for legacy `TradingEnv`: June 30, 2026 (post DAL-144 enforcement wave).

Check by inspection:
- feature tickets do not introduce net-new capabilities in `environment/`.
- migration/enforcement tickets move behavior toward domain/use-case/adapters flow.

Concrete check commands:

```bash
rg -n "class TradingDomain|class GymTradingEnvAdapter" src/rl_trading_lab/domain/trading_domain.py src/rl_trading_lab/infrastructure/adapters/gym_adapter.py
rg -n "DEPRECATED LEGACY ENVIRONMENT|read-only compatibility surface|Do not add new features here|Removal milestone|DeprecationWarning" src/rl_trading_lab/environment/trading_env.py
```

Expected:
- canonical classes are present,
- legacy file contains explicit deprecation/read-only declaration.

### Rule 6: Live trading is optional integration, not core domain

`src/rl_trading_lab/live/` and live-related scripts are optional integration surfaces.
They must not become the core domain contract.

Check by inspection:
- domain rules do not depend on live modules.
- documentation and runnable pathways distinguish core architecture from optional live integration.

Concrete check command:

```bash
rg -n "from rl_trading_lab\\.live|import rl_trading_lab\\.live" src/rl_trading_lab/domain src/rl_trading_lab/application src/rl_trading_lab/environment
```

Expected: no matches.

DAL-143 enforcement note:
- core offline training/evaluation paths must also avoid live imports:
  - `experiments/train.py`
  - `run_pipeline.py`
  - `src/rl_trading_lab/runtime/training_entrypoint.py`
  - `src/rl_trading_lab/application/use_cases/train_agent.py`
  - `src/rl_trading_lab/application/use_cases/evaluate_agent.py`
- live dependency installation is optional via `uv sync --extra live`.
- isolation policy and prerequisites are documented in `docs/live_trading.md`.

Concrete DAL-143 check commands:

```bash
rg -n "from rl_trading_lab\\.live|import rl_trading_lab\\.live" experiments/train.py run_pipeline.py src/rl_trading_lab/runtime/training_entrypoint.py src/rl_trading_lab/application/use_cases/train_agent.py src/rl_trading_lab/application/use_cases/evaluate_agent.py
rg -n "^live\\s*=\\s*\\[" pyproject.toml
rg -n "optional bounded integration zone|not part of the canonical offline training/evaluation path|uv sync --extra live|prerequisites" docs/live_trading.md
```

Expected:
- first command has no matches,
- second command has one match defining `[project.optional-dependencies].live`,
- third command has matches confirming isolation policy + prerequisites.

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

Concrete check commands:

```bash
rg -n "from rl_trading_lab\\.(application|infrastructure|live|agents|environment)\\.|import rl_trading_lab\\.(application|infrastructure|live|agents|environment)\\." src/rl_trading_lab/domain
rg -n "from (experiments|examples)\\.|import (experiments|examples)\\." src/rl_trading_lab/application
```

Expected: no matches.

### Rule 9: Trainer orchestration path is singular

Trainer orchestration must have one authoritative path:
- `src/rl_trading_lab/agents/trainer.py` (trainer implementation),
- `src/rl_trading_lab/agents/trainer_factory.py` (project-specific composition).

`src/rl_trading_lab/agents/sb3_agents.py` is compatibility-only and must remain a thin facade.

Check by inspection:
- `sb3_agents.py` contains re-export/facade imports only (no trainer orchestration implementation),
- `trainer_factory.py` imports `Trainer` from `agents.trainer`,
- new production wiring does not import `Trainer` from `agents.sb3_agents`.

Concrete check commands:

```bash
rg -n "class Trainer|stable_baselines3|CallbackList|evaluate_policy" src/rl_trading_lab/agents/sb3_agents.py
rg -n "from rl_trading_lab\\.agents\\.trainer import Trainer" src/rl_trading_lab/agents/trainer_factory.py
rg -n "from rl_trading_lab\\.agents\\.sb3_agents import Trainer" src tests experiments run_pipeline.py
```

Expected:
- first command has no matches except facade comments/docstring text,
- second command has one match,
- third command has no matches (or compatibility-only legacy references under controlled migration).

### Rule 10: Data loading must route through DataLoaderPort + factory

Core runtime and use-case surfaces must not import concrete data loader classes.
Loader resolution must happen through factory boundary, with application services
consuming `DataLoaderPort`.

Check by inspection:
- entrypoints/use-cases do not import `ParquetDataLoader`/`CsvDataLoader`/`BinanceDataAdapter`,
- runtime composition resolves loader via `create_data_loader(...)`,
- `EnvironmentService` receives `DataLoaderPort` and calls `.load(...)`.

Concrete check commands:

```bash
rg -n "from rl_trading_lab\\.application\\.ports\\.data_loader import ParquetDataLoader|from rl_trading_lab\\.infrastructure\\.adapters\\.csv_data_loader import CsvDataLoader|from rl_trading_lab\\.data\\.binance_adapter import BinanceDataAdapter" experiments run_pipeline.py src/rl_trading_lab/runtime src/rl_trading_lab/application/use_cases
rg -n "create_data_loader\\(" src/rl_trading_lab/runtime/training_entrypoint.py
rg -n "DataLoaderPort|_data_loader\\.load\\(" src/rl_trading_lab/application/services/environment_service.py
```

Expected:
- first command has no matches,
- second command has at least one match in canonical runtime composition,
- third command has matches confirming port-based loading in application service.

### Rule 11: Feature engineering must route through FeatureEngineeringPort + factory

Core runtime and use-case surfaces must not import concrete feature pipeline classes.
Feature pipeline resolution must happen through factory boundary, with application
services consuming `FeatureEngineeringPort`.

Check by inspection:
- entrypoints/use-cases do not import `FeaturePipeline` concrete implementation,
- runtime composition resolves feature pipeline via `create_feature_engineering(...)`,
- `EnvironmentService` receives feature port and applies `.transform(...)`.

Concrete check commands:

```bash
rg -n "from rl_trading_lab\\.data\\.feature_pipeline import FeaturePipeline|from rl_trading_lab\\.data import FeaturePipeline" src/rl_trading_lab/application src/rl_trading_lab/domain experiments run_pipeline.py
rg -n "create_feature_engineering\\(" src/rl_trading_lab/runtime/training_entrypoint.py
rg -n "FeatureEngineeringPort|_feature_pipeline\\.transform\\(" src/rl_trading_lab/application/services/environment_service.py
```

Expected:
- first command has no matches,
- second command has at least one match in canonical runtime composition,
- third command has matches confirming port-based feature transformation in application service.

## Enforcement ticket policy

All architecture enforcement tickets in this project should:
- cite this rules file,
- name the specific rule IDs being enforced (for example Rule 1, Rule 5),
- state what evidence was used (grep output, file inspection, tests if applicable),
- avoid unrelated refactors outside the selected rules.

## Enforcement ticket set (project)

The following project tickets are the enforcement set that should reference this document:
- `DAL-127` Enforce canonical runtime path across entrypoints
- `DAL-134` Enforce application layer usage
- `DAL-135` Consolidate trainer architecture
- `DAL-140` Deprecate legacy environment path
- `DAL-141` Enforce data loader port usage
- `DAL-142` Enforce feature engineering port usage
- `DAL-144` Remove responsibility overlaps across layers

Reference line to use in each enforcement ticket:

```text
Architecture Rules Reference: docs/architecture_rules.md
```
