# DAL-136 Trainer Consolidation Audit (`sb3_agents.py`)

Scope date: April 8, 2026.

Issue: `DAL-136`  
Parent: `DAL-135`

This audit is based on:
- the last pre-consolidation implementation of `src/rl_trading_lab/agents/sb3_agents.py` (`HEAD~1` on main),
- current composition modules:
  - `src/rl_trading_lab/agents/env_wrapper.py`
  - `src/rl_trading_lab/agents/callback_factory.py`
  - `src/rl_trading_lab/agents/trainer_factory.py`.

## Public method/function classification

### `Trainer` class methods

| Symbol | Classification | Notes |
|---|---|---|
| `Trainer.__init__` | Partially overlapping | Core agent construction is unique trainer behavior; environment/callback wiring inputs come from factory/builder path. |
| `Trainer.setup_logger` | Partially overlapping | Applies logger objects to SB3 agent; logging format/output objects are produced by `CallbackFactory.create_logging_setup()`. |
| `Trainer.train` | Unique | Canonical training execution (`agent.learn`) and terminal model save remain trainer responsibilities. |
| `Trainer.evaluate` | Unique | Evaluation policy dispatch (maskable vs standard) and metric aggregation are trainer-specific. |
| `Trainer.predict` | Unique | Inference helper with optional action-mask route is trainer-specific. |
| `Trainer.save` | Unique | Trainer-owned model persistence wrapper. |
| `Trainer.load` | Unique | Trainer-owned checkpoint restore path via `CheckpointManager`. |

### Module-level public symbols

| Symbol | Classification | Notes |
|---|---|---|
| `MASKABLE_AVAILABLE` | Unique | Capability flag used by trainer behavior and compatibility surfaces. |
| `_process_policy_kwargs` | Unique (helper) | Helper for resolving policy kwargs (`activation_fn`, feature extractor class references). |
| `_resolve_policy` | Unique (helper) | Helper for resolving custom policy aliases (for example transformer policy alias). |

## Overlap summary

- `env_wrapper.py` owns environment wrapping concerns (Monitor/DummyVecEnv/VecNormalize) and does not duplicate trainer execution methods.
- `callback_factory.py` owns callback/logging object construction and does not duplicate trainer execution/evaluation/persistence methods.
- `trainer_factory.py` owns project-specific composition from config objects and delegates runtime execution to `Trainer`.

Result: there is no meaningful method-level duplication left in the pre-consolidation `sb3_agents.py`; most prior monolith concerns were already extracted before DAL-136 execution.

## Recommendation

### Recommended option: **Option A (facade)**

Use `sb3_agents.py` as a thin compatibility facade and keep authoritative implementation in `agents/trainer.py` plus project wiring in `agents/trainer_factory.py`.

Rationale:
- preserves backward import compatibility for existing references,
- avoids unnecessary breakage while architecture converges,
- keeps one authoritative orchestration implementation without duplicate logic.

### Retirement criteria for eventual Option B

`sb3_agents.py` can be deleted once all downstream imports are migrated and compatibility guarantees are no longer required.

Concrete gate:
- repository-wide check for `from rl_trading_lab.agents.sb3_agents import Trainer` returns no production/test/docs usage requiring compatibility.
