# Live Trading Integration Boundary

This document defines the isolation contract for live/paper trading integration in RL Trading Lab.

Scope date: April 9, 2026.

## Isolation Policy

`src/rl_trading_lab/live/` is an optional bounded integration zone.

It is not part of the canonical offline training/evaluation path.

Canonical core runtime remains:

```text
CLI -> config -> use_case -> services -> domain -> adapters
```

For DAL-143 compliance:
- core training/evaluation paths must not import `rl_trading_lab.live.*`,
- live-specific dependencies must be installed via the `live` optional extra,
- live scripts remain optional integration surfaces.

## Prerequisites

Install baseline dependencies for core workflows:

```bash
uv sync
```

Install live integration dependencies only when needed:

```bash
uv sync --extra live
```

Additional prerequisites for live/paper modes:
- configured Binance testnet credentials (environment variables),
- a compatible trained checkpoint and feature configuration,
- operator-supervised sessions with explicit safety controls.

## Supported Entry Surfaces

Optional live integration surfaces include:
- `experiments/live_trading.py`
- `examples/live_trading_example.py`
- `src/rl_trading_lab/runtime/live_entrypoint.py`

These are intentionally outside canonical offline training/evaluation execution.

## Non-Goals

This document does not define:
- unattended production live deployment policy,
- secrets management architecture,
- infrastructure topology for production operation.
