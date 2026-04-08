# RL Trading Lab

This repository demonstrates how I design and implement **reinforcement learning systems for trading and decision-making problems**, with an emphasis on:

- explicit environment modeling
- reproducible experimentation
- configuration-driven training pipelines
- controlled and well-defined deployment boundaries

This is a **research and engineering lab**, not a turnkey trading bot.

---

## What this repository demonstrates

This project shows how I structure RL work for real-world, non-toy problems:

- Custom Gym-compatible environments for market data
- Reward functions aligned with risk and operational constraints
- Training pipelines using Stable-Baselines3
- Experiment management via Hydra and MLflow
- Checkpointing, evaluation, and reproducibility as first-class concerns

The codebase reflects how I approach RL systems where **traceability, iteration, and correctness** matter more than quick demos.

---

## What this repository is NOT

To avoid ambiguity:

- ❌ Not a plug-and-play trading bot
- ❌ Not a promise of profitability
- ❌ Not financial advice
- ❌ Not intended for unattended live trading

Any execution or “live trading” components exist to demonstrate **system integration patterns**, not to provide a production-ready trading system.


## Typical client engagements

When adapted for client work, this architecture is used for:

- proprietary datasets and custom simulators
- domain-specific reward functions and constraints
- research-grade benchmarking and evaluation
- offline backtesting and paper trading
- exploratory strategy research

Clients typically receive a **scoped, simplified subset** of this codebase, tailored to their objectives and data.


## Quick verification (60-second smoke test)

Set up the repo, then run the canonical smoke command from the command matrix:

```bash
git clone https://github.com/dali1981/rl-lab.git
```

```bash
cd rl-lab; uv sync
```

Use: [docs/commands.md](docs/commands.md) → `Smoke test (clean install)`


## Repository structure (high-level)

```text
.
├── src/
│   └── rl_trading_lab/        # Core environment, agent, and training logic
│
├── configs/                  # Hydra configuration files
│
├── experiments/              # Runnable experiment entry points
│
├── docs/                     # Architecture notes and design rationale
│
├── examples/                 # Minimal usage examples
│
├── README.md
└── LICENSE
```


## Documentation

- [Canonical command matrix](docs/commands.md) (authoritative)
- [Production boundary & supported modes](docs/production_boundary.md) (authoritative)
- [Canonical runtime execution path](docs/runtime_path.md) (authoritative)
- [Architecture rules](docs/architecture_rules.md) (authoritative)
- [Observability contract](docs/observability.md) (authoritative)
- [Architecture overview](docs/architecture.md)
- [Configuration & experiments](docs/configuration.md)
- [Training & checkpoints](docs/training.md)
- [Execution patterns (optional)](docs/execution.md)

## License

This project is released under the included license.

Client-specific adaptations are typically delivered as separate, scoped codebases.
