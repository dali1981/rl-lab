# RL Trading Lab — Upwork Demo Execution Plan

**Date:** 2026-03-28
**Goal:** Produce all visual assets, case study, and thumbnail for Upwork portfolio asset #5.
**Estimated time:** ~2.5 hours total

---

## Inventory Check (what we have)

| Asset | Status | Detail |
|-------|--------|--------|
| Trained models | 52 checkpoints | PPO (13), A2C (22), DQN (16), MaskablePPO (1) |
| MLflow runs | 79 runs across 2 experiments | Full metrics/params/artifacts |
| Sample data | `sample_data/btcusdt_sample_10k.parquet` | 10K bars, 14 columns, pre-featurized |
| Agent configs | 8 YAML files | ppo, a2c, dqn, maskable_ppo, 3 transformer variants, dqn_aggressive |
| Notebooks | 10+ notebooks | `debug_episode.ipynb` is the hero visual |
| Source code | ~13,700 LOC in `src/` | Domain, environment, agents, live, infrastructure |
| Best PPO checkpoint | `checkpoints/PPO_returns_20251028_143659/` | Has best_model + final_model + eval_logs |
| DEMO_PLAN.md | Exists | Recording instructions already written |
| CartPole example | `examples/gym_cartpole_example.py` | Proves reusable RL module (just added) |

---

## What's Missing

| Asset | Needed | Blockers |
|-------|--------|----------|
| MLflow screenshots | 5 screenshots | None — `mlruns/` has 79 runs, just launch UI |
| Training demo screenshot | 1 screenshot | Need working `train.py` run — sample data exists |
| Live trading screenshot | 1 screenshot | Needs Binance testnet keys + MinIO running |
| Notebook visualization | 1 screenshot | Run `debug_episode.ipynb` |
| SCARA case study | Written text | None — draft in the original plan |
| Thumbnail | 795x595 image | Design tool needed |

---

## Execution Steps

### Phase 1: MLflow Screenshots (15 min)

**Pre-check:** MLflow data is in `mlruns/` — no server needed to start, just the UI.

```bash
cd ~/trading_project/rl-trading-lab
uv run mlflow ui --port 5000
# Open http://localhost:5000
```

**Capture 5 screenshots (save to `docs/assets/`):**

| # | Screenshot | What to frame | Filename |
|---|-----------|---------------|----------|
| 1 | Experiments overview | All 79 runs in table view, show columns: run name, agent, reward, duration | `mlflow_experiments_overview.png` |
| 2 | Run comparison | Select 3-4 best PPO/A2C/DQN runs → Compare → metrics table | `mlflow_run_comparison.png` |
| 3 | Training curves | Click best PPO run → Metrics tab → `rollout/ep_rew_mean` chart | `mlflow_training_curves.png` |
| 4 | Parameters view | Same run → Parameters tab → show hyperparams logged | `mlflow_parameters.png` |
| 5 | Agent comparison | Filter by tag or name containing "PPO" vs "A2C" → compare curves | `mlflow_agent_comparison.png` |

**Tips:**
- Dark theme looks better in portfolio (MLflow supports it)
- Increase browser zoom to 110-125% for readability
- Ensure no sensitive data (API keys, paths) visible

---

### Phase 2: Training Demo Screenshot (15 min)

The `train.py` uses Hydra + the legacy `TradingEnv` which reads from a parquet file. Sample data exists.

```bash
cd ~/trading_project/rl-trading-lab

# Quick PPO training on sample data (should complete in ~30-60 seconds)
uv run python experiments/train.py \
  data.train_data_path=sample_data/btcusdt_sample_10k.parquet \
  training.total_timesteps=10000 \
  agent=ppo
```

**If this fails** (Hydra config resolution, missing override keys, etc.), fall back to the CartPole example which is known-working:

```bash
uv run python examples/gym_cartpole_example.py
```

**Capture:**
- Terminal with progress bar, training metrics (reward, loss, etc.)
- Save as `docs/assets/training_demo.png`

**Alternative hero screenshot:** Split-screen terminal (left: training, right: MLflow UI).

---

### Phase 3: Notebook Visualization (15 min)

```bash
cd ~/trading_project/rl-trading-lab
uv run jupyter lab notebooks/debug_episode.ipynb
```

**What to capture:**
- The 5-panel chart (Price, Actions, Positions, Rewards, Portfolio Value)
- If the notebook needs a trained model path, point it to: `checkpoints/PPO_returns_20251028_143659/best_model/best_model.zip`
- If the notebook won't run cleanly, any of the `inspect_trained_policy*.ipynb` notebooks are alternatives

**Save as:** `docs/assets/notebook_visualization.png`

---

### Phase 4: Live Trading Screenshot (15 min, optional)

**Requires:** Binance testnet API keys configured + data infrastructure running.

```bash
# Check if testnet config exists
cat ~/.env 2>/dev/null | grep -i binance || echo "No Binance env found"
```

If available:
```bash
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/final_model.zip \
  --days 1
```

**If not available:** Skip this. The MLflow + training + notebook screenshots are sufficient for a strong carousel. The live trading screenshot is high-impact but blocked on infrastructure.

**Save as:** `docs/assets/live_trading_dashboard.png`

---

### Phase 5: Architecture Diagram (10 min)

Create a clean text diagram or use a tool (Excalidraw, draw.io). Content should reflect the **actual** architecture after our refactoring:

```
┌───────────────────────────────────────────────────────────┐
│                     RL Trading Lab                         │
├───────────────────────────┬───────────────────────────────┤
│   Reusable RL Module      │   Trading Domain              │
│   ──────────────────      │   ──────────────              │
│   • Trainer               │   • TradingDomain (pure)      │
│   • EnvWrapperBuilder     │   • Position, Trade (VOs)     │
│   • CallbackFactory       │   • Reward strategies         │
│   • Any Gym env           │   • Risk management           │
├───────────────────────────┼───────────────────────────────┤
│   Training Pipeline       │   Live Trading                │
│   ──────────────────      │   ────────────                │
│   • PPO/A2C/DQN/SAC       │   • Binance WebSocket         │
│   • Transformer Policy    │   • Real-time predictions     │
│   • MLflow Tracking       │   • Circuit breakers          │
│   • Hydra Configs         │   • Position safety guards    │
├───────────────────────────┴───────────────────────────────┤
│   Data Ports: Parquet | CSV | Delta Lake (pluggable)      │
│   Tech: Python 3.12 | SB3 | Gymnasium | PyTorch | MLflow │
└───────────────────────────────────────────────────────────┘
```

**Save as:** `docs/assets/architecture_diagram.png`

---

### Phase 6: SCARA Case Study (30 min)

Write to `docs/scara_case_study.md`. Updated numbers from actual repo:

**S — Situation:**
Exploring reinforcement learning for systematic trading required reproducible experiment infrastructure across multiple agents, reward functions, and market environments. Needed to compare PPO, A2C, DQN, and MaskablePPO across different hyperparameter configurations on crypto market data.

**C — Complication:**
Ad-hoc RL experiments weren't reproducible. No way to compare hyperparameter sweeps or track model performance over time. The environment logic (positions, commissions, slippage, risk) was tangled with the RL framework, making it impossible to reuse for different asset classes. Risk management wasn't enforced at the environment level.

**A — Action:**
Built a modular RL framework with clean architecture (DDD):
- **Pure domain layer** (zero numpy/pandas deps) — `TradingDomain`, immutable value objects (`Position`, `Trade`, `Bar`), protocol-based services (reward, risk, position sizing)
- **Pluggable data ports** — `DataLoaderPort` + `FeatureEngineeringPort` with Parquet, CSV, and Delta Lake adapters
- **Composable trainer** — `Trainer` + `EnvWrapperBuilder` + `CallbackFactory` work with any Gymnasium environment, not just trading
- **Reproducible experiments** — 8 Hydra agent configs, MLflow tracking for metrics/artifacts/parameters
- **Live deployment** — Binance WebSocket integration with circuit breakers, rate limits, position controls

**R — Result:**
- 52 trained models across 4 agent architectures (PPO, A2C, DQN, MaskablePPO)
- 79 tracked experiments with full metric comparison in MLflow
- 13,700 LOC of production Python — type-safe, tested, documented
- Reusable RL module demonstrated on CartPole (works with any Gym env)
- Multi-layer safety: drawdown limits, consecutive loss stops, position size caps

**A — Artifact:**
- MLflow dashboard (79 runs)
- Training curves (PPO vs A2C vs DQN)
- CartPole reusability demo
- Architecture showing clean domain separation
- Optional: Rich terminal live trading dashboard

---

### Phase 7: Thumbnail (15 min)

**Specs:** 795 × 595 px, dark background

**Content:**
```
RL TRADING LAB
━━━━━━━━━━━━━━━━━━━━━━━━━
52 Trained Models  ·  79 Tracked Experiments
PPO  ·  A2C  ·  DQN  ·  MaskablePPO

13,700 LOC  ·  Clean Architecture  ·  Reusable

Gymnasium  ·  Stable-Baselines3  ·  MLflow  ·  Hydra
```

Use Figma, Canva, or Gamma to create. Dark background (#1a1a2e or similar), monospace font for tech feel.

---

### Phase 8: Video Recording (30 min, optional but high-impact)

Follow `DEMO_PLAN.md` in repo. Updated script:

| Act | Duration | Content | Command |
|-----|----------|---------|---------|
| 1 | 10s | Project structure | `tree -L 2 --dirsfirst` |
| 2 | 45s | Training demo | `uv run python experiments/train.py agent=ppo training.total_timesteps=10000` |
| 3 | 30s | MLflow dashboard | Browse `http://localhost:5000` — experiments, curves, comparison |
| 4 | 30s | CartPole reuse demo | `uv run python examples/gym_cartpole_example.py` |
| 5 | 30s | Notebook viz | Show 5-panel chart from `debug_episode.ipynb` |
| 6 | 10s | Closing | Architecture diagram + "Production-ready RL framework" |

**Total:** ~2.5 min. Record with QuickTime (Cmd+Shift+5) or OBS.

---

## Execution Order (Priority)

| Priority | Step | Time | Impact | Blocker? |
|----------|------|------|--------|----------|
| **P0** | Phase 1: MLflow screenshots | 15m | High — hero images | No |
| **P0** | Phase 6: SCARA case study | 30m | High — required text | No |
| **P1** | Phase 2: Training demo screenshot | 15m | High | Test `train.py` with sample data first |
| **P1** | Phase 3: Notebook visualization | 15m | Medium-High | May need model path fix |
| **P1** | Phase 7: Thumbnail | 15m | Required | Design tool |
| **P2** | Phase 5: Architecture diagram | 10m | Medium | None |
| **P2** | Phase 8: Video recording | 30m | High but optional | Needs phases 1-3 done |
| **P3** | Phase 4: Live trading screenshot | 15m | Medium | Binance testnet + MinIO |

**Critical path:** Phases 1 → 2 → 6 → 7 = minimum viable submission (~75 min).
Add Phase 3 + 5 + 8 for maximum impact (~2.5 hours total).

---

## Updated Numbers (verified from repo)

| Claim in original plan | Actual | Action |
|------------------------|--------|--------|
| "54 trained models" | 52 checkpoints | Update to 52 |
| "75+ runs tracked" | 79 MLflow runs | Update to 79 |
| "8,000+ Lines" | 13,700 LOC in `src/` | Update to 13,700 |
| "8 agent configs" | 8 YAML files | Correct |
| "PPO (8), A2C (24), DQN (17)" | PPO (13), A2C (22), DQN (16) | Update counts |
