# Asset #5 — RL Trading Lab

**PIN Priority**: Supporting #5
**Projects**: `rl-trading-lab` + `rl-trading-lab-mlflow` + `use-ray`
**Suggested Title**: "RL Trading Lab: Gymnasium Environments + MLflow Tracking + Distributed Training"
**Rate Justified**: $80–150/hr
**Client Types**: Quant researchers, AI trading startups

---

## Current State

| Aspect | Status | Detail |
|--------|--------|--------|
| Code | Done | `~/trading_project/rl-trading-lab/` — custom Gym envs, SB3 agents |
| Tests | 21 files | Functional but not comprehensive |
| Checkpoints | **54 trained models** | PPO (8), A2C (24), DQN (17), others (5+) |
| MLflow | **75+ runs tracked** | Full metrics, params, artifacts logged |
| Hydra configs | Done | 8 agent configs, env configs, feature engineering |
| DEMO_PLAN.md | **Exists** | Detailed video capture instructions in repo |
| Visual proof | **Missing** | Need MLflow dashboard screenshots + training curves |
| Case study | **Not written** | Need SCARA writeup |

---

## Action Items

### Step 1: Launch MLflow UI and Screenshot (15 minutes)

```bash
cd ~/trading_project/rl-trading-lab

# Launch MLflow UI (reads from local mlruns/ directory)
uv run mlflow ui --port 5000
# → Opens http://localhost:5000
```

**Take 5 screenshots**:
1. **Experiments tab** — shows 75+ runs across PPO/A2C/DQN (hero image)
2. **Run comparison** — select 3-4 best runs, show metrics side-by-side
3. **Training curves** — click a run, show reward/returns over training steps
4. **Parameters tab** — show hyperparameters logged per run
5. **Agent comparison** — filter by agent type, compare PPO vs A2C vs DQN

### Step 2: Run Quick Training Demo (15 minutes)

```bash
cd ~/trading_project/rl-trading-lab

# Quick PPO training (10K steps, ~45 seconds)
uv run python experiments/train.py agent=ppo trainer.max_steps=10000
# → Shows: progress bar, real-time metrics, MLflow logging

# Quick A2C training for comparison
uv run python experiments/train.py agent=a2c trainer.max_steps=10000
```

**Screenshot**: Terminal showing training progress with metrics (reward, episode length, loss).

### Step 3: Run Validation Demo (10 minutes, optional)

```bash
cd ~/trading_project/rl-trading-lab

# Validate best PPO checkpoint
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/final_model.zip \
  --days 1
# → Shows: Rich terminal UI with portfolio, positions, predictions
```

**Screenshot**: Rich terminal UI showing portfolio dashboard.

### Step 4: Record Video (30 minutes, optional but high-impact)

Follow the existing `DEMO_PLAN.md` in the repo. Two options:

**Option A — 2-3 Minute Video** (4 acts):
1. Training: `uv run python experiments/train.py agent=ppo`
2. MLflow: browse dashboard, compare runs
3. Live trading validation: run example script
4. Notebook visualization: open `notebooks/debug_episode.ipynb`

**Option B — 5-7 Image Carousel** (for Upwork portfolio):
1. MLflow experiments tab
2. Training curves chart
3. Agent comparison metrics
4. Rich terminal portfolio dashboard
5. Jupyter visualization (5-panel chart)

### Step 5: Write SCARA Case Study (30 minutes)

**S — Situation**:
Exploring reinforcement learning for systematic trading required reproducible experiment infrastructure across multiple agents, reward functions, and market environments. Needed to compare PPO, A2C, and DQN across different hyperparameter configurations.

**C — Complication**:
Ad-hoc RL experiments weren't reproducible. No way to compare hyperparameter sweeps or track model performance over time. Training on a single GPU was too slow for comprehensive sweeps. Risk management (stop-loss, margin calls) wasn't enforced at the environment level.

**A — Action**:
Built a modular RL framework: custom Gymnasium environments for market data (equities, crypto, FX), Hydra configuration management for reproducible experiments, MLflow tracking for metrics/artifacts, RayTune for distributed multi-GPU training. Risk management overlays (stop-loss, trailing stops, margin calls) enforced at the environment level, not as post-hoc filters.

**R — Result**:
54 trained models across 3 agent architectures. 75+ tracked experiments with full metric comparison in MLflow. Reproducible experiments via Hydra configs. Multi-asset support (equities, FX, crypto). Risk overlays prevent catastrophic losses during training.

**A — Artifact**:
- MLflow dashboard screenshot (75+ runs)
- Training curves (PPO vs A2C vs DQN)
- Rich terminal portfolio validation output
- Optional: 2-3 min demo video

### Step 6: Create Thumbnail (15 minutes)

**Specs**: 795 × 595 px
**Content**:
```
RL TRADING LAB
━━━━━━━━━━━━━━
54 Trained Models  |  75+ Tracked Experiments
PPO | A2C | DQN  |  Multi-Asset Support

Gymnasium  |  MLflow  |  Hydra  |  RayTune
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `experiments/train.py` | Main training entry point |
| `examples/live_trading_example.py` | Live trading validation |
| `configs/config.yaml` | Main Hydra config (MLflow, training, logging) |
| `configs/agent/*.yaml` | Agent configs (ppo, a2c, dqn, etc.) |
| `checkpoints/PPO_returns_20251028_143659/` | Best PPO checkpoint |
| `notebooks/debug_episode.ipynb` | 5-panel trading visualization |
| `DEMO_PLAN.md` | Full demo recording instructions |

---

## Definition of Done

- [ ] MLflow UI screenshots captured (5 shots)
- [ ] Training demo screenshot captured
- [ ] Validation demo screenshot captured (optional)
- [ ] Video recorded following DEMO_PLAN.md (optional, high-impact)
- [ ] SCARA case study written
- [ ] Thumbnail image created (795×595)
- [ ] Uploaded to Upwork as Supporting #5
