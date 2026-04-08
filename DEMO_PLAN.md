# RL Trading Lab - Upwork Demo Plan

## Target Audience
Upwork clients seeking:
- Custom algorithmic trading systems
- Reinforcement learning / ML engineering
- Binance/exchange integrations
- Safety-gated trading system prototypes

---

## Option A: Short Video (2-3 minutes)

### Structure

**Opening (10 sec)**
- Terminal: Show project structure with `tree -L 2`
- Voiceover: "Research RL trading lab - training to controlled live validation"

**Act 1: Training (45 sec)**
```bash
uv run python experiments/train.py agent=ppo training.total_timesteps=50000
```
- Show: Progress bar, real-time metrics (reward, returns)
- Voiceover: "Train PPO, A2C, DQN, or custom Transformer agents"
- Show: Config override flexibility with Hydra CLI

**Act 2: MLflow Dashboard (30 sec)**
```bash
mlflow ui --port 5000
```
- Show: Experiment comparison view
- Show: Metrics graphs (reward curves)
- Show: Hyperparameter tracking
- Voiceover: "100+ experiments tracked with full reproducibility"

**Act 3: Live Trading Dashboard (45 sec)**
```bash
uv run python examples/live_trading_example.py trade \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --symbol BTCUSDT --balance 10000
```
- Show: Rich terminal UI with:
  - Portfolio summary (balance, PnL, returns %)
  - Active positions table
  - Model predictions (BUY/SELL/HOLD + confidence)
  - Recent trades with PnL
  - Safety guard status
- Voiceover: "Real-time Binance WebSocket, multi-layer safety controls"

**Act 4: Notebook Visualization (30 sec)**
- Show: `debug_episode.ipynb` output
- 5-panel chart: Price, Actions, Positions, Rewards, Portfolio Value
- Voiceover: "Interactive episode analysis and debugging"

**Closing (10 sec)**
- Show: Architecture diagram from docs
- Voiceover: "Clean architecture, type-safe, and safety-gated for controlled operation"
- Text overlay: Contact info

---

## Option B: Image Carousel (5-7 images)

### Image 1: Hero Shot
**What to capture:** Split terminal showing:
- Left: Training running with metrics
- Right: MLflow UI with experiment table

**Caption:** "Reinforcement Learning Trading System with MLflow Tracking"

### Image 2: Configuration System
**What to capture:**
- VS Code with `configs/agent/ppo.yaml` open
- Terminal showing CLI override example

**Caption:** "50+ Config Templates | Every Parameter Tunable"

### Image 3: Live Trading Dashboard
**What to capture:** Rich terminal UI during live trading showing:
- Portfolio summary panel
- Positions table
- Predictions with confidence scores
- Safety status

**Caption:** "Real-time Trading with Multi-Layer Safety Controls"

### Image 4: Notebook Visualization
**What to capture:** Jupyter with `debug_episode.ipynb` showing:
- 5-subplot episode analysis
- Price + actions overlay

**Caption:** "Interactive Episode Analysis & Debugging"

### Image 5: Architecture
**What to capture:** Diagram from `internals/HOW_IT_WORKS.md` or create:
```
┌─────────────────────────────────────────────────────────────┐
│                        RL Trading Lab                        │
├─────────────────────────────────────────────────────────────┤
│  Training Pipeline          │  Live Trading                 │
│  ─────────────────          │  ────────────                 │
│  • PPO/A2C/DQN/SAC          │  • Binance WebSocket          │
│  • Transformer Policy       │  • Real-time predictions      │
│  • MLflow Tracking          │  • Risk management            │
│  • Walk-forward validation  │  • Circuit breakers           │
├─────────────────────────────────────────────────────────────┤
│  Tech Stack: Python 3.12 | Stable-Baselines3 | Gymnasium    │
│              Hydra | MLflow | Rich | Binance API            │
└─────────────────────────────────────────────────────────────┘
```

**Caption:** "Enterprise Architecture: Training to Controlled Execution"

### Image 6: Code Quality
**What to capture:** VS Code with:
- Type hints visible
- Clean function signatures
- Docstrings

**Caption:** "8,000+ Lines of Typed, Tested Engineering Code"

### Image 7: Results/Metrics
**What to capture:** MLflow metrics view showing:
- Training reward curves
- Multiple experiment comparison
- Final performance stats

**Caption:** "Track Every Experiment | Compare Agents | Reproduce Results"

---

## Quick Capture Commands

### Terminal Screenshots

```bash
# 1. Project structure
tree -L 2 --dirsfirst

# 2. Training with metrics
uv run python experiments/train.py agent=ppo training.total_timesteps=10000

# 3. MLflow UI
mlflow ui --port 5000
# Then screenshot http://localhost:5000

# 4. Live trading (testnet safe)
uv run python examples/live_trading_example.py validate \
  --model checkpoints/PPO_returns_20251028_143659/best_model.zip \
  --days 1

# 5. Jupyter visualization
cd notebooks && uv run jupyter lab
# Run debug_episode.ipynb
```

### Screen Recording Tools
- macOS: QuickTime Player (Cmd+Shift+5)
- Terminal: `asciinema rec demo.cast` for animated terminal

---

## Key Selling Points to Highlight

1. **Not a gambling bot** - Research framework with proper ML validation
2. **Safety-first** - Circuit breakers, rate limits, position controls
3. **Reproducible** - Every experiment tracked with full configs
4. **Controlled execution-ready** - Real Binance integration (testnet proven)
5. **Clean architecture** - Domain-driven design, type safety
6. **Flexible** - 8 agent types, 4 reward functions, 50+ configs

---

## Upwork Profile Text

### Headline
"RL Trading Systems | ML Engineering | Binance Integration"

### Bio Snippet
"I build reinforcement learning trading systems with strong safety controls. My latest project is an end-to-end RL trading lab with PPO/A2C/DQN agents, custom Transformer policies, MLflow tracking, and controlled Binance testnet validation workflows. 8,000+ lines of clean, typed Python."

### Skills to Tag
- Reinforcement Learning
- Algorithmic Trading
- Python
- Machine Learning
- Binance API
- PyTorch
- MLOps
- Trading Bots

---

## Recording Checklist

- [ ] Clean terminal (no sensitive data visible)
- [ ] Use testnet credentials only
- [ ] Increase font size for readability
- [ ] Dark theme for professional look
- [ ] Close unrelated browser tabs
- [ ] Mute notifications during recording
- [ ] Practice narration 1-2 times
- [ ] Keep under 3 minutes total
