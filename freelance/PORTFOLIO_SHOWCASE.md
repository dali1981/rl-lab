# Portfolio Showcase

Screenshots, code samples, and case studies to demonstrate expertise.

---

## 1. System Architecture

### Architecture Diagram

Use this diagram from the README for portfolio presentations:

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                                │
│  Raw Ticks → Dollar Volume Bars → Technical Indicators           │
│             → Z-Score Normalization → ML-Ready Features          │
│                                                                   │
│  Tools: Kedro, MinIO, Delta Lake                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     RL TRAINING LAB                              │
│  Features → Trading Environment → RL Agents → Trained Models     │
│                                                                   │
│  Algorithms: PPO, A2C, DQN, SAC                                  │
│  Tracking: MLflow, TensorBoard                                   │
│  Management: Hydra configs, CheckpointManager                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     LIVE TRADING                                 │
│  WebSocket Stream → Dollar Bars → Features → Model → Orders     │
│                                                                   │
│  Safety: Circuit breakers, Rate limits, Risk management          │
│  Platform: Binance Testnet/Mainnet                              │
└─────────────────────────────────────────────────────────────────┘
```

**What this demonstrates:**
- End-to-end system design
- Production thinking (not just research)
- Integration of multiple technologies

---

## 2. Code Samples

### Sample 1: Safety Guard System

From `src/rl_trading_lab/live/safety.py`:

```python
@dataclass
class SafetyConfig:
    """Configuration for safety guards."""
    max_drawdown_pct: float = 0.20  # Stop at 20% drawdown
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    max_consecutive_losses: int = 5
    min_balance_usd: float = 100.0
    trading_hours: tuple[int, int] | None = None  # (start_hour, end_hour)


class SafetyGuard:
    """Multi-layer safety system for live trading."""

    def __init__(self, config: SafetyConfig, initial_balance: float):
        self.config = config
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.consecutive_losses = 0
        self.trades_this_hour = 0
        self.trades_today = 0
        self.circuit_breaker_triggered = False

    def check_all(self, current_balance: float) -> tuple[bool, str]:
        """Run all safety checks. Returns (is_safe, reason)."""

        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker active"

        # Check drawdown
        drawdown = (self.peak_balance - current_balance) / self.peak_balance
        if drawdown >= self.config.max_drawdown_pct:
            self.circuit_breaker_triggered = True
            return False, f"Max drawdown exceeded: {drawdown:.1%}"

        # Check consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"Too many consecutive losses: {self.consecutive_losses}"

        # Check rate limits
        if self.trades_this_hour >= self.config.max_trades_per_hour:
            return False, "Hourly trade limit reached"

        if self.trades_today >= self.config.max_trades_per_day:
            return False, "Daily trade limit reached"

        # Check minimum balance
        if current_balance < self.config.min_balance_usd:
            return False, f"Balance too low: ${current_balance:.2f}"

        return True, "All checks passed"
```

**What this demonstrates:**
- Production-quality code with type hints
- Dataclass configuration pattern
- Multi-layer safety architecture
- Clear documentation and naming

---

### Sample 2: Model Checkpoint with Embedded Config

From `src/rl_trading_lab/utils/checkpoint_manager.py`:

```python
class CheckpointManager:
    """Manages model checkpoints with embedded configuration."""

    @staticmethod
    def save_checkpoint(
        model: BaseAlgorithm,
        path: Path,
        config: DictConfig,
        vecnormalize: VecNormalize | None = None,
    ) -> None:
        """Save model with embedded training configuration."""

        path.mkdir(parents=True, exist_ok=True)

        # Save the model
        model.save(path / "model.zip")

        # Save VecNormalize if present
        if vecnormalize is not None:
            vecnormalize.save(path / "vecnormalize.pkl")

        # Embed configuration in checkpoint
        config_dict = OmegaConf.to_container(config, resolve=True)
        metadata = {
            "config": config_dict,
            "timestamp": datetime.now().isoformat(),
            "observation_features": config.observation.input_features,
            "reward_type": config.env.environment_params.reward_type,
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    @staticmethod
    def get_training_config(checkpoint_path: Path) -> dict:
        """Retrieve training configuration from checkpoint."""

        metadata_path = checkpoint_path.parent / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            return {
                "source": "embedded",
                **metadata["config"],
            }

        # Fallback to MLflow if available
        return CheckpointManager._try_mlflow_config(checkpoint_path)
```

**What this demonstrates:**
- Self-contained deployment pattern
- Metadata preservation
- Graceful fallback handling
- Clean API design

---

### Sample 3: Real-Time Feature Computation

From `src/rl_trading_lab/live/features.py`:

```python
class StreamingFeatureComputer:
    """Compute features in real-time matching training data format."""

    def __init__(self, config: FeatureConfig, training_stats: dict):
        self.config = config
        self.training_stats = training_stats  # mean/std from training
        self.price_buffer = deque(maxlen=200)  # Rolling window

    def compute_features(self, bar: Bar) -> np.ndarray:
        """Compute features for a single bar."""

        self.price_buffer.append(bar.close)
        prices = np.array(self.price_buffer)

        features = {}

        # SMA ratios
        for period in [5, 20, 50, 200]:
            if len(prices) >= period:
                sma = np.mean(prices[-period:])
                features[f"sma_{period}_ratio"] = bar.close / sma
            else:
                features[f"sma_{period}_ratio"] = 1.0

        # Volatility
        if len(prices) >= 20:
            features["range_ratio"] = (bar.high - bar.low) / bar.close
        else:
            features["range_ratio"] = 0.0

        # Normalize using training statistics
        normalized = self._normalize(features)

        return np.array([normalized[f] for f in self.config.feature_order])

    def _normalize(self, features: dict) -> dict:
        """Z-score normalize using training statistics."""

        normalized = {}
        for name, value in features.items():
            mean = self.training_stats[name]["mean"]
            std = self.training_stats[name]["std"]
            normalized[name] = (value - mean) / (std + 1e-8)

        return normalized
```

**What this demonstrates:**
- Real-time computation matching training
- Rolling window implementation
- Proper normalization handling
- Training/inference consistency

---

## 3. Screenshots to Capture

### MLflow Dashboard
- Experiment comparison view
- Metric charts (reward, Sharpe ratio)
- Hyperparameter table
- Model registry

### Rich Terminal Dashboard
Run: `uv run python examples/live_trading_example.py trade --model <path>`

Capture:
- Portfolio overview panel
- Active positions
- Model predictions
- Safety status
- Trade history

### TensorBoard
- Training curves
- Episode reward over time
- Loss graphs

### Code Structure
Screenshot of `src/rl_trading_lab/` directory tree showing organization

---

## 4. Case Study Template

### Case Study: Production RL Trading System

**Client Challenge:**
Build an automated trading system that can:
- Learn from historical market data
- Execute trades on Binance
- Protect capital with safety features
- Run 24/7 with minimal oversight

**Solution Delivered:**

**1. Data Pipeline**
- Processed tick data into dollar volume bars
- Engineered 20+ technical features
- Z-score normalization for stationarity

**2. RL Training System**
- Custom Gymnasium trading environment
- Trained PPO, SAC, DQN agents
- MLflow tracking for reproducibility
- Self-contained checkpoints

**3. Live Deployment**
- WebSocket streaming from Binance
- Real-time feature computation
- Model inference with VecNormalize

**4. Safety Architecture**
- Circuit breaker at 20% drawdown
- Rate limiting (10 trades/hour)
- Consecutive loss limits
- Minimum balance checks

**Results:**
- System deployed to Binance testnet
- 7,600+ lines of production code
- 100+ experiments tracked
- Full documentation delivered

**Technologies Used:**
- Python, PyTorch, Stable-Baselines3
- MLflow, Hydra, TensorBoard
- Polars, Delta Lake, MinIO
- python-binance, WebSocket

---

## 5. Video Demo Script

### 2-Minute Demo Video

**0:00-0:15 - Introduction**
"I build production-grade trading bots with reinforcement learning. Let me show you what that looks like."

**0:15-0:45 - Architecture**
Show architecture diagram.
"The system has three layers: data pipeline, training lab, and live trading. Each is designed for production use."

**0:45-1:15 - Training**
Show MLflow UI.
"I use MLflow to track every experiment. Here you can see different algorithms compared, with metrics like Sharpe ratio and returns."

**1:15-1:45 - Live Dashboard**
Show Rich terminal dashboard.
"When deployed, you get a real-time dashboard showing portfolio value, positions, predictions, and safety status."

**1:45-2:00 - Call to Action**
"If you need a trading system that actually works in production, let's talk."

---

## 6. Technical Differentiators

### What Sets This Apart

| Feature | Typical Bots | This System |
|---------|--------------|-------------|
| Sampling | Time bars | Dollar volume bars |
| Strategy | Rule-based | RL-learned |
| Config | Hardcoded | Embedded in checkpoint |
| Safety | Basic | Multi-layer |
| Deployment | Manual | Self-contained |
| Monitoring | Logs | Real-time dashboard |

### Talking Points

1. **"Self-contained checkpoints"**
   - Models embed their configuration
   - Deploy without MLflow server
   - No "what features did I train with?" confusion

2. **"Information-driven sampling"**
   - Dollar volume bars, not time bars
   - More statistically sound
   - Based on academic research

3. **"Production safety architecture"**
   - Not an afterthought
   - Circuit breakers, rate limits, position controls
   - Multi-layer protection

4. **"Real-time feature consistency"**
   - Live features match training exactly
   - Uses training statistics for normalization
   - No distribution shift

---

## 7. Testimonial Templates

*To request from future clients:*

### Template 1: Technical
"[Your name] built a sophisticated RL trading system for our crypto trading operation. The safety features gave us confidence to deploy with real capital. The code quality was exceptional - well-documented and easy to extend."

### Template 2: Process
"Working with [your name] was seamless. They understood our requirements quickly, delivered ahead of schedule, and the system has been running reliably for [X] months."

### Template 3: Results
"The trading bot [your name] built has been [performing well / exceeded expectations]. The MLflow integration makes it easy to experiment with new approaches. Highly recommend for any quantitative trading project."

---

## 8. Quick Stats for Marketing

Use these numbers:

- **7,600+** lines of production code
- **4** RL algorithms supported (PPO, SAC, DQN, A2C)
- **5** reward functions (returns, Sharpe, Sortino, PnL, Calmar)
- **6** safety guard layers
- **Live** Binance testnet deployment
- **Self-contained** checkpoints (offline-capable)

---

## 9. GitHub Repository Preparation

### Make Public Showcase

Consider creating a public "lite" version:
- Remove proprietary strategies
- Keep architecture and safety code
- Add detailed README
- Include example notebooks

### Repository Badges

```markdown
![Python](https://img.shields.io/badge/python-3.12+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style](https://img.shields.io/badge/code%20style-black-black)
```

### Demo Mode

Add a demo mode that:
- Uses sample data
- Shows dashboard functionality
- Demonstrates safety guards
- No real trading

---

## 10. LinkedIn Content Ideas

### Post 1: Architecture
"Just finished building a production RL trading system. Here's the architecture...
[Image: Architecture diagram]
Key insight: Safety features aren't optional. Circuit breakers saved us from flash crash scenarios."

### Post 2: Technical Deep Dive
"Why I use dollar volume bars instead of time bars for trading ML:
1. More normally distributed returns
2. Market-adaptive sampling
3. Better features for RL
[Link to explanation]"

### Post 3: Lessons Learned
"5 things I learned building a live trading bot:
1. Backtest performance ≠ live performance
2. Safety features first, not last
3. Self-contained deployments win
4. Log everything
5. Start on testnet, stay on testnet longer than you think"
