# RL Module Reusability Assessment

**Date:** 2026-03-28
**Scope:** Evaluate whether the RL module in `rl-trading-lab` is clean enough to isolate and reuse in other projects.

---

## Architecture Overview

The codebase has a well-structured DDD-inspired architecture with clear layering:

```
domain/          -> Pure trading logic (zero framework deps)
  value_objects/ -> Position, Bar, Trade, FeatureWindow (frozen dataclasses)
  services/      -> Reward, Risk, PositionSizing (protocol-based)
  ports/         -> MarketDataPort (abstract interface)
infrastructure/  -> Gym adapter, Market data adapter, MLflow
environment/     -> Legacy TradingEnv (pre-refactor, still in use)
agents/          -> SB3 Trainer wrapper
config/          -> Pydantic-based configuration
data/            -> Binance adapter, feature pipeline
live/            -> Live trading (executor, dashboard, stream)
```

---

## What's Clean and Reusable

### Domain Layer (`domain/`)

The domain layer is genuinely portable:

- **Zero numpy/pandas/gym imports** - uses only stdlib + own value objects
- **Protocol-based interfaces**: `MarketDataPort`, `RewardCalculationService`, `RiskManagementService`, `PositionSizingService`
- **Immutable value objects**: `Position`, `CompletedTrade`, `Bar`, `FeatureWindow` with proper validation
- **Pluggable strategies**: returns vs PnL reward, Kelly vs fixed sizing, standard vs conservative risk
- **`TradingDomain`** as a clean orchestrator with no framework leakage

### Gym Adapter (`infrastructure/adapters/gym_adapter.py`)

- Clean anti-corruption layer between `TradingDomain` and Gymnasium
- Only deps: `gymnasium`, `numpy` (minimal and expected)
- Factory function `create_gym_trading_env()` for convenient setup

### Market Data Adapter (`infrastructure/adapters/market_data_adapter.py`)

- Implements `MarketDataPort` for pandas DataFrames
- Keeps pandas in infrastructure layer, domain never sees it

---

## What Blocks Clean Reuse

### 1. Two Parallel Environment Implementations

| File | Description |
|------|-------------|
| `environment/trading_env.py` | Legacy monolith: takes `pd.DataFrame` directly, mixes domain logic with gym concerns |
| `infrastructure/adapters/gym_adapter.py` | Clean refactored version wrapping `TradingDomain` |

Both exist simultaneously. Unclear which is actually used in training. The legacy `TradingEnv` duplicates domain logic (position tracking, reward calculation, risk checks) that `TradingDomain` already handles properly.

### 2. Trainer is Heavily Coupled (`agents/sb3_agents.py`)

567 lines mixing multiple concerns:

- **Hard imports**: MLflow, TensorBoard, SB3 callbacks, MaskablePPO, custom callbacks
- **Direct config coupling**: References `RootConfig`, `AgentConfig`, `EnvConfig`
- **Mixed responsibilities**: Training loop, evaluation, checkpointing, logging, environment wrapping
- **Bug**: `logger` referenced at lines 35/53 before definition at line 56

### 3. Data Layer is Binance-Specific

- `BinanceDataAdapter` hardcodes MinIO/S3 storage paths and default credentials
- `FeaturePipeline` has a fragile `sys.path.insert` hack to import from sibling `tools` package
- Features are crypto-specific (SMA ratios, fracdiff)
- `application/ports/data_loader.py` exists but is unused

### 4. Live Module is Binance-Bound (~2,474 LOC)

- `stream_consumer.py` -> Binance WebSocket
- `executor.py` -> Binance order execution
- `dashboard.py` -> Binance-specific display
- Not abstractable without major rework

### 5. Config is Monolithic

- `RootConfig` bundles everything in one object
- `DataConfig` hardcodes `train_data_path` as string
- No pluggable data source concept

### 6. Heavy Dependency Footprint

23+ runtime dependencies:

| Category | Packages |
|----------|----------|
| RL | `stable-baselines3`, `gymnasium`, `sb3-contrib`, `shimmy` |
| Data | `pandas`, `polars`, `numpy`, `pyarrow`, `deltalake`, `boto3` |
| Tracking | `mlflow`, `optuna`, `tensorboard` |
| Trading | `python-binance` |
| UI | `matplotlib`, `seaborn`, `plotly`, `rich` |
| Config | `hydra-core`, `pydantic`, `typer` |

For a reusable RL module, at least half should be optional.

---

## Extraction Plan

### What to Extract (reusable)

| Component | LOC (approx) | Dependencies |
|-----------|-------------|--------------|
| `domain/` (all) | ~620 | stdlib only |
| `infrastructure/adapters/gym_adapter.py` | ~340 | gymnasium, numpy |
| `infrastructure/adapters/market_data_adapter.py` | ~200 | pandas (optional) |

**Total: ~1,200 LOC**, deps: `gymnasium` + `numpy` required, `pandas` optional.

### What to Leave Behind (project-specific)

| Component | Reason |
|-----------|--------|
| `data/binance_adapter.py` | Binance/MinIO specific |
| `data/feature_pipeline.py` | Crypto-specific features, fragile imports |
| `live/` (all) | Binance-bound execution |
| `agents/sb3_agents.py` | Too coupled to MLflow/config/callbacks |
| `config/` | Project-specific config structure |
| `utils/` | MLflow callbacks, checkpoint manager |
| `environment/trading_env.py` | Legacy duplicate of domain logic |

---

## Recommendations

### 1. Delete `environment/trading_env.py`

The pre-refactor monolith. Migrate all callers to `GymTradingEnvAdapter` + `TradingDomain`.

### 2. Split Trainer Into Composition

Separate the training loop, callback setup, and environment wrapping. Make MLflow/TensorBoard optional hooks, not hard imports. See detailed plan below.

### 3. Make Data Loading a Port

Define a proper `DataLoaderPort` and route all data loading through it. The Binance adapter and feature pipeline should implement this port. See detailed plan below.

### 4. Thin the Dependency List

For the extracted package: `gymnasium` and `numpy` as required; everything else optional with lazy imports.

### 5. Fix the Logger Ordering Bug

In `sb3_agents.py`, lines 35/53 reference `logger` before line 56 defines it.

---

## Detailed Plans

### Plan: Split Trainer Into Composition (Recommendation #2)

**Problem:** `agents/sb3_agents.py` is a 567-line monolith that hard-imports MLflow, TensorBoard, 6+ SB3 modules, and custom callbacks. It cannot be reused without dragging in the entire dependency tree.

**Goal:** Decompose into composable pieces so another project can use the training loop without MLflow, or use MLflow without the custom checkpoint logic.

#### Step 1: Extract Environment Wrapping

Create `agents/env_wrapper.py`:

```python
class EnvWrapperBuilder:
    """Builds wrapped environments from config."""

    def build(self, env: gym.Env, vec_normalize_config, is_eval: bool) -> VecEnv:
        monitored = Monitor(env)
        vec_env = DummyVecEnv([lambda: monitored])
        if vec_normalize_config.enabled:
            vec_env = VecNormalize(vec_env, ...)
        return vec_env
```

This isolates the Monitor -> DummyVecEnv -> VecNormalize chain. Only depends on SB3 (expected).

#### Step 2: Extract Callback Factory

Create `agents/callback_factory.py`:

```python
class CallbackFactory:
    """Creates training callbacks based on config."""

    def create_eval_callback(self, eval_env, save_path, metadata, ...) -> BaseCallback
    def create_checkpoint_callback(self, save_path, metadata, ...) -> BaseCallback
    def create_trading_metrics_callback(self, one_trade_mode, ...) -> Optional[BaseCallback]
    def create_logging_callbacks(self) -> list[BaseCallback]  # MLflow, TB - lazy imports
```

MLflow and TensorBoard imports move here behind lazy `importlib` guards. If MLflow isn't installed, `create_logging_callbacks()` returns an empty list.

#### Step 3: Slim Down Trainer to Orchestration Only

The remaining `Trainer` becomes ~150 lines:

```python
class Trainer:
    def __init__(
        self,
        algo_class: type[BaseAlgorithm],
        train_env: VecEnv,
        eval_env: Optional[VecEnv],
        hyperparams: dict,
        callbacks: list[BaseCallback],
        save_path: Path,
    ): ...

    def train(self, total_timesteps: int) -> dict: ...
    def evaluate(self, env, n_episodes) -> dict: ...
    def save(self, path): ...
    def load(self, path): ...
```

No config objects, no MLflow, no callback construction. Takes pre-built components.

#### Step 4: Create a Convenience Factory

For backward compatibility and ease of use in this project:

```python
class TrainerFactory:
    """Project-specific factory that wires everything together."""

    @staticmethod
    def from_config(
        agent_config: AgentConfig,
        env_config: EnvConfig,
        make_env: Callable,
    ) -> Trainer:
        wrapper = EnvWrapperBuilder()
        callbacks = CallbackFactory()
        # ... wire together and return Trainer
```

#### File Changes

| Action | File | Notes |
|--------|------|-------|
| Create | `agents/env_wrapper.py` | ~60 LOC, SB3 deps only |
| Create | `agents/callback_factory.py` | ~120 LOC, lazy MLflow/TB imports |
| Rewrite | `agents/sb3_agents.py` | ~150 LOC, pure orchestration |
| Create | `agents/trainer_factory.py` | ~80 LOC, project-specific wiring |
| Update | `experiments/train.py` | Use `TrainerFactory.from_config()` |

---

### Plan: Make Data Loading a Port (Recommendation #3)

**Problem:** Data loading is hardcoded to Binance/MinIO via `BinanceDataAdapter`. The application port `data_loader.py` exists but is unused. Feature engineering is coupled to crypto-specific indicators. A new project wanting to use this RL module with different data (equities, forex, CSV files) would need to rewrite the data layer from scratch.

**Goal:** Define a clean data loading interface that the domain/training code programs against, with Binance as one implementation.

#### Step 1: Define the DataLoaderPort Properly

Update `application/ports/data_loader.py`:

```python
class DataLoaderPort(Protocol):
    """Port for loading training/evaluation data."""

    def load_train_data(self) -> pd.DataFrame:
        """Load training data as OHLCV+ DataFrame."""
        ...

    def load_eval_data(self) -> pd.DataFrame:
        """Load evaluation/validation data."""
        ...

    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        ...

    @property
    def feature_columns(self) -> list[str]:
        """List of feature column names available."""
        ...

    @property
    def price_column(self) -> str:
        """Name of the price column used for execution."""
        ...
```

This is the contract that training code depends on.

#### Step 2: Define a FeatureEngineeringPort

Create `application/ports/feature_engineering.py`:

```python
class FeatureEngineeringPort(Protocol):
    """Port for transforming raw OHLCV data into ML features."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature columns to DataFrame."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Names of features this pipeline produces."""
        ...
```

This decouples the crypto-specific `FeaturePipeline` from the training loop.

#### Step 3: Implement Adapters

Rename and restructure existing implementations:

```
infrastructure/
  adapters/
    data/
      binance_delta_loader.py    # Current BinanceDataAdapter -> implements DataLoaderPort
      parquet_file_loader.py     # New: simple local file loader -> implements DataLoaderPort
      csv_file_loader.py         # New: CSV loader for quick testing
    features/
      crypto_feature_pipeline.py # Current FeaturePipeline -> implements FeatureEngineeringPort
      passthrough_features.py    # New: no-op for pre-computed features
    market_data_adapter.py       # Existing (already clean)
    gym_adapter.py               # Existing (already clean)
```

The `parquet_file_loader.py` implementation would be ~40 LOC:

```python
class ParquetFileLoader(DataLoaderPort):
    def __init__(self, train_path: str, eval_path: str, test_path: str,
                 feature_cols: list[str], price_col: str = "close"):
        ...

    def load_train_data(self) -> pd.DataFrame:
        return pd.read_parquet(self._train_path)
```

#### Step 4: Wire Through Config

Add a data source discriminator to config:

```python
class DataSourceConfig(BaseModel):
    source_type: Literal["binance_delta", "parquet_files", "csv_files"]
    # Source-specific params as optional fields
    train_path: Optional[str] = None
    eval_path: Optional[str] = None
    # Binance-specific
    bucket_url: Optional[str] = None
    symbol: Optional[str] = None
    ...
```

A factory resolves the config to the right adapter:

```python
def create_data_loader(config: DataSourceConfig) -> DataLoaderPort:
    if config.source_type == "binance_delta":
        return BinanceDeltaLoader(...)
    elif config.source_type == "parquet_files":
        return ParquetFileLoader(...)
    ...
```

#### Step 5: Update Training Entry Points

`experiments/train.py` and `run_pipeline.py` call `create_data_loader(config.data)` instead of directly instantiating `BinanceDataAdapter`.

#### File Changes

| Action | File | Notes |
|--------|------|-------|
| Rewrite | `application/ports/data_loader.py` | Define proper Protocol |
| Create | `application/ports/feature_engineering.py` | ~30 LOC |
| Move | `data/binance_adapter.py` -> `infrastructure/adapters/data/binance_delta_loader.py` | Implement DataLoaderPort |
| Move | `data/feature_pipeline.py` -> `infrastructure/adapters/features/crypto_feature_pipeline.py` | Implement FeatureEngineeringPort |
| Create | `infrastructure/adapters/data/parquet_file_loader.py` | ~60 LOC |
| Create | `infrastructure/adapters/data/csv_file_loader.py` | ~50 LOC |
| Create | `infrastructure/adapters/features/passthrough_features.py` | ~20 LOC |
| Create | `infrastructure/factories/data_factory.py` | ~40 LOC |
| Update | `config/data.py` | Add `source_type` discriminator |
| Update | `experiments/train.py` | Use factory |
| Update | `run_pipeline.py` | Use factory |
