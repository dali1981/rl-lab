# DDD & Clean Architecture Review

**Date**: 2024-11-29
**Reviewer**: Architecture Review
**References**:
- Evans, Eric. "Domain-Driven Design: Tackling Complexity in the Heart of Software" (2003)
- Vernon, Vaughn. "Implementing Domain-Driven Design" (2013)
- Fowler, Martin. "Patterns of Enterprise Application Architecture" (2002)
- Martin, Robert C. "Clean Architecture: A Craftsman's Guide to Software Structure and Design" (2017)

---

## Executive Summary

The codebase demonstrates **solid foundational architecture** with clear separation of concerns and good use of configuration patterns. However, there are several areas where it deviates from canonical DDD and Clean Architecture principles. The issues range from domain model purity violations to dependency direction problems and missing strategic patterns.

---

## 1. Domain Model Analysis

### 1.1 Aggregate Design Issues

**Reference**: Evans, DDD pp. 125-129 (Aggregates)

**Current State**: `TradingEnv` acts as the primary aggregate root, containing `Portfolio` which itself contains `Position` and `Cash`.

**Issues**:

| Issue | Location | Problem |
|-------|----------|---------|
| **Aggregate boundary violation** | `trading_env.py:60` | `TradingEnv` stores a mutable `df.copy()` inside the aggregate. DataFrames are infrastructure concerns, not domain objects. |
| **Inconsistent invariant enforcement** | `portfolio.py:344-358` | `Cash.debit()` allows balance to go negative without raising an exception. Domain invariant "cash cannot be negative" is not enforced. |
| **Mixed responsibilities** | `trading_env.py:112-132` | NaN cleaning logic (infrastructure) embedded in domain constructor. |

**Recommendation**: Extract data access behind a Repository, enforce invariants strictly in domain.

```python
# Current (violates aggregate purity)
class TradingEnv:
    def __init__(self, df: pd.DataFrame, ...):
        self.df = df.copy()  # Infrastructure leaks into domain

# Should be
class TradingEnv:
    def __init__(self, market_data: MarketDataPort, ...):  # Port abstraction
        self.market_data = market_data
```

---

### 1.2 Entity vs Value Object Classification

**Reference**: Evans, DDD pp. 97-103 (Entities), pp. 103-109 (Value Objects)

**Current State**:

| Object | Classified As | Should Be | Issue |
|--------|--------------|-----------|-------|
| `Position` | `@dataclass` (mutable) | **Value Object** | Position should be immutable; changes create new positions |
| `Cash` | Class with state | **Value Object** or **Entity** | Ambiguous identity semantics |
| `Trade` (in history) | Dict | **Value Object** | Round-trip trades are immutable facts |
| `Action` | `IntEnum` | ✅ Correct | Good use of enum for constrained values |

**Issue at `portfolio.py:45-52`**:
```python
@dataclass
class Position:
    size: float = 0.0
    entry_price: float = 0.0
    # ... mutable dataclass
```

Per Vernon (IDDD, Ch. 5): "Value Objects should be immutable." Position represents a snapshot of state that should not change after creation.

**Recommendation**:
```python
@dataclass(frozen=True)
class Position:
    size: float
    entry_price: float
    entry_bar: int

    def with_new_bar(self, bar: int) -> 'Position':
        return Position(self.size, self.entry_price, self.entry_bar, bar)
```

---

### 1.3 Ubiquitous Language Gaps

**Reference**: Evans, DDD pp. 24-30 (Ubiquitous Language)

| Term in Code | Domain Term | Mismatch |
|--------------|-------------|----------|
| `execute_trade()` | Should be `open_position()` / `close_position()` | Trade = complete round-trip, not a single order |
| `signal` parameter | Action or Order Intent | "Signal" is technical analysis jargon, not trading domain language |
| `step` | Bar/Candle/Period | RL terminology mixed with trading terminology |
| `df` | MarketData or PriceHistory | Technical implementation detail |

**Example at `portfolio.py:270`**:
```python
def execute_trade(self, signal: float, ...):  # "signal" is TA language
```

Should use domain language:
```python
def process_order(self, order: Order, ...):  # Trading domain language
```

---

## 2. Layered Architecture Assessment

### 2.1 Dependency Rule Violations

**Reference**: Martin, Clean Architecture pp. 203-207 (The Dependency Rule)

> "Source code dependencies must point only inward, toward higher-level policies."

**Current Dependency Flow**:
```
Presentation (train.py) → Application (Trainer) → Domain (TradingEnv) → Infrastructure (pandas)
                                                          ↑
                                                      VIOLATION
```

**Violations Found**:

| Location | Violation | Severity |
|----------|-----------|----------|
| `trading_env.py:8` | `import pandas as pd` | **HIGH** - Domain depends on infrastructure |
| `trading_env.py:9` | `import gymnasium as gym` | **MEDIUM** - External framework in domain |
| `portfolio.py:39` | `import pandas as pd` | **MEDIUM** - Only used for timestamps |
| `feature_pipeline.py:15-20` | Domain logic mixed with pandas operations | **HIGH** |

**Correct layering should be**:
```
┌──────────────────────────────────────────────────────────────┐
│  Frameworks & Drivers (train.py, Gymnasium, SB3, MLflow)     │
├──────────────────────────────────────────────────────────────┤
│  Interface Adapters (Trainer, DataProcessor, OrderExecutor)  │
├──────────────────────────────────────────────────────────────┤
│  Application Services (use cases, orchestration)             │
├──────────────────────────────────────────────────────────────┤
│  Domain (TradingEnv, Portfolio, Position) - NO pandas here   │
└──────────────────────────────────────────────────────────────┘
```

---

### 2.2 Anti-Corruption Layer Issues

**Reference**: Evans, DDD pp. 364-370 (Anti-Corruption Layer)

**Good**: `BinanceDataAdapter` (`binance_adapter.py`) properly isolates external data source.

**Issues**:

1. **Missing ACL for Gymnasium** at `trading_env.py:26`:
   ```python
   class TradingEnv(gym.Env):  # Domain directly extends external framework
   ```

   Should be:
   ```python
   # Domain layer
   class TradingDomain:
       """Pure domain logic, no gym dependency"""

   # ACL layer
   class GymTradingEnvAdapter(gym.Env):
       """Adapts domain to gymnasium interface"""
       def __init__(self, domain: TradingDomain): ...
   ```

2. **Missing ACL for SB3** - `Trainer` class directly returns SB3 models and uses VecEnv internally without translation.

---

## 3. Repository Pattern Analysis

**Reference**: Fowler, PEAA pp. 322-327 (Repository)

### Current State
There is **no explicit Repository pattern**. Data access is scattered:

| Component | Data Access Method | Issue |
|-----------|-------------------|-------|
| `DataProcessor` | Direct `pd.read_parquet()` | No abstraction |
| `BinanceDataAdapter` | Direct Delta Lake access | OK as infrastructure |
| `TradingEnv` | `self.df.iloc[idx]` | Domain accesses raw data |
| `PortfolioManager` | Direct SQLite access | Infrastructure in domain |

### Missing Repositories

1. **MarketDataRepository**: Abstract price/bar access
   ```python
   class MarketDataRepository(Protocol):
       def get_bar(self, index: int) -> Bar: ...
       def get_price(self, index: int, column: str) -> float: ...
       def get_feature_window(self, start: int, end: int) -> FeatureWindow: ...
   ```

2. **TradeHistoryRepository**: Abstract trade persistence
   ```python
   class TradeHistoryRepository(Protocol):
       def save(self, trade: CompletedTrade) -> None: ...
       def get_all(self) -> List[CompletedTrade]: ...
   ```

---

## 4. Application Services Assessment

**Reference**: Vernon, IDDD pp. 110-115 (Application Services)

### Current Structure

`Trainer` (`sb3_agents.py`) acts as the primary application service but violates several principles:

**Issues**:

1. **Too many responsibilities** - `Trainer` handles:
   - Environment creation and wrapping
   - Agent configuration
   - Training orchestration
   - Evaluation
   - Model persistence
   - Logger configuration

2. **Domain logic in application service** at `sb3_agents.py:354-361`:
   ```python
   # Application service deciding domain logic (one_trade_mode)
   if not one_trade_mode:
       trading_callback = TradingMetricsCallback(...)
   ```

3. **Missing Use Case classes** - Per Vernon: "Each use case becomes an Application Service method."

   Should have:
   ```python
   class TrainAgentUseCase:
       def execute(self, config: TrainingConfig) -> TrainingResult: ...

   class EvaluateAgentUseCase:
       def execute(self, model: Model, environment: Environment) -> EvaluationResult: ...
   ```

---

## 5. Domain Services Analysis

**Reference**: Evans, DDD pp. 104-107 (Domain Services)

### Missing Domain Services

Several operations don't belong to entities but are domain logic:

1. **PositionSizingService** - `calculate_position_size()` at `portfolio.py:225-241` should be a domain service as it represents trading policy, not portfolio state.

2. **RewardCalculationService** - Reward logic at `trading_env.py:336-361` is domain policy, not environment mechanics.

3. **RiskManagementService** - Termination conditions at `trading_env.py:363-383` encode risk rules that should be explicit.

---

## 6. Configuration & Dependency Injection

**Reference**: Martin, Clean Architecture pp. 229-233 (The Humble Object Pattern)

### Good Practices ✅

1. **Pydantic configs** (`config/main.py`) - Type-safe, validated configuration
2. **Hydra integration** - Composable configuration
3. **Discriminated unions** for agents - Good pattern for polymorphic config

### Issues

1. **No DI Container** - Dependencies are created inline:
   ```python
   # train.py:568-574 - Manual wiring
   trainer = Trainer(
       agent_config=config.agent,
       env_config=config.env,
       make_env=make_env,  # Factory passed as dependency ✅
       ...
   )
   ```

2. **Hidden dependencies** at `sb3_agents.py:14-15`:
   ```python
   import mlflow  # Global import, not injected
   ```

3. **Configuration coupled to implementation** - `RootConfig` knows about specific agent types (PPO, A2C, DQN), violating OCP.

---

## 7. Bounded Context Analysis

**Reference**: Evans, DDD pp. 336-343 (Bounded Contexts), Vernon IDDD Ch. 4

### Implicit Bounded Contexts Identified

1. **Training Context** - RL agent training (SB3, VecEnv, callbacks)
2. **Trading Context** - Market simulation (Portfolio, Position, trades)
3. **Live Trading Context** - Binance integration (executor, stream consumer)
4. **Data Context** - Feature engineering, bar creation

### Context Mapping Issues

1. **No explicit context boundaries** - All contexts share the same `TradingEnv`:
   - Training uses it with VecNormalize
   - Backtesting uses it directly
   - Both expect different behaviors (randomization, etc.)

2. **Shared Kernel anti-pattern** at `live/portfolio.py` vs `environment/portfolio.py`:
   - Two different `Portfolio` classes
   - Similar concepts but different implementations
   - No explicit translation or shared abstraction

3. **Missing Context Map** - No documentation of how contexts relate (Customer/Supplier, Conformist, etc.)

---

## 8. CQRS & Event Considerations

**Reference**: Vernon, IDDD pp. 389-412 (CQRS/Event Sourcing)

### Current State
The codebase uses a traditional mutable state model.

### Opportunities

1. **Trade Events** - `trade_history` at `portfolio.py:174` stores dicts, could be domain events:
   ```python
   class PositionOpened(DomainEvent):
       position_id: str
       symbol: str
       size: float
       entry_price: float

   class PositionClosed(DomainEvent):
       position_id: str
       exit_price: float
       realized_pnl: float
   ```

2. **Query optimization** - Separation of `PortfolioManager.get_stats()` (query) from `record_trade()` (command) is implicit but could be explicit.

---

## 9. Specific Code Quality Issues

### 9.1 TODOs Indicating Design Debt

| Location | TODO | Design Issue |
|----------|------|--------------|
| `trading_env.py:113` | "This should be done BEFORE splitting data" | Data cleaning in wrong layer |
| `trading_env.py:277-280` | "Investigate if keeping both..." | Unclear domain model |

### 9.2 Magic Numbers

| Location | Value | Should Be |
|----------|-------|-----------|
| `trading_env.py:361` | `np.clip(reward, -10.0, 10.0)` | Named constant or config |
| `trading_env.py:373` | `0.2` (80% loss threshold) | Part of risk config |
| `trading_env.py:380` | `0.3` (30% drawdown) | Part of risk config |

---

## 10. Recommendations Summary

### High Priority

1. **Extract domain from pandas** - Create `MarketDataPort` interface, move pandas to infrastructure
2. **Make Position immutable** - Convert to frozen dataclass
3. **Create Anti-Corruption Layer for Gymnasium** - Separate domain logic from RL framework
4. **Introduce Repository pattern** - Abstract data access from domain

### Medium Priority

5. **Split Trainer into focused services** - TrainingService, EvaluationService, CheckpointService
6. **Make domain services explicit** - PositionSizing, RewardCalculation, RiskManagement
7. **Document Bounded Contexts** - Create context map, define explicit boundaries
8. **Enforce domain invariants** - Cash should throw on negative balance

### Lower Priority

9. **Consider domain events** - Trade events for better audit trail
10. **Add DI container** - For better testability and configuration
11. **Ubiquitous language alignment** - Rename `signal` → `order_intent`, `step` → `bar`

---

## 11. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│  train.py, live_trading.py, CLI, Dashboard                      │
├─────────────────────────────────────────────────────────────────┤
│                    APPLICATION LAYER                             │
│  TrainAgentUseCase, EvaluateAgentUseCase, BacktestUseCase       │
│  (orchestrates domain objects, no business logic)                │
├─────────────────────────────────────────────────────────────────┤
│                      DOMAIN LAYER                                │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────────────┐   │
│  │ Trading Env │  │  Portfolio  │  │   Domain Services      │   │
│  │ (Aggregate) │  │  (Entity)   │  │ - PositionSizing       │   │
│  │             │  │             │  │ - RewardCalculation    │   │
│  │  Position   │  │    Cash     │  │ - RiskManagement       │   │
│  │ (Value Obj) │  │ (Value Obj) │  └────────────────────────┘   │
│  └─────────────┘  └─────────────┘                                │
│                                                                  │
│  Ports (Interfaces):                                             │
│  MarketDataPort, TradeHistoryPort, OrderExecutionPort            │
├─────────────────────────────────────────────────────────────────┤
│                   INFRASTRUCTURE LAYER                           │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────────┐  │
│  │  SB3Adapter  │  │ DataAdapter │  │   BinanceAdapter       │  │
│  │ (Gymnasium)  │  │ (Parquet)   │  │   (Live Trading)       │  │
│  └──────────────┘  └─────────────┘  └────────────────────────┘  │
│                                                                  │
│  Implementations of Ports:                                       │
│  ParquetMarketDataAdapter, SQLiteTradeHistoryAdapter             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Implementation Roadmap

### Phase 1: Domain Purity (Week 1-2)
- [ ] Create `MarketDataPort` protocol
- [ ] Create `ParquetMarketDataAdapter` implementation
- [ ] Refactor `TradingEnv` to use port instead of DataFrame
- [ ] Make `Position` immutable (frozen dataclass)
- [ ] Enforce `Cash` invariants (throw on negative balance)

### Phase 2: Layer Separation (Week 3-4)
- [ ] Create `TradingDomain` pure domain class
- [ ] Create `GymTradingEnvAdapter` ACL for gymnasium
- [ ] Extract domain services (PositionSizing, RewardCalculation, RiskManagement)
- [ ] Move magic numbers to configuration

### Phase 3: Application Layer (Week 5-6)
- [ ] Create `TrainAgentUseCase` class
- [ ] Create `EvaluateAgentUseCase` class
- [ ] Split `Trainer` responsibilities
- [ ] Add proper DI for MLflow, TensorBoard

### Phase 4: Strategic Design (Week 7-8)
- [ ] Document Bounded Contexts
- [ ] Create Context Map
- [ ] Resolve `Portfolio` duplication between contexts
- [ ] Consider domain events for trade history

---

## Appendix A: File-by-File Issues

| File | Primary Issues | Severity |
|------|---------------|----------|
| `environment/trading_env.py` | DataFrame in domain, gym inheritance, magic numbers | HIGH |
| `environment/portfolio.py` | Mutable Position, weak invariants, pandas import | MEDIUM |
| `agents/sb3_agents.py` | Too many responsibilities, hidden dependencies | MEDIUM |
| `data/feature_pipeline.py` | Mixed domain/infrastructure | MEDIUM |
| `live/portfolio.py` | Duplicate concept, SQLite in domain | MEDIUM |
| `live/executor.py` | No ACL for Binance client | LOW |
| `config/main.py` | Coupled to specific implementations | LOW |

---

## Appendix B: Glossary Alignment

| Current Term | Proposed Domain Term | Rationale |
|--------------|---------------------|-----------|
| `step` | `bar` or `period` | Trading domain language |
| `signal` | `order_intent` | Clearer intent |
| `execute_trade` | `process_order` | Trade = round-trip |
| `df` | `market_data` | Abstract the structure |
| `features_to_use` | `observation_features` | RL-specific but clearer |
| `current_step` | `current_bar_index` | Trading context |
