# Transformer-Based Policy for RL Trading

## Overview

This implementation adds a transformer-based policy architecture to process temporal sequences of market data. The transformer encoder captures temporal dependencies and patterns in price movements, technical indicators, and other features.

## Architecture

```
Input Observations (batch, seq_len * n_features + 4)
    ↓
Split into temporal sequence and position info
    ↓
Feature Projection: Linear(n_features → d_model)
    ↓
Add Sinusoidal Positional Encoding
    ↓
Transformer Encoder (shared)
    ├─ Multi-head Self-Attention (nhead heads)
    ├─ Feed-Forward Network (dim_feedforward)
    └─ Layer Normalization
    ↓
Sequence Aggregation (mean pooling)
    ↓
    ├──→ Policy Head (Actor): Linear layers → Action distribution
    └──→ Value Head (Critic): Linear layers → State value
```

## Key Features

- **Shared Encoder**: Single transformer encoder for both policy and value networks (parameter efficient)
- **Sinusoidal Positional Encoding**: Provides temporal information without learnable parameters
- **End-to-End Training**: All parameters trainable during RL training
- **Flexible Aggregation**: Supports mean pooling, last token, or CLS token
- **Compatible with SB3**: Works with PPO, A2C, and other SB3 algorithms

## Components

### 1. SinusoidalPositionalEncoding
- Fixed (non-learnable) positional encoding
- Sine and cosine functions at different frequencies
- Provides temporal ordering information

### 2. TransformerFeatureExtractor
- Projects input features to transformer dimension
- Applies positional encoding
- Processes sequence through transformer encoder
- Aggregates output to fixed-size representation

### 3. TransformerActorCriticPolicy
- Custom SB3 ActorCriticPolicy
- Uses TransformerFeatureExtractor as backbone
- Separate MLP heads for policy and value

## Configuration

### Basic Usage

To use the transformer policy, update your agent config:

```yaml
# configs/agent/ppo_transformer.yaml
policy: "TransformerPolicy"
policy_kwargs:
  net_arch:
    pi: [128, 64]  # Policy head layers
    vf: [128, 64]  # Value head layers

  features_extractor_class: rl_trading_lab.models.TransformerFeatureExtractor
  features_extractor_kwargs:
    lookback_window: 20    # Must match environment
    n_features: 4          # Must match observation config
    d_model: 128           # Transformer hidden dimension
    nhead: 4               # Number of attention heads
    num_encoder_layers: 2  # Number of transformer layers
    dim_feedforward: 256   # FFN dimension
    dropout: 0.1           # Dropout rate
    aggregation: "mean"    # "mean", "last", or "cls"
```

### Hyperparameters

#### Transformer Architecture
- **d_model**: Transformer hidden dimension (64, 128, 256)
  - Larger = more capacity but slower training
  - Must be divisible by nhead
- **nhead**: Number of attention heads (2, 4, 8)
  - More heads = can attend to different patterns
  - d_model must be divisible by nhead
- **num_encoder_layers**: Number of transformer layers (1, 2, 3)
  - More layers = deeper representation but slower
  - Start with 2, increase if underfitting
- **dim_feedforward**: FFN hidden dimension (128, 256, 512)
  - Typically 2-4x d_model
- **dropout**: Dropout rate (0.0, 0.1, 0.2)
  - Regularization to prevent overfitting

#### Observation Parameters
- **lookback_window**: Number of timesteps (must match environment)
- **n_features**: Number of features per timestep (must match observation config)

#### Aggregation Method
- **"mean"**: Average all sequence outputs (default, robust)
- **"last"**: Use last timestep only (focuses on recent state)
- **"cls"**: Learnable CLS token (BERT-style, more flexible)

#### Training Settings
- **learning_rate**: Use lower LR than MLP (3e-4 instead of 1e-3)
- **max_grad_norm**: Keep gradient clipping (0.5) for stability
- **ent_coef**: May need lower entropy coefficient (0.01 vs 0.1)

## Usage Examples

### 1. Train with Transformer Policy

```bash
# Use transformer with PPO
uv run python experiments/train.py agent=ppo_transformer

# Override hyperparameters
uv run python experiments/train.py \
  agent=ppo_transformer \
  agent.hyperparameters.policy_kwargs.features_extractor_kwargs.d_model=256 \
  agent.hyperparameters.policy_kwargs.features_extractor_kwargs.num_encoder_layers=3 \
  agent.hyperparameters.learning_rate=1e-4
```

### 2. Use in Code

```python
from rl_trading_lab.models import TransformerActorCriticPolicy
from stable_baselines3 import PPO

policy_kwargs = dict(
    features_extractor_class=TransformerFeatureExtractor,
    features_extractor_kwargs=dict(
        lookback_window=20,
        n_features=4,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        aggregation="mean",
    ),
    net_arch=dict(pi=[128, 64], vf=[128, 64]),
)

model = PPO(
    TransformerActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
```

### 3. Using Agent Wrapper

```python
from rl_trading_lab.config import load_config
from rl_trading_lab.agents import create_agent_from_config

# Load transformer config
config = load_config("ppo_transformer")

# Create agent (automatically handles TransformerPolicy)
agent = create_agent_from_config(config, env, eval_env)

# Train
agent.train(
    total_timesteps=1_000_000,
    eval_freq=5_000,
    n_eval_episodes=10,
)
```

## Important Considerations

### 1. Observation Format
The environment observation must be structured as:
```
[feature_1_t-20, feature_2_t-20, ..., feature_N_t-20,  # Timestep t-20
 feature_1_t-19, feature_2_t-19, ..., feature_N_t-19,  # Timestep t-19
 ...
 feature_1_t, feature_2_t, ..., feature_N_t,           # Current timestep t
 position_size, entry_price, pnl, cash_pct]            # Position info (4 dims)
```

Total dimensions = `lookback_window * n_features + 4`

### 2. Feature Matching
Ensure `n_features` in transformer config matches the number of features in your observation config:

```yaml
# configs/observation/default.yaml
input_features:
  - "ratio_sma_5_close_zscore"
  - "ratio_sma_20_close_zscore"
  - "ratio_range_close_zscore"
  - "fracdiff_0.4_zscore"
# n_features = 4

# configs/agent/ppo_transformer.yaml
features_extractor_kwargs:
  n_features: 4  # Must match!
```

### 3. Performance
- **Training Speed**: Transformers are slower than MLPs (~2-3x)
- **Memory**: Requires more GPU memory (use smaller batch_size if needed)
- **Convergence**: May need more timesteps to converge
- **Hyperparameters**: More sensitive to learning rate and architecture choices

### 4. When to Use Transformer
Use transformer when:
- ✅ Long-range temporal dependencies matter
- ✅ Complex patterns in sequence data
- ✅ Sufficient training data available
- ✅ Computational resources available

Stick with MLP when:
- ❌ Simple features or short lookback
- ❌ Limited training data
- ❌ Need fast training/inference
- ❌ Features already capture temporal info (e.g., technical indicators)

## Testing

Run the test suite to verify the implementation:

```bash
uv run python experiments/test_transformer.py
```

This tests:
1. Positional encoding
2. Feature extractor forward pass
3. Policy forward pass and gradient flow
4. Integration with SB3 PPO

## Troubleshooting

### Error: "features_extractor_class not found"
Make sure the string reference is correct:
```yaml
features_extractor_class: rl_trading_lab.models.TransformerFeatureExtractor
```

### Error: "dimension mismatch"
Check that:
- `lookback_window` matches environment
- `n_features` matches observation config
- `d_model` is divisible by `nhead`

### Training is unstable
Try:
- Lower learning rate (3e-4 → 1e-4)
- Increase gradient clipping (`max_grad_norm: 0.5`)
- Add more dropout (`dropout: 0.1 → 0.2`)
- Use smaller model (`d_model: 128 → 64`)

### Slow training
Try:
- Reduce `d_model` (256 → 128)
- Reduce `num_encoder_layers` (3 → 2)
- Reduce `dim_feedforward` (512 → 256)
- Use smaller batch size

## Architecture Details

### Parameter Count
For default config (d_model=128, nhead=4, num_encoder_layers=2):
- Feature extractor: ~266K parameters
- Policy head (128→64→actions): ~25K parameters
- Value head (128→64→1): ~25K parameters
- **Total**: ~316K parameters

Compare to MLP policy ([256, 256]):
- MLP: ~150K parameters
- Transformer: ~316K parameters (2.1x larger)

### Memory Complexity
- Self-attention: O(L²·d) where L is sequence length
- For L=20, d=128: manageable on CPU/GPU
- For longer sequences (L>100), consider reducing d_model

## References

- "Attention is All You Need" (Vaswani et al., 2017)
- Stable-Baselines3 Documentation: https://stable-baselines3.readthedocs.io/
- Transformer in RL: Decision Transformer, Trajectory Transformer, etc.

## Files Created

1. `src/rl_trading_lab/models/__init__.py` - Module initialization
2. `src/rl_trading_lab/models/transformer_policy.py` - Main implementation
3. `configs/agent/ppo_transformer.yaml` - Transformer agent config
4. `experiments/test_transformer.py` - Test suite
5. `TRANSFORMER_POLICY_GUIDE.md` - This guide

## Next Steps

1. **Tune Hyperparameters**: Experiment with d_model, num_layers, learning_rate
2. **Compare Performance**: Baseline vs Transformer on your data
3. **Ablation Studies**: Test different aggregation methods, layer counts
4. **Optimize**: Profile and optimize if training is too slow
5. **Scale Up**: Try larger models if you have compute resources

---

**Status**: ✅ Fully implemented and tested
**Branch**: `model/transformer`
**Last Updated**: October 27, 2025
