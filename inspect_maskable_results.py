"""
Quick script to inspect MaskablePPO training results.
"""
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl_trading_lab.environment.trading_env import Action
from rl_trading_lab.environment.factory import create_make_env
from hydra import compose, initialize

print("="*80)
print("MaskablePPO Training Results Inspection")
print("="*80)

# Find latest checkpoint
checkpoint_dir = Path("checkpoints")
maskable_ppo_runs = sorted(checkpoint_dir.glob("MaskablePPO_*"),
                           key=lambda x: x.stat().st_mtime, reverse=True)

if not maskable_ppo_runs:
    print("\n❌ No MaskablePPO checkpoints found!")
    exit(1)

latest_run = maskable_ppo_runs[0]
model_path = latest_run / "best_model" / "best_model.zip"
vecnorm_path = latest_run / "best_model" / "vecnormalize.pkl"

print(f"\nLatest run: {latest_run.name}")
print(f"Model exists: {model_path.exists()}")
print(f"VecNormalize exists: {vecnorm_path.exists()}")

if not model_path.exists():
    print("\n⏳ Model not ready yet")
    exit(1)

# Load config and create environment
with initialize(config_path="configs", version_base=None):
    config = compose(config_name="config", overrides=["agent=maskable_ppo"])

make_env = create_make_env(
    data_path=config.data.path,
    observation_config=config.observation,
    feature_engineering_config=config.feature_engineering,
    env_config=config.env,
    val_split=config.data.val_split,
    test_split=config.data.test_split,
)

# Load model
print("\nLoading model...")
model = MaskablePPO.load(model_path)

# Create environment
train_env_raw = make_env('train')
train_env = DummyVecEnv([lambda: train_env_raw])

if vecnorm_path.exists():
    train_env = VecNormalize.load(vecnorm_path, train_env)
    train_env.training = False
    train_env.norm_reward = False

print("✓ Model and environment loaded\n")

# Run episode
print("Running episode to collect action statistics...")
obs = train_env.reset()
step = 0
max_steps = 500
actions_taken = []
invalid_actions = []

while step < max_steps:
    # Get action mask
    unwrapped_env = train_env.envs[0]
    if hasattr(unwrapped_env, 'venv'):
        trading_env = unwrapped_env.venv.envs[0]
    else:
        trading_env = unwrapped_env

    action_mask = trading_env.action_masks()
    position_before = trading_env.portfolio.position.size

    # Predict with mask
    action, _ = model.predict(obs, action_masks=action_mask, deterministic=False)

    # Check validity
    is_valid = action_mask[action[0]]
    if not is_valid:
        invalid_actions.append({
            'step': step,
            'position': position_before,
            'action': Action(action[0]).name,
        })

    # Step
    obs, reward, done, info = train_env.step(action)
    actions_taken.append(action[0])
    step += 1

    if done[0]:
        break

print(f"Collected {step} steps\n")

# Analyze action distribution
from collections import Counter
action_counts = Counter(actions_taken)
action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

print("="*80)
print("ACTION DISTRIBUTION")
print("="*80)
for action_id in sorted(action_counts.keys()):
    count = action_counts[action_id]
    pct = 100 * count / len(actions_taken)
    print(f"  {action_names[action_id]:5s}: {count:4d} ({pct:5.1f}%)")

# Check for degenerate policy
max_pct = 100 * max(action_counts.values()) / len(actions_taken)
print(f"\n{'='*80}")
if max_pct > 90:
    print(f"❌ DEGENERATE POLICY: {max_pct:.1f}% of actions are the same!")
else:
    print(f"✓ POLICY IS BALANCED: max action frequency is {max_pct:.1f}%")
print(f"{'='*80}\n")

# Check invalid actions
print(f"Invalid actions: {len(invalid_actions)}")
if invalid_actions:
    print("\n❌ WARNING: Invalid actions were sampled!")
    for inv in invalid_actions[:5]:
        print(f"  Step {inv['step']}: {inv['action']} with position={inv['position']:.2f}")
else:
    print("✓ No invalid actions sampled!\n")

# Load evaluation logs
eval_logs_path = latest_run / "eval_logs" / "evaluations.npz"
if eval_logs_path.exists():
    eval_data = np.load(eval_logs_path)
    timesteps = eval_data['timesteps']
    mean_rewards = eval_data['results'].mean(axis=1)
    mean_lengths = eval_data['ep_lengths'].mean(axis=1)

    print("="*80)
    print("TRAINING PROGRESS")
    print("="*80)
    print(f"Total timesteps: {timesteps[-1]:,}")
    print(f"Final mean reward: {mean_rewards[-1]:.6f}")
    print(f"Final mean ep length: {mean_lengths[-1]:.1f}")
    print(f"{'='*80}\n")

print("✓ Inspection complete!")
