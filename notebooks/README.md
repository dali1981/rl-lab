# Notebooks

## debug_episode.ipynb

Debug and visualize trading environment episodes with random actions or trained agents.

### What it does:
- Loads your trading data and configuration
- Creates a test environment
- Runs episodes with either:
  - Random actions (for testing environment)
  - Trained agent (if checkpoint exists)
- Visualizes:
  - Portfolio value over time
  - Asset price movements
  - Actions taken (Buy/Hold/Sell)
  - Position sizes
  - Rewards
  - Balance vs Position Value analysis

### How to run:

1. **Start Jupyter:**
   ```bash
   cd notebooks
   uv run jupyter lab
   ```

2. **Open the notebook:**
   - Open `debug_episode.ipynb` in Jupyter
   - Run all cells (Cell → Run All)

3. **What to expect:**
   - First run: Uses random actions (no trained model needed)
   - After training: Automatically finds and loads latest checkpoint

### Requirements:

**Data file must exist:**
```bash
# Check data path in config
cat ../configs/config.yaml | grep train_data_path

# Update if needed or override in the notebook:
# config.data.train_data_path = "path/to/your/data.parquet"
```

### Fixed Issues:

✅ Updated imports (`TradingDataLoader` → `DataProcessor`)
✅ Fixed config loading (now uses Hydra properly)
✅ Corrected data loading API
✅ Auto-detects trained models in checkpoints/
✅ Handles missing models gracefully (falls back to random actions)

### Customization:

**Use different config:**
```python
# In cell 3, change overrides:
cfg = compose(config_name='config', overrides=['agent=ppo_transformer'])
```

**Use specific model:**
```python
# In cell 10, set path directly:
model_path = Path("../checkpoints/your_model/best_model.zip")
```

**Change episode length:**
```python
# In cell 8 or 11, change safety limit:
if step > 1000:  # Run longer episodes
    break
```

### Troubleshooting:

**Error: "Data file not found"**
- Update `data.train_data_path` in `configs/config.yaml`
- Or set it in the notebook after loading config

**Error: "No module named 'seaborn'"**
```bash
uv add seaborn
```

**No trained model found:**
- Normal on first run - uses random actions
- Train a model first: `uv run python experiments/train.py`
- Notebook will auto-detect checkpoints

**Plots not showing:**
- Make sure you're using Jupyter Lab or Notebook
- Check that `%matplotlib inline` is in cell 1

### Output:

The notebook generates:
- 📊 **Episode visualization** (5 subplots)
- 📈 **Balance vs Position analysis** (3 subplots)
- 📋 **Episode summary statistics**
- ⚠️ **Warnings** for open positions at episode end

### Example Output:

```
✓ Environment created
  Observation space: (84,)
  Action space: Discrete(3)
  Max steps: 991
  Hold closes position: True

Episode completed: 891 steps
Final portfolio value: $10,234.56
Final return: 2.35%
Final position: 0.0000

============================================================
EPISODE SUMMARY
============================================================
Total steps: 891
Initial balance: $10,000.00
Final portfolio value: $10,234.56
Final return: 2.35%

Action Distribution:
0    445  # Hold
1    223  # Buy
2    223  # Sell

✓ Position is FLAT at end of episode (good!)
```

---

**Ready to debug!** Run the notebook to visualize your trading environment behavior.
