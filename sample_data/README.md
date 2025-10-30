# Sample Data

This directory contains sample datasets for testing and experimentation without requiring the full dataset.

## Files

### `btcusdt_sample_10k.parquet`

**Description**: Last 10,000 rows of BTC/USDT trading data with pre-computed technical indicators.

**Source**: Extracted from `../tools/examples/btcusdt_fractional_indicators.parquet` (last 10K of 56,663 rows)

**Time Range**: 2025-10-21 11:06:35 to 2025-10-25 09:44:59 (approximately 4 days)

**Size**: ~1.2 MB

**Rows**: 10,000 (9,954 clean rows, 46 with NaN in fracdiff indicators)

**Columns** (14 total):
- **OHLCV Data**:
  - `timestamp`: Datetime of the bar
  - `open`: Opening price
  - `high`: Highest price
  - `low`: Lowest price
  - `close`: Closing price
  - `volume`: Trading volume

- **Technical Indicators (Raw)**:
  - `ratio_sma_5_close`: SMA(5) / close price
  - `ratio_sma_20_close`: SMA(20) / close price
  - `ratio_range_close`: (high - low) / close
  - `fracdiff_0.4`: Fractionally differentiated price series (d=0.4)

- **Z-Score Normalized Indicators** (Used for RL observations):
  - `ratio_sma_5_close_zscore`: Z-score of ratio_sma_5_close
  - `ratio_sma_20_close_zscore`: Z-score of ratio_sma_20_close
  - `ratio_range_close_zscore`: Z-score of ratio_range_close
  - `fracdiff_0.4_zscore`: Z-score of fracdiff_0.4

## Usage

```python
import pandas as pd

# Load sample data
df = pd.read_parquet('sample_data/btcusdt_sample_10k.parquet')

# Use in training config
# Edit configs/data/default.yaml:
# train_data_path: "../sample_data/btcusdt_sample_10k.parquet"
```

## Notes

- **High Quality Data**: Only 46 rows (~0.5%) contain NaN values in fracdiff indicators
- Most indicators are fully computed since this is the last 10K rows of the dataset
- Z-score features are computed using rolling windows for stationarity
- This is real market data (BTC/USDT) from October 2025
- Perfect for:
  - Testing ONE_TRADE mode
  - Notebook experiments and prototyping
  - Quick debugging without loading full 56K dataset
- For full training runs, use the complete dataset in `../tools/examples/`
