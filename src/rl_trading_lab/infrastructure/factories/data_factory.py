"""
Data factory - creates data loaders and feature pipelines from configuration.

Resolves a source_type string to the correct DataLoaderPort implementation,
and a feature pipeline type to the correct FeatureEngineeringPort implementation.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_data_loader(
    source_type: str = "parquet",
    val_split: float = 0.1,
    test_split: float = 0.1,
    required_columns: Optional[List[str]] = None,
    **kwargs,
):
    """
    Create a DataLoaderPort implementation based on source type.

    Args:
        source_type: One of "parquet", "csv", "binance_delta"
        val_split: Validation split ratio
        test_split: Test split ratio
        required_columns: Columns that must be present
        **kwargs: Additional source-specific arguments

    Returns:
        DataLoaderPort implementation

    Raises:
        ValueError: If source_type is not recognized
    """
    if source_type == "parquet":
        from rl_trading_lab.application.ports.data_loader import ParquetDataLoader

        return ParquetDataLoader(
            val_split=val_split,
            test_split=test_split,
            required_columns=required_columns,
        )

    elif source_type == "csv":
        from rl_trading_lab.infrastructure.adapters.csv_data_loader import CsvDataLoader

        return CsvDataLoader(
            val_split=val_split,
            test_split=test_split,
            required_columns=required_columns,
            **kwargs,
        )

    elif source_type == "binance_delta":
        # Lazy import - only needed if Binance data source is selected
        from rl_trading_lab.data.binance_adapter import BinanceDataAdapter

        logger.info("Using Binance Delta Lake data source")
        return BinanceDataAdapter(**kwargs)

    else:
        raise ValueError(
            f"Unknown data source type: '{source_type}'. "
            f"Valid options: 'parquet', 'csv', 'binance_delta'"
        )


def create_feature_engineering(
    pipeline_type: str = "crypto",
    feature_names: Optional[List[str]] = None,
    **kwargs,
):
    """
    Create a FeatureEngineeringPort implementation based on pipeline type.

    Args:
        pipeline_type: One of "crypto", "passthrough"
        feature_names: For "passthrough" - list of pre-existing feature names
        **kwargs: Additional pipeline-specific arguments

    Returns:
        FeatureEngineeringPort implementation

    Raises:
        ValueError: If pipeline_type is not recognized
    """
    if pipeline_type == "crypto":
        from rl_trading_lab.data.feature_pipeline import FeaturePipeline

        return FeaturePipeline(**kwargs)

    elif pipeline_type == "passthrough":
        from rl_trading_lab.application.ports.feature_engineering import PassthroughFeatures

        if not feature_names:
            raise ValueError("feature_names is required for passthrough pipeline")
        return PassthroughFeatures(feature_names=feature_names)

    else:
        raise ValueError(
            f"Unknown feature pipeline type: '{pipeline_type}'. "
            f"Valid options: 'crypto', 'passthrough'"
        )
