"""
Binance Data Adapter - Load data from MinIO/Delta Lake.

This module provides a bridge between the dlt-starter Binance data pipeline
(which stores data in MinIO using Delta Lake format) and the RL trading lab.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class BinanceDataAdapter:
    """
    Adapter to load Binance tick data from MinIO/Delta Lake storage.

    This adapter connects to the dlt-starter data infrastructure and loads
    aggregated trade data for specified symbols and date ranges.

    Data is stored in: s3://binance-data/warehouse/binance_test/{SYMBOL}/

    Example:
        >>> adapter = BinanceDataAdapter()
        >>> df = adapter.load_symbol_data("BTCUSDT", start_date="2024-01-01", days=7)
        >>> print(df.head())
    """

    def __init__(
        self,
        bucket_url: str = "s3://binance-data/warehouse",
        dataset_name: str = "binance_test",
        storage_options: Optional[dict] = None,
    ):
        """
        Initialize the Binance data adapter.

        Args:
            bucket_url: S3 bucket URL where data is stored
            dataset_name: Dataset name in the warehouse
            storage_options: S3 storage options (for MinIO)
                If None, defaults to MinIO localhost settings
        """
        self.bucket_url = bucket_url
        self.dataset_name = dataset_name

        # Default MinIO storage options
        if storage_options is None:
            self.storage_options = {
                "AWS_ACCESS_KEY_ID": "minioadmin",
                "AWS_SECRET_ACCESS_KEY": "minioadmin",
                "AWS_ENDPOINT_URL": "http://localhost:9000",
                "AWS_REGION": "us-east-1",
                "AWS_ALLOW_HTTP": "true",
            }
        else:
            self.storage_options = storage_options

        logger.info(f"Initialized BinanceDataAdapter: {bucket_url}/{dataset_name}")

    def get_table_path(self, symbol: str) -> str:
        """
        Get the Delta Lake table path for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")

        Returns:
            Full path to Delta table
        """
        return f"{self.bucket_url}/{self.dataset_name}/{symbol.upper()}"

    def load_symbol_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load aggregated trade data for a symbol from Delta Lake.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            days: Number of days to load (if start_date not provided)
                 If None and no dates specified, loads all data

        Returns:
            DataFrame with columns:
                - agg_trade_id: Aggregated trade ID
                - price: Trade price (as string, needs conversion)
                - quantity: Trade quantity (as string)
                - timestamp: Trade timestamp (milliseconds)
                - first_trade_id: First trade ID in aggregate
                - last_trade_id: Last trade ID in aggregate
                - is_buyer_maker: Whether buyer is maker
                - symbol: Trading pair symbol
                - date: Trade date (for partitioning)

        Raises:
            ImportError: If deltalake package not installed
            FileNotFoundError: If Delta table doesn't exist
            ValueError: If date parameters are invalid
        """
        try:
            from deltalake import DeltaTable
            import pyarrow.compute as pc
        except ImportError:
            raise ImportError(
                "deltalake package required. Install with: uv add deltalake pyarrow"
            )

        table_path = self.get_table_path(symbol)
        logger.info(f"Loading data from: {table_path}")

        try:
            # Load Delta table
            dt = DeltaTable(table_path, storage_options=self.storage_options)
            logger.info(f"Delta table version: {dt.version()}")

            # Build filter expression
            filters = []

            # Handle date parameters
            if start_date is None and days is not None:
                # Calculate start date from days
                end = datetime.now() if end_date is None else datetime.fromisoformat(end_date)
                start = end - timedelta(days=days)
                start_date = start.date().isoformat()

            if start_date is not None:
                filters.append(pc.field("date") >= start_date)

            if end_date is not None:
                filters.append(pc.field("date") <= end_date)

            # Combine filters
            if len(filters) == 0:
                filter_expr = None
            elif len(filters) == 1:
                filter_expr = filters[0]
            else:
                filter_expr = filters[0]
                for f in filters[1:]:
                    filter_expr = filter_expr & f

            # Load data with filter
            logger.info(f"Applying filter: {filter_expr}")
            df = dt.to_pyarrow_dataset().to_table(filter=filter_expr).to_pandas()

            logger.info(f"Loaded {len(df):,} trades for {symbol}")

            if len(df) == 0:
                logger.warning(f"No data found for {symbol} in date range")
                return df

            # Convert data types
            df['price'] = df['price'].astype(float)
            df['quantity'] = df['quantity'].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

            return df

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Delta table not found for {symbol} at {table_path}. "
                f"Make sure the data pipeline has been run. "
                f"Run: cd ../dlt-starter && uv run python examples/01_run_pipeline_example.py "
                f"--symbol {symbol} --delta"
            ) from e
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise

    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            days: Number of days to load (if start_date not provided)

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        data = {}

        for symbol in symbols:
            logger.info(f"Loading {symbol}...")
            try:
                df = self.load_symbol_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    days=days,
                )
                data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue

        logger.info(f"Loaded {len(data)}/{len(symbols)} symbols successfully")
        return data

    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols in the dataset.

        Returns:
            List of symbol names

        Note:
            This requires listing S3 objects, which may not work with all
            storage backends. Returns empty list if listing fails.
        """
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Parse bucket URL
            bucket_name = self.bucket_url.replace("s3://", "").split("/")[0]
            prefix = f"{self.dataset_name}/"

            # Create S3 client for MinIO
            s3_client = boto3.client(
                's3',
                endpoint_url=self.storage_options.get("AWS_ENDPOINT_URL"),
                aws_access_key_id=self.storage_options.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=self.storage_options.get("AWS_SECRET_ACCESS_KEY"),
                region_name=self.storage_options.get("AWS_REGION"),
            )

            # List objects
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                Delimiter='/'
            )

            # Extract symbol names from common prefixes
            symbols = []
            if 'CommonPrefixes' in response:
                for prefix_info in response['CommonPrefixes']:
                    prefix_path = prefix_info['Prefix']
                    # Extract symbol from path like "binance_test.db/BTCUSDT/"
                    symbol = prefix_path.rstrip('/').split('/')[-1]
                    symbols.append(symbol)

            logger.info(f"Found {len(symbols)} symbols: {symbols}")
            return sorted(symbols)

        except Exception as e:
            logger.warning(f"Could not list symbols: {e}")
            return []

    def get_date_range(self, symbol: str) -> tuple[str, str]:
        """
        Get the available date range for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (min_date, max_date) as strings
        """
        try:
            from deltalake import DeltaTable

            table_path = self.get_table_path(symbol)
            dt = DeltaTable(table_path, storage_options=self.storage_options)

            # Load just date column
            df = dt.to_pyarrow_dataset().to_table(columns=['date']).to_pandas()

            min_date = df['date'].min()
            max_date = df['date'].max()

            return (min_date, max_date)

        except Exception as e:
            logger.error(f"Could not get date range for {symbol}: {e}")
            raise
