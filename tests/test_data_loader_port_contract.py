"""Layer 3 data-loader port contract checks."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from rl_trading_lab.infrastructure.factories import create_data_loader


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "sample_data" / "btcusdt_sample_10k.parquet"
REQUIRED_OHLCV = {"timestamp", "open", "high", "low", "close", "volume"}


def _build_parquet_loader():
    return create_data_loader(
        source_type="parquet",
        val_split=0.2,
        test_split=0.1,
        required_columns=["open", "high", "low", "close", "volume"],
    )


def _assert_frame_contract(df: pd.DataFrame) -> None:
    assert not df.empty
    assert REQUIRED_OHLCV.issubset(set(df.columns))
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].is_monotonic_increasing


def test_data_loader_factory_resolves_parquet_port_contract() -> None:
    loader = _build_parquet_loader()
    for method_name in ("load", "load_with_splits", "get_features"):
        assert hasattr(loader, method_name), f"Loader missing {method_name}"


def test_parquet_loader_contract_schema_types_and_time_ordering() -> None:
    loader = _build_parquet_loader()

    train_df = loader.load(FIXTURE_PATH, mode="train")
    eval_df = loader.load(FIXTURE_PATH, mode="eval")
    test_df = loader.load(FIXTURE_PATH, mode="test")

    _assert_frame_contract(train_df)
    _assert_frame_contract(eval_df)
    _assert_frame_contract(test_df)

    # Chronological split integrity: no train/eval/test time leakage.
    assert train_df["timestamp"].max() <= eval_df["timestamp"].min()
    assert eval_df["timestamp"].max() <= test_df["timestamp"].min()


def test_parquet_loader_load_with_splits_matches_mode_loads() -> None:
    loader = _build_parquet_loader()

    train_by_mode = loader.load(FIXTURE_PATH, mode="train")
    eval_by_mode = loader.load(FIXTURE_PATH, mode="eval")
    test_by_mode = loader.load(FIXTURE_PATH, mode="test")

    train_df, eval_df, test_df = loader.load_with_splits(FIXTURE_PATH)

    assert len(train_df) == len(train_by_mode)
    assert len(eval_df) == len(eval_by_mode)
    assert len(test_df) == len(test_by_mode)
    assert train_df["timestamp"].equals(train_by_mode["timestamp"])
    assert eval_df["timestamp"].equals(eval_by_mode["timestamp"])
    assert test_df["timestamp"].equals(test_by_mode["timestamp"])


def test_entrypoints_and_use_cases_do_not_import_concrete_data_loaders() -> None:
    targets = [
        ROOT / "experiments" / "train.py",
        ROOT / "run_pipeline.py",
        ROOT / "src" / "rl_trading_lab" / "runtime" / "training_entrypoint.py",
        ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "train_agent.py",
    ]
    forbidden_modules = {
        "rl_trading_lab.application.ports.data_loader",
        "rl_trading_lab.infrastructure.adapters.csv_data_loader",
        "rl_trading_lab.data.binance_adapter",
    }
    forbidden_names = {"ParquetDataLoader", "CsvDataLoader", "BinanceDataAdapter"}
    violations: list[str] = []

    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in forbidden_modules:
                    for alias in node.names:
                        if alias.name in forbidden_names:
                            violations.append(
                                f"{path}:{node.lineno} imports {alias.name} from {node.module}"
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append(f"{path}:{node.lineno} imports module {alias.name}")

    assert not violations, (
        "Entry points/use cases must resolve loaders through DataLoaderPort + factory:\n"
        + "\n".join(violations)
    )
