"""Layer 3 feature-engineering port contract and boundary checks."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from rl_trading_lab.application.ports.feature_engineering import FeatureEngineeringPort
from rl_trading_lab.infrastructure.factories import create_feature_engineering


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "sample_data" / "btcusdt_sample_10k.parquet"
EXPECTED_ENGINEERED_FEATURES = [
    "ratio_sma_5_close_zscore",
    "ratio_sma_20_close_zscore",
    "ratio_range_close_zscore",
    "fracdiff_0.4_zscore",
]


def test_feature_factory_resolves_port_contract() -> None:
    feature_pipeline = create_feature_engineering(pipeline_type="crypto")
    assert isinstance(feature_pipeline, FeatureEngineeringPort)
    feature_names = feature_pipeline.feature_names
    assert isinstance(feature_names, list)
    assert feature_names


def test_feature_pipeline_contract_on_fixture_columns_order_types_and_nan() -> None:
    feature_pipeline = create_feature_engineering(pipeline_type="crypto")
    source_df = pd.read_parquet(FIXTURE_PATH)

    transformed_df = feature_pipeline.transform(source_df)

    assert not transformed_df.empty
    assert pd.api.types.is_datetime64_any_dtype(transformed_df["timestamp"])

    for column in EXPECTED_ENGINEERED_FEATURES:
        assert column in transformed_df.columns
        assert pd.api.types.is_numeric_dtype(transformed_df[column]), (
            f"Engineered feature must be numeric: {column}"
        )
    for column in feature_pipeline.feature_names:
        assert column in transformed_df.columns

    observed_positions = [transformed_df.columns.get_loc(col) for col in EXPECTED_ENGINEERED_FEATURES]
    assert observed_positions == sorted(observed_positions), (
        "Expected feature column order drifted in transformed output"
    )

    non_nullable = ["timestamp", "open", "high", "low", "close", "volume", *EXPECTED_ENGINEERED_FEATURES]
    assert not transformed_df[non_nullable].isna().any().any()


def test_passthrough_pipeline_enforces_feature_column_contract() -> None:
    passthrough = create_feature_engineering(
        pipeline_type="passthrough",
        feature_names=["close", "volume"],
    )
    fixture_df = pd.read_parquet(FIXTURE_PATH)
    transformed_df = passthrough.transform(fixture_df)
    assert list(transformed_df.columns) == list(fixture_df.columns)


def test_canonical_runtime_must_resolve_feature_pipeline_through_factory() -> None:
    training_entrypoint = ROOT / "src" / "rl_trading_lab" / "runtime" / "training_entrypoint.py"
    tree = ast.parse(training_entrypoint.read_text(encoding="utf-8"))

    has_factory_call = any(
        isinstance(node, ast.Call)
        and (
            (
                isinstance(node.func, ast.Name)
                and node.func.id == "create_feature_engineering"
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_feature_engineering"
            )
        )
        for node in ast.walk(tree)
    )
    assert has_factory_call, (
        "Canonical runtime must resolve feature engineering via "
        "create_feature_engineering(...) in training_entrypoint.py"
    )


def test_no_concrete_feature_pipeline_imports_outside_factory_boundary() -> None:
    scan_roots = [
        ROOT / "src" / "rl_trading_lab" / "application",
        ROOT / "src" / "rl_trading_lab" / "domain",
        ROOT / "experiments",
    ]
    direct_targets = [ROOT / "run_pipeline.py"]
    forbidden_modules = {"rl_trading_lab.data.feature_pipeline", "rl_trading_lab.data"}
    forbidden_names = {"FeaturePipeline"}
    violations: list[str] = []

    targets: list[Path] = []
    for root in scan_roots:
        if root.exists():
            targets.extend(sorted(root.rglob("*.py")))
    targets.extend(path for path in direct_targets if path.exists())

    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in forbidden_modules:
                    for alias in node.names:
                        if alias.name in forbidden_names:
                            violations.append(
                                f"{path}:{node.lineno} imports {alias.name} from {module}"
                            )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_modules:
                        violations.append(f"{path}:{node.lineno} imports module {alias.name}")

    assert not violations, (
        "Concrete feature pipeline imports are forbidden outside factory boundary:\n"
        + "\n".join(violations)
    )
