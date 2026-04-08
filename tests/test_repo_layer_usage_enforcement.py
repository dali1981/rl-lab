"""DAL-134 enforcement across application, agents, and experiments surfaces."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_FILES = sorted((ROOT / "src" / "rl_trading_lab" / "application").rglob("*.py"))
AGENT_FILES = sorted((ROOT / "src" / "rl_trading_lab" / "agents").rglob("*.py"))
EXPERIMENT_FILES = sorted((ROOT / "experiments").glob("*.py"))
RUN_PIPELINE = ROOT / "run_pipeline.py"

DEPRECATED_EXPERIMENT_ALLOWLIST = {
    "experiments/live_trading.py",
    "experiments/validate_data_pipeline.py",
    "experiments/validate_live.py",
    "experiments/test_one_trade_mode.py",
    "experiments/test_transformer.py",
}


def _imported_modules(file_path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    modules: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append((node.module, node.lineno))
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append((alias.name, node.lineno))

    return modules


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module.startswith(prefix) for prefix in prefixes)


def test_application_layer_has_no_concrete_infra_live_imports() -> None:
    forbidden_prefixes = (
        "rl_trading_lab.infrastructure",
        "rl_trading_lab.live",
        "rl_trading_lab.data.binance_adapter",
    )
    violations: list[str] = []

    for file_path in APPLICATION_FILES:
        if file_path.name == "__init__.py":
            continue
        for module, lineno in _imported_modules(file_path):
            if _starts_with_any(module, forbidden_prefixes):
                rel = file_path.relative_to(ROOT)
                violations.append(f"{rel}: {module} (line {lineno})")

    assert not violations, "Application-layer import violations:\n" + "\n".join(violations)


def test_agents_layer_has_no_direct_infra_live_data_imports() -> None:
    forbidden_prefixes = (
        "rl_trading_lab.infrastructure",
        "rl_trading_lab.live",
        "rl_trading_lab.data",
        "src.rl_trading_lab.infrastructure",
        "src.rl_trading_lab.live",
        "src.rl_trading_lab.data",
    )
    violations: list[str] = []

    for file_path in AGENT_FILES:
        if file_path.name == "__init__.py":
            continue
        for module, lineno in _imported_modules(file_path):
            if _starts_with_any(module, forbidden_prefixes):
                rel = file_path.relative_to(ROOT)
                violations.append(f"{rel}: {module} (line {lineno})")

    assert not violations, "Agent-layer import violations:\n" + "\n".join(violations)


def test_experiments_bypass_imports_are_allowlisted_and_deprecated() -> None:
    bypass_prefixes = (
        "rl_trading_lab.infrastructure",
        "rl_trading_lab.live",
        "rl_trading_lab.data",
        "rl_trading_lab.environment",
        "src.rl_trading_lab.environment",
    )
    violations: list[str] = []

    for file_path in EXPERIMENT_FILES:
        if file_path.name == "__init__.py":
            continue

        rel = str(file_path.relative_to(ROOT))
        bypass_imports = [
            f"{module} (line {lineno})"
            for module, lineno in _imported_modules(file_path)
            if _starts_with_any(module, bypass_prefixes)
        ]

        if not bypass_imports:
            continue

        if rel not in DEPRECATED_EXPERIMENT_ALLOWLIST:
            joined = ", ".join(bypass_imports)
            violations.append(
                f"{rel}: unexpected bypass imports outside allowlist: {joined}"
            )
            continue

        text = file_path.read_text(encoding="utf-8")
        if "DeprecationWarning" not in text:
            violations.append(
                f"{rel}: allowlisted bypass entrypoint must emit DeprecationWarning"
            )

    assert not violations, "Experiment bypass enforcement violations:\n" + "\n".join(violations)


def test_run_pipeline_has_no_direct_infra_live_data_imports() -> None:
    forbidden_prefixes = (
        "rl_trading_lab.infrastructure",
        "rl_trading_lab.live",
        "rl_trading_lab.data",
    )
    violations = [
        f"{module} (line {lineno})"
        for module, lineno in _imported_modules(RUN_PIPELINE)
        if _starts_with_any(module, forbidden_prefixes)
    ]
    assert not violations, "run_pipeline import violations:\n" + "\n".join(violations)
