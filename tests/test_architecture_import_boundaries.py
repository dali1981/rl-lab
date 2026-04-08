"""Repo-level architecture import boundary enforcement for DAL-134."""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]

SCAN_PATHS = [
    ROOT / "src" / "rl_trading_lab" / "application",
    ROOT / "src" / "rl_trading_lab" / "agents",
    ROOT / "experiments",
]


def _iter_files() -> list[Path]:
    files: list[Path] = []
    for base in SCAN_PATHS:
        files.extend(sorted(base.rglob("*.py")))
    return files


def _forbidden(module: str) -> bool:
    normalized = module.lstrip(".")

    # Explicitly forbidden dependency surfaces in scanned layers.
    if normalized.startswith("rl_trading_lab.infrastructure"):
        return True
    if normalized.startswith("rl_trading_lab.live"):
        return True

    # Relative import equivalents inside package trees.
    if normalized.startswith("infrastructure"):
        return True
    if normalized.startswith("live"):
        return True

    # Concrete adapter/data/environment imports are forbidden outside runtime.
    concrete_prefixes = (
        "rl_trading_lab.data.",
        "rl_trading_lab.environment.",
        "src.rl_trading_lab.infrastructure.",
        "src.rl_trading_lab.live.",
        "src.rl_trading_lab.data.",
        "src.rl_trading_lab.environment.",
        "data.",
        "environment.",
    )
    if normalized.startswith(concrete_prefixes):
        return True
    if normalized in {"data", "environment"}:
        return True

    if ".adapters." in normalized or normalized.endswith(".adapters"):
        return True

    return False


def _imports(file_path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    out: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno))

    return out


def _collect_violations(files: list[Path]) -> list[str]:
    violations: list[str] = []

    for file_path in files:
        if file_path.name == "__init__.py":
            continue
        for module, lineno in _imports(file_path):
            if _forbidden(module):
                rel = file_path.relative_to(ROOT)
                violations.append(f"{rel}: {module} (line {lineno})")

    return violations


def test_architecture_import_boundaries() -> None:
    violations = _collect_violations(_iter_files())

    assert not violations, "Architecture import boundary violations:\n" + "\n".join(violations)


def test_architecture_import_boundaries_catches_reintroduced_violation() -> None:
    fixture = ROOT / "tests" / "_boundary_violation_fixture.py"
    fixture.write_text(
        dedent(
            """
            from rl_trading_lab.infrastructure.adapters.market_data_adapter import ParquetMarketDataAdapter
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    try:
        violations = _collect_violations([fixture])
        assert violations, "Boundary enforcement failed to detect a forbidden infrastructure import"
    finally:
        fixture.unlink(missing_ok=True)
