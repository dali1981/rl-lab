"""Architecture enforcement for application-layer import boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPLICATION_FILES = sorted((ROOT / "src" / "rl_trading_lab" / "application").rglob("*.py"))


def _is_forbidden(module: str) -> bool:
    return (
        module.startswith("rl_trading_lab.infrastructure")
        or module.startswith("rl_trading_lab.live")
        or module.startswith("rl_trading_lab.data.binance_adapter")
    )


def _forbidden_imports(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and _is_forbidden(node.module):
            hits.append(f"from {node.module} import ... (line {node.lineno})")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    hits.append(f"import {alias.name} (line {node.lineno})")

    return hits


def test_application_layer_avoids_concrete_infrastructure_imports() -> None:
    violations: list[str] = []

    for file_path in APPLICATION_FILES:
        if file_path.name == "__init__.py":
            continue
        for hit in _forbidden_imports(file_path):
            violations.append(f"{file_path.relative_to(ROOT)}: {hit}")

    assert not violations, "Application-layer import violations:\n" + "\n".join(violations)
