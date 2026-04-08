"""Enforce entrypoint boundary: no direct domain/infrastructure imports."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = [
    ROOT / "run_pipeline.py",
    *sorted((ROOT / "experiments").glob("*.py")),
    *sorted((ROOT / "examples").glob("*.py")),
]


def _forbidden_imports(file_path: Path) -> list[str]:
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("rl_trading_lab.domain") or node.module.startswith(
                "rl_trading_lab.infrastructure"
            ):
                hits.append(f"from {node.module} import ... (line {node.lineno})")

        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("rl_trading_lab.domain") or alias.name.startswith(
                    "rl_trading_lab.infrastructure"
                ):
                    hits.append(f"import {alias.name} (line {node.lineno})")

    return hits


def test_entrypoints_do_not_import_domain_or_infrastructure_directly() -> None:
    violations: list[str] = []
    for file_path in ENTRYPOINTS:
        hits = _forbidden_imports(file_path)
        violations.extend([f"{file_path.relative_to(ROOT)}: {hit}" for hit in hits])

    assert not violations, "Direct domain/infrastructure imports found:\n" + "\n".join(violations)
