"""DAL-143 checks for live module isolation as optional integration surface."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LIVE_DOC = ROOT / "docs" / "live_trading.md"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_live_dependencies_are_optional_extra_only() -> None:
    data = _load_pyproject()
    project = data["project"]
    optional = project["optional-dependencies"]
    base_dependencies = project["dependencies"]

    assert "live" in optional, "Expected [project.optional-dependencies].live in pyproject.toml"
    assert any(dep.startswith("python-binance") for dep in optional["live"])
    assert not any(dep.startswith("python-binance") for dep in base_dependencies), (
        "python-binance must be isolated to the live extra"
    )


def test_core_training_eval_paths_do_not_import_live_modules() -> None:
    targets = [
        ROOT / "experiments" / "train.py",
        ROOT / "run_pipeline.py",
        ROOT / "src" / "rl_trading_lab" / "runtime" / "training_entrypoint.py",
        ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "train_agent.py",
        ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "evaluate_agent.py",
    ]
    violations: list[str] = []

    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("rl_trading_lab.live"):
                    violations.append(f"{path}:{node.lineno} imports from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("rl_trading_lab.live"):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")

    assert not violations, "Core training/eval paths must not import live modules:\n" + "\n".join(violations)


def test_live_trading_doc_declares_isolation_and_prerequisites() -> None:
    assert LIVE_DOC.exists(), "docs/live_trading.md must exist"
    text = LIVE_DOC.read_text(encoding="utf-8")
    required_snippets = [
        "optional bounded integration zone",
        "not part of the canonical offline training/evaluation path",
        "uv sync --extra live",
        "prerequisites",
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    assert not missing, f"docs/live_trading.md missing required isolation/prereq content: {missing}"
