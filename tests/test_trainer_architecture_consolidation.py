"""DAL-135 trainer architecture consolidation checks."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SB3_FACADE = ROOT / "src" / "rl_trading_lab" / "agents" / "sb3_agents.py"
TRAINER_MODULE = ROOT / "src" / "rl_trading_lab" / "agents" / "trainer.py"
TRAINER_FACTORY = ROOT / "src" / "rl_trading_lab" / "agents" / "trainer_factory.py"
SRC_ROOT = ROOT / "src"
EXPERIMENTS_ROOT = ROOT / "experiments"
RUN_PIPELINE = ROOT / "run_pipeline.py"


def _ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_sb3_agents_is_facade_only() -> None:
    tree = _ast(SB3_FACADE)
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    class_names = {node.name for node in class_defs}
    assert class_names <= {"Trainer"}, (
        "sb3_agents.py facade may only define compatibility Trainer shim"
    )

    direct_framework_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("stable_baselines3"):
            direct_framework_imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("stable_baselines3"):
                    direct_framework_imports.append(alias.name)
    assert not direct_framework_imports, (
        "sb3_agents.py must not directly import SB3/framework internals"
    )


def test_sb3_agents_facade_targets_trainer_module() -> None:
    text = SB3_FACADE.read_text(encoding="utf-8")
    assert "from rl_trading_lab.agents.trainer import" in text


def test_sb3_agents_uses_deprecation_warning() -> None:
    text = SB3_FACADE.read_text(encoding="utf-8")
    assert "DeprecationWarning" in text
    assert "warnings.warn(" in text


def test_trainer_factory_uses_authoritative_trainer_module() -> None:
    text = TRAINER_FACTORY.read_text(encoding="utf-8")
    assert "from rl_trading_lab.agents.trainer import Trainer" in text
    assert "from rl_trading_lab.agents.sb3_agents import Trainer" not in text


def test_authoritative_trainer_module_defines_trainer() -> None:
    tree = _ast(TRAINER_MODULE)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "Trainer" in class_names, "trainer.py must define Trainer"


def test_no_production_imports_from_sb3_agents_trainer() -> None:
    violations: list[str] = []
    scan_files = list(SRC_ROOT.rglob("*.py")) + list(EXPERIMENTS_ROOT.rglob("*.py")) + [RUN_PIPELINE]
    for path in scan_files:
        text = path.read_text(encoding="utf-8")
        if "from rl_trading_lab.agents.sb3_agents import Trainer" in text:
            rel = path.relative_to(ROOT)
            violations.append(str(rel))
    assert not violations, (
        "Production paths must not import Trainer from sb3_agents facade:\n"
        + "\n".join(violations)
    )
