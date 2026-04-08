"""DAL-135 trainer architecture consolidation checks."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SB3_FACADE = ROOT / "src" / "rl_trading_lab" / "agents" / "sb3_agents.py"
TRAINER_MODULE = ROOT / "src" / "rl_trading_lab" / "agents" / "trainer.py"
TRAINER_FACTORY = ROOT / "src" / "rl_trading_lab" / "agents" / "trainer_factory.py"


def _ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_sb3_agents_is_facade_only() -> None:
    tree = _ast(SB3_FACADE)
    class_defs = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert not class_defs, "sb3_agents.py must be facade-only (no class definitions)"

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


def test_trainer_factory_uses_authoritative_trainer_module() -> None:
    text = TRAINER_FACTORY.read_text(encoding="utf-8")
    assert "from rl_trading_lab.agents.trainer import Trainer" in text
    assert "from rl_trading_lab.agents.sb3_agents import Trainer" not in text


def test_authoritative_trainer_module_defines_trainer() -> None:
    tree = _ast(TRAINER_MODULE)
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "Trainer" in class_names, "trainer.py must define Trainer"
