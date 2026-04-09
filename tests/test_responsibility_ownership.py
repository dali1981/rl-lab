"""DAL-144 ownership enforcement for canonical responsibility boundaries."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
        elif isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
    return out


def test_reward_policy_choice_routes_through_domain_factory() -> None:
    path = ROOT / "src" / "rl_trading_lab" / "application" / "services" / "environment_service.py"
    text = path.read_text(encoding="utf-8")
    assert "create_reward_service(" in text
    assert "ReturnsRewardCalculation(" not in text
    assert "PnLRewardCalculation(" not in text


def test_checkpoint_lifecycle_owned_by_checkpoint_service_in_canonical_flow() -> None:
    runtime = ROOT / "src" / "rl_trading_lab" / "runtime" / "training_entrypoint.py"
    use_case = ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "train_agent.py"
    runtime_text = runtime.read_text(encoding="utf-8")
    use_case_text = use_case.read_text(encoding="utf-8")

    assert "CheckpointService(" in runtime_text
    assert "create_training_callbacks(" in use_case_text
    assert "save_final_model(" in use_case_text

    forbidden = [
        "from rl_trading_lab.utils.checkpoint_manager import",
        "from rl_trading_lab.agents.callback_factory import",
        "from rl_trading_lab.agents.trainer import",
    ]
    for marker in forbidden:
        assert marker not in runtime_text
        assert marker not in use_case_text


def test_core_modules_do_not_depend_on_legacy_environment_factory_or_data_processor() -> None:
    targets = [
        ROOT / "src" / "rl_trading_lab" / "application" / "services" / "environment_service.py",
        ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "train_agent.py",
        ROOT / "src" / "rl_trading_lab" / "application" / "use_cases" / "evaluate_agent.py",
        ROOT / "src" / "rl_trading_lab" / "runtime" / "training_entrypoint.py",
        ROOT / "experiments" / "train.py",
    ]
    forbidden_prefixes = (
        "rl_trading_lab.environment.factory",
        "rl_trading_lab.utils.data_processor",
    )
    violations: list[str] = []

    for path in targets:
        for module in _imports(path):
            if module.startswith(forbidden_prefixes):
                violations.append(f"{path}: imports forbidden module {module}")

    assert not violations, "Canonical modules depend on legacy split/assembly utilities:\n" + "\n".join(violations)


def test_dal144_ownership_table_present_in_architecture_rules() -> None:
    doc = ROOT / "docs" / "architecture_rules.md"
    text = doc.read_text(encoding="utf-8")
    required = [
        "Responsibility Ownership Table (DAL-144)",
        "Environment assembly (domain + adapter wiring)",
        "Reward/risk policy choice",
        "Checkpoint lifecycle (save/best/final callbacks)",
        "Data split logic (train/eval/test boundaries)",
        "Evaluation metrics ownership",
    ]
    missing = [token for token in required if token not in text]
    assert not missing, f"architecture_rules.md missing DAL-144 ownership tokens: {missing}"
