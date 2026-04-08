"""Documentation contract checks for DAL-128 command unification."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

COMMANDS_DOC = ROOT / "docs" / "commands.md"
DOCS_REQUIRING_MATRIX_REFERENCE = [
    ROOT / "README.md",
    ROOT / "docs" / "runtime_path.md",
    ROOT / "docs" / "production_boundary.md",
    ROOT / "docs" / "configuration.md",
    ROOT / "docs" / "training.md",
    ROOT / "docs" / "execution.md",
    ROOT / "docs" / "upwork_demo_execution_plan.md",
]

REQUIRED_MATRIX_ROWS = [
    "Smoke test (clean install)",
    "Local training",
    "Evaluation (within canonical training flow)",
    "Pre-live validation surface",
    "Paper trading surface",
    "MLflow UI",
    "Notebook inspection",
]


def test_commands_doc_exists_and_has_required_rows() -> None:
    text = COMMANDS_DOC.read_text(encoding="utf-8")
    for row in REQUIRED_MATRIX_ROWS:
        assert row in text, f"Missing required command matrix row: {row}"


def test_core_docs_reference_commands_matrix() -> None:
    missing: list[str] = []
    for path in DOCS_REQUIRING_MATRIX_REFERENCE:
        text = path.read_text(encoding="utf-8")
        if "commands.md" not in text:
            missing.append(str(path.relative_to(ROOT)))
    assert not missing, "Docs missing commands matrix reference:\n" + "\n".join(missing)


def test_legacy_cli_key_variants_removed_from_authoritative_docs() -> None:
    legacy_markers = ("trainer.max_steps", "env.dataset=sample")
    violations: list[str] = []
    for path in DOCS_REQUIRING_MATRIX_REFERENCE + [COMMANDS_DOC]:
        text = path.read_text(encoding="utf-8")
        for marker in legacy_markers:
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)}: {marker}")
    assert not violations, "Legacy CLI/config markers found:\n" + "\n".join(violations)
