"""Layer 4 smoke checks for canonical runtime surfaces."""

from __future__ import annotations

import os
import subprocess
import sys
import ast
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "default::DeprecationWarning"
    return subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_train_help_smoke() -> None:
    result = _run([PYTHON, "experiments/train.py", "--help"])
    assert result.returncode == 0, result.stderr
    assert "train is powered by Hydra" in result.stdout


def test_run_pipeline_help_smoke() -> None:
    result = _run([PYTHON, "run_pipeline.py", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Running canonical training entrypoint..." in result.stdout


def test_live_trading_example_help_smoke() -> None:
    result = _run([PYTHON, "examples/live_trading_example.py", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Commands" in result.stdout


def test_live_trading_example_validate_help_smoke() -> None:
    result = _run([PYTHON, "examples/live_trading_example.py", "validate", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Validate trading pipeline" in result.stdout


def test_live_trading_example_trade_help_smoke() -> None:
    result = _run([PYTHON, "examples/live_trading_example.py", "trade", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Run live trading" in result.stdout


def test_live_trading_example_analyze_help_smoke() -> None:
    result = _run([PYTHON, "examples/live_trading_example.py", "analyze", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage: live_trading_example.py analyze" in result.stdout


def test_live_entrypoint_help_smoke() -> None:
    result = _run([PYTHON, "experiments/live_trading.py", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Run live trading system." in result.stdout


def test_mlflow_help_smoke() -> None:
    result = _run([PYTHON, "-m", "mlflow", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Usage: python -m mlflow" in result.stdout


def test_jupyter_help_smoke() -> None:
    result = _run([PYTHON, "-m", "jupyter", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Jupyter: Interactive Computing" in result.stdout


def test_short_canonical_training_smoke(tmp_path: Path) -> None:
    save_path = tmp_path / "checkpoints"
    hydra_dir = tmp_path / "hydra"

    cmd = [
        PYTHON,
        "experiments/train.py",
        "data.train_data_path=sample_data/btcusdt_sample_10k.parquet",
        "training.total_timesteps=16",
        "training.eval_freq=8",
        "training.save_freq=16",
        "training.n_eval_episodes=1",
        f"training.save_path={save_path}",
        "logging.mlflow.enabled=false",
        "logging.tensorboard.enabled=false",
        "logging.console.progress_bar=false",
        f"hydra.run.dir={hydra_dir}",
    ]
    result = _run(cmd)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert (save_path / "final_model" / "model.zip").exists()


def test_smoke_suite_has_no_sb3_agents_import() -> None:
    source = (ROOT / "tests" / "test_smoke_commands.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "rl_trading_lab.agents.sb3_agents":
            forbidden.append(f"from {node.module} import ... (line {node.lineno})")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "rl_trading_lab.agents.sb3_agents":
                    forbidden.append(f"import {alias.name} (line {node.lineno})")
    assert not forbidden, "Smoke tests must not import sb3_agents:\n" + "\n".join(forbidden)


def test_canonical_training_500_step_smoke_with_mlflow(tmp_path: Path) -> None:
    save_path = tmp_path / "checkpoints"
    hydra_dir = tmp_path / "hydra"
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_uri()
    experiment_name = "dal138-canonical-smoke"

    cmd = [
        PYTHON,
        "experiments/train.py",
        "data.train_data_path=sample_data/btcusdt_sample_10k.parquet",
        "training.total_timesteps=500",
        "training.eval_freq=100",
        "training.save_freq=500",
        "training.n_eval_episodes=1",
        f"training.save_path={save_path}",
        "logging.mlflow.enabled=true",
        f"logging.mlflow.tracking_uri={tracking_uri}",
        f"logging.mlflow.experiment_name={experiment_name}",
        "logging.tensorboard.enabled=false",
        "logging.console.progress_bar=false",
        f"hydra.run.dir={hydra_dir}",
    ]
    result = _run(cmd)
    combined = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, combined
    assert "warning" not in combined.lower(), combined
    assert (save_path / "final_model" / "model.zip").exists()

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None, "MLflow experiment was not created"

    client = MlflowClient(tracking_uri=tracking_uri)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    assert runs, "MLflow run was not created"
    latest = runs[0]

    assert latest.data.params, "Expected MLflow params to be logged"
    assert latest.data.metrics, "Expected MLflow metrics to be logged"
