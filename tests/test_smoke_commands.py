"""Layer 4 smoke checks for canonical runtime surfaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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


def test_live_entrypoint_help_smoke() -> None:
    result = _run([PYTHON, "experiments/live_trading.py", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Run live trading system." in result.stdout


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
