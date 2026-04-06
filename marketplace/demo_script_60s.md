# 60-Second Demo Script

Goal:
- Show config -> run -> MLflow metrics, emphasize reproducibility and research scope.

Time 0-10s (Intro):
- On-screen: repo root, README open
- Voiceover: "This is RL Trading Lab, a research-grade reinforcement learning workflow for trading experiments."

Time 10-25s (Config):
- On-screen: open configs/ and show env and agent configs
- Voiceover: "Experiments are fully config-driven via Hydra. Reward functions and constraints are explicit and reproducible."

Time 25-40s (Run):
- On-screen: terminal run command
  - uv run python experiments/train.py trainer.max_steps=1000 env.dataset=sample
- Voiceover: "I run a short training loop to validate the pipeline and logging."

Time 40-55s (MLflow):
- On-screen: MLflow UI with metrics
- Voiceover: "Metrics and artifacts are logged to MLflow for traceable experiments and comparisons."

Time 55-60s (Close):
- On-screen: summary slide
- Voiceover: "Research-grade experiments only. No profit guarantees."
