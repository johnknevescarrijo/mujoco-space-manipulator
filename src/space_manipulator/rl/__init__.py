"""Stable-Baselines3 training and RL-vs-GJM benchmark utilities."""

from __future__ import annotations

from space_manipulator.rl.benchmark import (
    evaluate_policy,
    plot_benchmark_comparison,
    plot_learning_curve,
    run_benchmark,
)
from space_manipulator.rl.train import TrainConfig, build_vec_env, train_agent

__all__ = [
    "TrainConfig",
    "build_vec_env",
    "evaluate_policy",
    "plot_benchmark_comparison",
    "plot_learning_curve",
    "run_benchmark",
    "train_agent",
]
