"""Lightweight smoke and integration tests for SB3 training and RL-vs-GJM benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from space_manipulator.rl.benchmark import (
    parse_args as benchmark_parse_args,
    run_benchmark,
)
from space_manipulator.rl.train import (
    TrainConfig,
    parse_args as train_parse_args,
    train_agent,
)


def test_ppo_smoke_train_and_paired_benchmark(tmp_path: Path) -> None:
    """Run a tiny PPO smoke training with VecNormalize, then benchmark vs. GJM."""
    run_dir = tmp_path / "ppo_smoke"
    cfg = TrainConfig(
        algo="ppo",
        mission="reach",
        action_mode="residual_gjm",
        total_timesteps=32,
        n_envs=1,
        seed=7,
        max_episode_steps=10,
        eval_freq=16,
        eval_episodes=1,
        normalize_obs=True,
        device="cpu",
        output_dir=run_dir,
        n_steps=16,
        batch_size=16,
        n_epochs=1,
    )
    train_out = train_agent(cfg)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()
    assert train_out["final_model_path"] == run_dir / "final_model.zip"
    assert train_out["vecnormalize_path"] == run_dir / "vecnormalize.pkl"

    saved_cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert saved_cfg["algo"] == "ppo"
    assert saved_cfg["normalize_obs"] is True
    assert saved_cfg["env_config"]["mission"] == "reach"

    bench_dir = tmp_path / "bench_out"
    bench_out = run_benchmark(
        model_path=run_dir / "final_model.zip",
        n_episodes=2,
        seed=100,
        max_episode_steps=20,
        device="cpu",
        output_dir=bench_dir,
    )

    summary_path = bench_dir / "benchmark_summary.json"
    csv_path = bench_dir / "benchmark_episodes.csv"
    plot_path = bench_dir / "benchmark_comparison.png"
    assert summary_path.exists()
    assert csv_path.exists()
    assert plot_path.exists()
    assert bench_out["summary_json_path"] == summary_path

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "GJM Classic" in summary["methods"]
    assert "RL (PPO)" in summary["methods"]
    for method in ("GJM Classic", "RL (PPO)"):
        m_summary = summary["methods"][method]["summary"]
        assert "success_rate" in m_summary
        assert "final_ee_error_m" in m_summary
        assert "max_base_attitude_deg" in m_summary
        assert "cumulative_fuel_impulse" in m_summary

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4  # 2 episodes * 2 methods
    assert {r["method"] for r in rows} == {"GJM Classic", "RL (PPO)"}

    if (run_dir / "evaluations.npz").exists():
        assert (bench_dir / "learning_curve.png").exists()


def test_sac_smoke_train_save_and_load(tmp_path: Path) -> None:
    """Run a tiny SAC smoke training and verify model saving and loading."""
    run_dir = tmp_path / "sac_smoke"
    cfg = TrainConfig(
        algo="sac",
        mission="reach",
        action_mode="torque",
        total_timesteps=12,
        n_envs=1,
        seed=11,
        max_episode_steps=10,
        eval_freq=12,
        eval_episodes=1,
        normalize_obs=False,
        device="cpu",
        output_dir=run_dir,
        learning_starts=4,
        batch_size=4,
    )
    train_agent(cfg)

    assert (run_dir / "config.json").exists()
    assert (run_dir / "final_model.zip").exists()

    bench_dir = tmp_path / "sac_bench"
    run_benchmark(
        model_path=run_dir / "final_model.zip",
        n_episodes=1,
        seed=21,
        max_episode_steps=10,
        device="cpu",
        output_dir=bench_dir,
    )
    summary = json.loads((bench_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert "RL (SAC)" in summary["methods"]


def test_benchmark_baseline_only(tmp_path: Path) -> None:
    """Verify baseline_only=True runs GJM Classic without any RL model."""
    bench_dir = tmp_path / "baseline_only"
    run_benchmark(
        baseline_only=True,
        mission="fuel_optimal",
        n_episodes=2,
        seed=5,
        max_episode_steps=15,
        output_dir=bench_dir,
    )

    assert (bench_dir / "benchmark_summary.json").exists()
    assert (bench_dir / "benchmark_episodes.csv").exists()
    assert (bench_dir / "benchmark_comparison.png").exists()

    summary = json.loads((bench_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert list(summary["methods"].keys()) == ["GJM Classic"]


def test_cli_parse_args() -> None:
    """Verify CLI argument parsing for both train.py and benchmark.py."""
    t_args = train_parse_args(
        [
            "--algo",
            "sac",
            "--mission",
            "low_disturbance",
            "--action-mode",
            "residual_gjm",
            "--total-timesteps",
            "50000",
            "--n-envs",
            "2",
            "--normalize-obs",
            "--device",
            "cpu",
        ]
    )
    assert t_args.algo == "sac"
    assert t_args.mission == "low_disturbance"
    assert t_args.action_mode == "residual_gjm"
    assert t_args.total_timesteps == 50000
    assert t_args.n_envs == 2
    assert t_args.normalize_obs is True
    assert t_args.device == "cpu"

    b_args = benchmark_parse_args(
        [
            "--baseline-only",
            "--mission",
            "moving_target",
            "--n-episodes",
            "5",
            "--seed",
            "99",
            "--output-dir",
            "runs/test_bench",
        ]
    )
    assert b_args.baseline_only is True
    assert b_args.mission == "moving_target"
    assert b_args.n_episodes == 5
    assert b_args.seed == 99
    assert b_args.output_dir == "runs/test_bench"
