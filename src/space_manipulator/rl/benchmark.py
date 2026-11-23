"""Paired RL vs. classical GJM evaluation, metrics, CSV/JSON export, and plotting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from space_manipulator.envs import (
    VALID_ACTION_MODES,
    VALID_MISSION_MODES,
    ActionMode,
    EnvConfig,
    MissionMode,
    SpaceManipulatorEnv,
)
from space_manipulator.rl.train import VALID_ALGOS

METRIC_KEYS: tuple[str, ...] = (
    "success_rate",
    "final_ee_error_m",
    "mean_ee_error_m",
    "max_base_attitude_deg",
    "final_base_attitude_deg",
    "cumulative_fuel_impulse",
    "mean_arm_effort",
    "episode_return",
    "episode_length",
)


def _aggregate_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, std, and median for a list of scalar values."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def _pad_and_aggregate_series(
    series_list: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Pad variable-length episode series with their final value and return (mean, std)."""
    max_len = max(len(s) for s in series_list)
    mat = np.empty((len(series_list), max_len), dtype=np.float64)
    for idx, seq in enumerate(series_list):
        n = len(seq)
        mat[idx, :n] = seq
        if n < max_len:
            mat[idx, n:] = seq[-1]
    return np.mean(mat, axis=0).tolist(), np.std(mat, axis=0).tolist()


def evaluate_policy(
    env: SpaceManipulatorEnv,
    policy_fn: Callable[[NDArray[np.float32]], NDArray[np.float32]],
    *,
    n_episodes: int = 10,
    seed: int = 42,
    method_name: str = "policy",
    obs_normalizer: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None,
) -> dict[str, Any]:
    """Evaluate a policy deterministically across ``n_episodes`` seeded rollouts."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1.")

    episodes: list[dict[str, Any]] = []
    all_ee_errors: list[list[float]] = []
    all_base_atts: list[list[float]] = []

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        obs, info = env.reset(seed=ep_seed)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_ee_errors: list[float] = []
        ep_base_atts: list[float] = []
        ep_arm_efforts: list[float] = []

        while not done:
            model_obs = obs_normalizer(obs) if obs_normalizer is not None else obs
            action = np.asarray(policy_fn(model_obs), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_len += 1
            ep_ee_errors.append(float(info["ee_error"]))
            ep_base_atts.append(float(info["base_attitude_deg"]))
            ep_arm_efforts.append(float(info["arm_effort"]))
            done = bool(terminated or truncated)

        all_ee_errors.append(ep_ee_errors)
        all_base_atts.append(ep_base_atts)

        episodes.append(
            {
                "episode_idx": ep_idx,
                "seed": ep_seed,
                "success": bool(info["is_success"]),
                "final_ee_error_m": float(info["ee_error"]),
                "mean_ee_error_m": float(np.mean(ep_ee_errors)),
                "max_base_attitude_deg": float(np.max(ep_base_atts)),
                "final_base_attitude_deg": float(info["base_attitude_deg"]),
                "cumulative_fuel_impulse": float(info["cumulative_fuel_impulse"]),
                "mean_arm_effort": float(np.mean(ep_arm_efforts)),
                "episode_return": float(ep_return),
                "episode_length": int(ep_len),
            }
        )

    summary: dict[str, dict[str, float]] = {
        "success_rate": _aggregate_stats(
            [1.0 if ep["success"] else 0.0 for ep in episodes]
        ),
    }
    for key in METRIC_KEYS:
        if key == "success_rate":
            continue
        summary[key] = _aggregate_stats([float(ep[key]) for ep in episodes])

    ee_mean, ee_std = _pad_and_aggregate_series(all_ee_errors)
    att_mean, att_std = _pad_and_aggregate_series(all_base_atts)
    dt = float(env.model.opt.timestep) * env.config.frame_skip
    steps = list(range(1, len(ee_mean) + 1))

    return {
        "method": method_name,
        "episodes": episodes,
        "summary": summary,
        "time_series": {
            "steps": steps,
            "time_s": [s * dt for s in steps],
            "dt": dt,
            "ee_error_mean": ee_mean,
            "ee_error_std": ee_std,
            "base_attitude_deg_mean": att_mean,
            "base_attitude_deg_std": att_std,
        },
    }


def plot_benchmark_comparison(
    results: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Create a multi-panel comparison figure (time series + metric bars)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = Figure(figsize=(13, 9), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    ax_ee, ax_att = axes[0, 0], axes[0, 1]
    ax_bar1, ax_bar2 = axes[1, 0], axes[1, 1]

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    methods = list(results.keys())

    # Top panels: time series (mean +/- 1 std)
    for idx, method in enumerate(methods):
        ts = results[method]["time_series"]
        t = np.asarray(ts["time_s"], dtype=np.float64)
        ee_m = np.asarray(ts["ee_error_mean"], dtype=np.float64)
        ee_s = np.asarray(ts["ee_error_std"], dtype=np.float64)
        att_m = np.asarray(ts["base_attitude_deg_mean"], dtype=np.float64)
        att_s = np.asarray(ts["base_attitude_deg_std"], dtype=np.float64)
        c = colors[idx % len(colors)]

        ax_ee.plot(t, ee_m, label=method, color=c, linewidth=2.0)
        ax_ee.fill_between(t, np.maximum(0.0, ee_m - ee_s), ee_m + ee_s, color=c, alpha=0.2)

        ax_att.plot(t, att_m, label=method, color=c, linewidth=2.0)
        ax_att.fill_between(
            t, np.maximum(0.0, att_m - att_s), att_m + att_s, color=c, alpha=0.2
        )

    ax_ee.set_title("End-Effector Position Error (mean ± 1 std)")
    ax_ee.set_xlabel("Time (s)")
    ax_ee.set_ylabel("EE Error (m)")
    ax_ee.grid(True, alpha=0.3)
    ax_ee.legend()

    ax_att.set_title("Base Attitude Drift (mean ± 1 std)")
    ax_att.set_xlabel("Time (s)")
    ax_att.set_ylabel("Base Attitude (deg)")
    ax_att.grid(True, alpha=0.3)
    ax_att.legend()

    # Bottom panels: grouped bar charts
    x = np.arange(len(methods), dtype=np.float64)
    width = 0.35

    succ_means = [results[m]["summary"]["success_rate"]["mean"] * 100.0 for m in methods]
    ee_final_means = [results[m]["summary"]["final_ee_error_m"]["mean"] for m in methods]
    ee_final_stds = [results[m]["summary"]["final_ee_error_m"]["std"] for m in methods]

    ax_bar1_twin = ax_bar1.twinx()
    b1 = ax_bar1.bar(
        x - width / 2, succ_means, width, label="Success Rate (%)", color="#2ca02c", alpha=0.85
    )
    b2 = ax_bar1_twin.bar(
        x + width / 2,
        ee_final_means,
        width,
        yerr=ee_final_stds,
        capsize=4,
        label="Final EE Error (m)",
        color="#1f77b4",
        alpha=0.85,
    )
    ax_bar1.set_xticks(x)
    ax_bar1.set_xticklabels(methods)
    ax_bar1.set_ylim(0.0, 110.0)
    ax_bar1.set_ylabel("Success Rate (%)")
    ax_bar1_twin.set_ylabel("Final EE Error (m)")
    ax_bar1.set_title("Accuracy & Success Rate")
    ax_bar1.grid(True, axis="y", alpha=0.3)
    ax_bar1.legend([b1, b2], ["Success Rate (%)", "Final EE Error (m)"], loc="upper right")

    att_max_means = [results[m]["summary"]["max_base_attitude_deg"]["mean"] for m in methods]
    att_max_stds = [results[m]["summary"]["max_base_attitude_deg"]["std"] for m in methods]
    fuel_means = [results[m]["summary"]["cumulative_fuel_impulse"]["mean"] for m in methods]
    fuel_stds = [results[m]["summary"]["cumulative_fuel_impulse"]["std"] for m in methods]

    ax_bar2_twin = ax_bar2.twinx()
    b3 = ax_bar2.bar(
        x - width / 2,
        att_max_means,
        width,
        yerr=att_max_stds,
        capsize=4,
        label="Max Base Attitude (deg)",
        color="#ff7f0e",
        alpha=0.85,
    )
    b4 = ax_bar2_twin.bar(
        x + width / 2,
        fuel_means,
        width,
        yerr=fuel_stds,
        capsize=4,
        label="Fuel Impulse",
        color="#9467bd",
        alpha=0.85,
    )
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(methods)
    ax_bar2.set_ylabel("Max Base Attitude (deg)")
    ax_bar2_twin.set_ylabel("Cumulative Fuel Impulse")
    ax_bar2.set_title("Base Disturbance & Fuel Usage")
    ax_bar2.grid(True, axis="y", alpha=0.3)
    ax_bar2.legend(
        [b3, b4], ["Max Base Attitude (deg)", "Fuel Impulse"], loc="upper right"
    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def plot_learning_curve(
    evaluations_path: str | Path,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Plot evaluation return (mean ± 1 std) from SB3 ``evaluations.npz``."""
    eval_npz = Path(evaluations_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(eval_npz)
    timesteps = np.asarray(data["timesteps"], dtype=np.float64)
    results = np.asarray(data["results"], dtype=np.float64)
    mean_ret = np.mean(results, axis=1)
    std_ret = np.std(results, axis=1)

    fig = Figure(figsize=(8, 5), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(timesteps, mean_ret, color="#1f77b4", linewidth=2.0, label="Mean Eval Return")
    ax.fill_between(
        timesteps, mean_ret - std_ret, mean_ret + std_ret, color="#1f77b4", alpha=0.2
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Return")
    ax.set_title(title or "SB3 Evaluation Learning Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out


def _infer_algo(model_path: Path, config_data: dict[str, Any] | None) -> str:
    """Infer SB3 algorithm ('ppo' or 'sac') from config.json or filename."""
    if config_data and config_data.get("algo") in VALID_ALGOS:
        return str(config_data["algo"])
    lowered = str(model_path).lower()
    if "sac" in lowered:
        return "sac"
    return "ppo"


def _load_sb3_model(algo: str, model_path: Path, device: str) -> PPO | SAC:
    """Load a saved SB3 PPO or SAC model."""
    if algo == "ppo":
        return PPO.load(str(model_path), device=device)
    if algo == "sac":
        return SAC.load(str(model_path), device=device)
    raise ValueError(f"Unsupported algo '{algo}'. Expected one of {VALID_ALGOS}.")


def _format_summary_table(results: dict[str, dict[str, Any]]) -> str:
    """Format a human-readable comparison table for terminal output."""
    methods = list(results.keys())
    header = f"{'Metric':<26} | " + " | ".join(f"{m:<22}" for m in methods)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for key in METRIC_KEYS:
        row_cells: list[str] = []
        for m in methods:
            stats = results[m]["summary"][key]
            row_cells.append(f"{stats['mean']:>8.4f} ± {stats['std']:<8.4f}")
        lines.append(f"{key:<26} | " + " | ".join(f"{c:<22}" for c in row_cells))
    lines.append(sep)
    return "\n".join(lines)


def _write_episodes_csv(
    results: dict[str, dict[str, Any]],
    csv_path: Path,
) -> Path:
    """Write per-episode benchmark metrics to CSV."""
    fieldnames = [
        "method",
        "episode_idx",
        "seed",
        "success",
        "final_ee_error_m",
        "mean_ee_error_m",
        "max_base_attitude_deg",
        "final_base_attitude_deg",
        "cumulative_fuel_impulse",
        "mean_arm_effort",
        "episode_return",
        "episode_length",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, res in results.items():
            for ep in res["episodes"]:
                row = {"method": method, **ep}
                writer.writerow(row)
    return csv_path


def run_benchmark(
    *,
    model_path: str | Path | None = None,
    baseline_only: bool = False,
    algo: str | None = None,
    mission: MissionMode | None = None,
    action_mode: ActionMode | None = None,
    n_episodes: int = 10,
    seed: int = 42,
    frame_skip: int | None = None,
    max_episode_steps: int | None = None,
    init_base_angvel_std: float | None = None,
    vecnormalize_path: str | Path | None = None,
    device: str = "auto",
    output_dir: str | Path = "runs/benchmark",
) -> dict[str, Any]:
    """Run paired RL vs. classical GJM benchmark and export tables, CSV, JSON, and plots."""
    if not baseline_only and model_path is None:
        raise ValueError("model_path is required unless baseline_only=True.")

    resolved_model_path = Path(model_path) if model_path is not None else None
    run_config: dict[str, Any] | None = None
    if resolved_model_path is not None:
        cfg_file = resolved_model_path.parent / "config.json"
        if cfg_file.exists():
            run_config = json.loads(cfg_file.read_text(encoding="utf-8"))

    resolved_mission: MissionMode = (
        mission
        or (run_config.get("mission") if run_config else None)
        or "reach"
    )
    resolved_action_mode: ActionMode = (
        action_mode
        or (run_config.get("action_mode") if run_config else None)
        or "torque"
    )
    resolved_frame_skip = (
        frame_skip
        if frame_skip is not None
        else int(run_config["frame_skip"]) if run_config and "frame_skip" in run_config else 10
    )
    resolved_max_steps = (
        max_episode_steps
        if max_episode_steps is not None
        else int(run_config["max_episode_steps"])
        if run_config and "max_episode_steps" in run_config
        else 250
    )
    resolved_angvel_std = (
        init_base_angvel_std
        if init_base_angvel_std is not None
        else float(run_config["init_base_angvel_std"])
        if run_config and "init_base_angvel_std" in run_config
        else 0.0
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    # 1. Classical GJM baseline (residual_gjm with zero residual action)
    gjm_cfg = EnvConfig(
        mission=resolved_mission,
        action_mode="residual_gjm",
        frame_skip=resolved_frame_skip,
        max_episode_steps=resolved_max_steps,
        init_base_angvel_std=resolved_angvel_std,
    )
    gjm_env = SpaceManipulatorEnv(gjm_cfg)
    try:
        zero_action = np.zeros(gjm_env.action_space.shape, dtype=np.float32)
        results["GJM Classic"] = evaluate_policy(
            gjm_env,
            lambda _obs: zero_action,
            n_episodes=n_episodes,
            seed=seed,
            method_name="GJM Classic",
        )
    finally:
        gjm_env.close()

    # 2. RL policy (if not baseline_only)
    learning_curve_path: Path | None = None
    if not baseline_only and resolved_model_path is not None:
        resolved_algo = algo or _infer_algo(resolved_model_path, run_config)
        model = _load_sb3_model(resolved_algo, resolved_model_path, device=device)

        rl_cfg = EnvConfig(
            mission=resolved_mission,
            action_mode=resolved_action_mode,
            frame_skip=resolved_frame_skip,
            max_episode_steps=resolved_max_steps,
            init_base_angvel_std=resolved_angvel_std,
        )

        resolved_vecnorm: Path | None = None
        if vecnormalize_path is not None:
            resolved_vecnorm = Path(vecnormalize_path)
        else:
            candidate = resolved_model_path.parent / "vecnormalize.pkl"
            if candidate.exists():
                resolved_vecnorm = candidate

        vec_norm_env: VecNormalize | None = None
        obs_normalizer: Callable[[NDArray[np.float32]], NDArray[np.float32]] | None = None
        if resolved_vecnorm is not None and resolved_vecnorm.exists():
            dummy = DummyVecEnv([lambda: SpaceManipulatorEnv(rl_cfg)])
            vec_norm_env = VecNormalize.load(str(resolved_vecnorm), dummy)
            vec_norm_env.training = False
            vec_norm_env.norm_reward = False
            obs_normalizer = lambda obs: vec_norm_env.normalize_obs(obs)  # noqa: E731

        rl_env = SpaceManipulatorEnv(rl_cfg)
        try:
            method_label = f"RL ({resolved_algo.upper()})"
            results[method_label] = evaluate_policy(
                rl_env,
                lambda obs: model.predict(obs, deterministic=True)[0],
                n_episodes=n_episodes,
                seed=seed,
                method_name=method_label,
                obs_normalizer=obs_normalizer,
            )
        finally:
            rl_env.close()
            if vec_norm_env is not None:
                vec_norm_env.close()

        eval_npz = resolved_model_path.parent / "evaluations.npz"
        if eval_npz.exists():
            learning_curve_path = plot_learning_curve(
                eval_npz,
                out_dir / "learning_curve.png",
                title=f"Learning Curve ({resolved_mission} / {resolved_algo.upper()})",
            )

    summary_payload = {
        "metadata": {
            "mission": resolved_mission,
            "action_mode": resolved_action_mode,
            "n_episodes": n_episodes,
            "seed": seed,
            "frame_skip": resolved_frame_skip,
            "max_episode_steps": resolved_max_steps,
            "init_base_angvel_std": resolved_angvel_std,
            "baseline_only": baseline_only,
            "model_path": str(resolved_model_path) if resolved_model_path else None,
        },
        "methods": {
            m: {
                "summary": res["summary"],
                "time_series": res["time_series"],
            }
            for m, res in results.items()
        },
    }

    summary_json_path = out_dir / "benchmark_summary.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    episodes_csv_path = _write_episodes_csv(results, out_dir / "benchmark_episodes.csv")
    comparison_plot_path = plot_benchmark_comparison(
        results,
        out_dir / "benchmark_comparison.png",
        title=f"Benchmark: {resolved_mission} ({n_episodes} paired episodes)",
    )

    print(_format_summary_table(results))

    return {
        "output_dir": out_dir,
        "summary_json_path": summary_json_path,
        "episodes_csv_path": episodes_csv_path,
        "comparison_plot_path": comparison_plot_path,
        "learning_curve_path": learning_curve_path,
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``space-manipulator-benchmark``."""
    parser = argparse.ArgumentParser(
        description="Benchmark SB3 RL policy against classical GJM controller."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--algo", choices=VALID_ALGOS, default=None)
    parser.add_argument("--mission", choices=VALID_MISSION_MODES, default=None)
    parser.add_argument("--action-mode", choices=VALID_ACTION_MODES, default=None)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--init-base-angvel-std", type=float, default=None)
    parser.add_argument("--vecnormalize-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="runs/benchmark")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for RL vs. GJM benchmark."""
    args = parse_args(argv)
    out = run_benchmark(
        model_path=args.model_path,
        baseline_only=args.baseline_only,
        algo=args.algo,
        mission=args.mission,
        action_mode=args.action_mode,
        n_episodes=args.n_episodes,
        seed=args.seed,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        init_base_angvel_std=args.init_base_angvel_std,
        vecnormalize_path=args.vecnormalize_path,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(f"Benchmark artifacts saved to: {out['output_dir']}")


if __name__ == "__main__":
    main()
