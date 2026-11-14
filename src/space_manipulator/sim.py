"""Simulation runner, telemetry logging, matplotlib plotting, and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time as time_mod
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from space_manipulator.controller import ControlMode, SpaceManipulatorController
from space_manipulator.kinematics import compute_system_momentum, orientation_error
from space_manipulator.model_loader import (
    get_model_indices,
    get_site_pose,
    load_model,
    read_sensor,
    reset_state,
)


@dataclass(frozen=True)
class SimulationTelemetry:
    """Time-series telemetry recorded during a space manipulator simulation."""

    mode: ControlMode
    target_pos: NDArray[np.float64]
    time: NDArray[np.float64]
    ee_pos: NDArray[np.float64]
    ee_error_norm: NDArray[np.float64]
    base_pos: NDArray[np.float64]
    base_pos_drift: NDArray[np.float64]
    base_ori_error_deg: NDArray[np.float64]
    joint_pos: NDArray[np.float64]
    joint_torques: NDArray[np.float64]
    base_wrench: NDArray[np.float64]
    linear_momentum_norm: NDArray[np.float64]
    angular_momentum_norm: NDArray[np.float64]
    com_pos: NDArray[np.float64]


def run_simulation(
    mode: ControlMode = "free-floating",
    duration: float = 8.0,
    target_pos: ArrayLike = (1.15, 0.35, 0.25),
    headless: bool = True,
    realtime: bool = False,
    *,
    arm_qpos: ArrayLike = (0.0, 0.3, -0.5),
) -> SimulationTelemetry:
    """Run a closed-loop MuJoCo simulation and return recorded telemetry."""
    if duration <= 0.0:
        raise ValueError("Simulation duration must be positive.")

    target_arr = np.asarray(target_pos, dtype=float)
    if target_arr.shape != (3,):
        raise ValueError(f"Expected target_pos shape (3,), got {target_arr.shape}.")

    model, data = load_model()
    indices = get_model_indices(model)
    base_ref_pos = np.zeros(3, dtype=float)
    base_ref_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    reset_state(
        model,
        data,
        base_pos=base_ref_pos,
        base_quat=base_ref_quat,
        arm_qpos=arm_qpos,
        target_pos=target_arr,
    )

    controller = SpaceManipulatorController(model, mode=mode)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_sat")
    dt = float(model.opt.timestep)
    n_steps = max(1, int(round(duration / dt)))

    t_log = np.zeros(n_steps, dtype=float)
    ee_pos_log = np.zeros((n_steps, 3), dtype=float)
    ee_err_log = np.zeros(n_steps, dtype=float)
    base_pos_log = np.zeros((n_steps, 3), dtype=float)
    base_drift_log = np.zeros(n_steps, dtype=float)
    base_ori_deg_log = np.zeros(n_steps, dtype=float)
    joint_pos_log = np.zeros((n_steps, 3), dtype=float)
    joint_torques_log = np.zeros((n_steps, 3), dtype=float)
    base_wrench_log = np.zeros((n_steps, 6), dtype=float)
    lin_mom_log = np.zeros(n_steps, dtype=float)
    ang_mom_log = np.zeros(n_steps, dtype=float)
    com_pos_log = np.zeros((n_steps, 3), dtype=float)

    arm_qpos_idx = list(indices.arm_qpos_adrs)

    def _step_and_record(step_idx: int) -> None:
        out = controller.step(
            data,
            target_pos=target_arr,
            base_ref_pos=base_ref_pos,
            base_ref_quat=base_ref_quat,
        )
        mujoco.mj_forward(model, data)

        ee_pos, _ = get_site_pose(model, data, "ee_site")
        base_pos, _ = get_site_pose(model, data, "base_imu_site")
        base_quat = read_sensor(model, data, "base_quat")
        att_err = orientation_error(base_quat, base_ref_quat, frame="body")
        lin_mom, ang_mom = compute_system_momentum(model, data)

        t_log[step_idx] = float(data.time)
        ee_pos_log[step_idx] = ee_pos
        ee_err_log[step_idx] = float(np.linalg.norm(target_arr - ee_pos))
        base_pos_log[step_idx] = base_pos
        base_drift_log[step_idx] = float(np.linalg.norm(base_pos - base_ref_pos))
        base_ori_deg_log[step_idx] = float(np.rad2deg(np.linalg.norm(att_err)))
        joint_pos_log[step_idx] = data.qpos[arm_qpos_idx]
        joint_torques_log[step_idx] = out.u_arm
        base_wrench_log[step_idx] = out.u_base
        lin_mom_log[step_idx] = float(np.linalg.norm(lin_mom))
        ang_mom_log[step_idx] = float(np.linalg.norm(ang_mom))
        com_pos_log[step_idx] = data.subtree_com[base_body_id].copy()

    if headless:
        for i in range(n_steps):
            step_start = time_mod.perf_counter()
            _step_and_record(i)
            if realtime:
                remaining = dt - (time_mod.perf_counter() - step_start)
                if remaining > 0.0:
                    time_mod.sleep(remaining)
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            for i in range(n_steps):
                if not viewer.is_running():
                    n_steps = i
                    break
                step_start = time_mod.perf_counter()
                _step_and_record(i)
                viewer.sync()
                if realtime or True:
                    remaining = dt - (time_mod.perf_counter() - step_start)
                    if remaining > 0.0:
                        time_mod.sleep(remaining)

    return SimulationTelemetry(
        mode=mode,
        target_pos=target_arr.copy(),
        time=t_log[:n_steps],
        ee_pos=ee_pos_log[:n_steps],
        ee_error_norm=ee_err_log[:n_steps],
        base_pos=base_pos_log[:n_steps],
        base_pos_drift=base_drift_log[:n_steps],
        base_ori_error_deg=base_ori_deg_log[:n_steps],
        joint_pos=joint_pos_log[:n_steps],
        joint_torques=joint_torques_log[:n_steps],
        base_wrench=base_wrench_log[:n_steps],
        linear_momentum_norm=lin_mom_log[:n_steps],
        angular_momentum_norm=ang_mom_log[:n_steps],
        com_pos=com_pos_log[:n_steps],
    )


def _save_figure(fig: Figure, output_path: Path | str) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def plot_telemetry(
    telemetry: SimulationTelemetry,
    output_path: Path | str,
) -> Path:
    """Plot single-mode simulation telemetry and save to ``output_path``."""
    fig = Figure(figsize=(11, 9))
    axes = fig.subplots(3, 1, sharex=True)

    # 1. End-effector tracking error
    ax0 = axes[0]
    ax0.plot(telemetry.time, telemetry.ee_error_norm * 100.0, color="#1f77b4", lw=2.0)
    ax0.axhline(2.0, color="#d62728", ls="--", lw=1.0, label="2 cm threshold")
    ax0.set_ylabel("EE Error [cm]")
    ax0.set_title(
        f"Space Manipulator Telemetry ({telemetry.mode}) | Target = "
        f"({telemetry.target_pos[0]:.2f}, {telemetry.target_pos[1]:.2f}, {telemetry.target_pos[2]:.2f}) m"
    )
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    # 2. Base reaction: displacement [mm] and rotation [deg]
    ax1 = axes[1]
    ax1_r = ax1.twinx()
    l1 = ax1.plot(
        telemetry.time,
        telemetry.base_pos_drift * 1000.0,
        color="#ff7f0e",
        lw=2.0,
        label="Base displacement [mm]",
    )
    l2 = ax1_r.plot(
        telemetry.time,
        telemetry.base_ori_error_deg,
        color="#2ca02c",
        lw=1.8,
        ls="-.",
        label="Base rotation [deg]",
    )
    ax1.set_ylabel("Base Drift [mm]", color="#ff7f0e")
    ax1_r.set_ylabel("Base Rotation [deg]", color="#2ca02c")
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left")

    # 3. Angular momentum & joint torques
    ax2 = axes[2]
    ax2_r = ax2.twinx()
    torque_lines = []
    colors = ("#9467bd", "#8c564b", "#e377c2")
    for j in range(3):
        torque_lines.extend(
            ax2.plot(
                telemetry.time,
                telemetry.joint_torques[:, j],
                color=colors[j],
                lw=1.5,
                label=f"tau_joint{j + 1} [N*m]",
            )
        )
    mom_line = ax2_r.plot(
        telemetry.time,
        telemetry.angular_momentum_norm,
        color="#17becf",
        lw=1.8,
        ls="--",
        label="||Angular Momentum|| [N*m*s]",
    )
    ax2.set_ylabel("Joint Torques [N*m]")
    ax2_r.set_ylabel("||L|| [N*m*s]", color="#17becf")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    all_lines = torque_lines + mom_line
    ax2.legend(all_lines, [ln.get_label() for ln in all_lines], loc="upper right", ncol=2)

    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_comparison(
    tel_floating: SimulationTelemetry,
    tel_flying: SimulationTelemetry,
    output_path: Path | str,
) -> Path:
    """Plot side-by-side comparison of free-floating vs free-flying telemetry."""
    fig = Figure(figsize=(12, 10))
    axes = fig.subplots(4, 1, sharex=True)

    # 1. End-effector tracking error
    ax0 = axes[0]
    ax0.plot(
        tel_floating.time,
        tel_floating.ee_error_norm * 100.0,
        color="#1f77b4",
        lw=2.0,
        label="free-floating (GJM J*)",
    )
    ax0.plot(
        tel_flying.time,
        tel_flying.ee_error_norm * 100.0,
        color="#d62728",
        lw=2.0,
        ls="--",
        label="free-flying (RCS + J_m)",
    )
    ax0.axhline(2.0, color="gray", ls=":", lw=1.0, label="2 cm threshold")
    ax0.set_ylabel("EE Error [cm]")
    ax0.set_title("Free-Floating vs Free-Flying Space Manipulator Comparison")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right")

    # 2. Base displacement & rotation reaction
    ax1 = axes[1]
    ax1_r = ax1.twinx()
    l1 = ax1.plot(
        tel_floating.time,
        tel_floating.base_pos_drift * 1000.0,
        color="#1f77b4",
        lw=2.0,
        label="Drift free-floating [mm]",
    )
    l2 = ax1.plot(
        tel_flying.time,
        tel_flying.base_pos_drift * 1000.0,
        color="#d62728",
        lw=2.0,
        ls="--",
        label="Drift free-flying [mm]",
    )
    l3 = ax1_r.plot(
        tel_floating.time,
        tel_floating.base_ori_error_deg,
        color="#2ca02c",
        lw=1.6,
        ls="-.",
        label="Rot free-floating [deg]",
    )
    l4 = ax1_r.plot(
        tel_flying.time,
        tel_flying.base_ori_error_deg,
        color="#ff7f0e",
        lw=1.6,
        ls=":",
        label="Rot free-flying [deg]",
    )
    ax1.set_ylabel("Base Drift [mm]")
    ax1_r.set_ylabel("Base Rotation [deg]")
    ax1.grid(True, alpha=0.3)
    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", ncol=2)

    # 3. Total angular momentum norm
    ax2 = axes[2]
    ax2.plot(
        tel_floating.time,
        tel_floating.angular_momentum_norm,
        color="#1f77b4",
        lw=2.0,
        label="||L|| free-floating (conserved ~ 0)",
    )
    ax2.plot(
        tel_flying.time,
        tel_flying.angular_momentum_norm,
        color="#d62728",
        lw=1.8,
        ls="--",
        label="||L|| free-flying (external RCS torque)",
    )
    ax2.set_ylabel("||L|| [N*m*s]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 4. Arm joint torque norms
    ax3 = axes[3]
    tau_float_norm = np.linalg.norm(tel_floating.joint_torques, axis=1)
    tau_fly_norm = np.linalg.norm(tel_flying.joint_torques, axis=1)
    ax3.plot(
        tel_floating.time,
        tau_float_norm,
        color="#1f77b4",
        lw=1.8,
        label="||tau_arm|| free-floating [N*m]",
    )
    ax3.plot(
        tel_flying.time,
        tau_fly_norm,
        color="#d62728",
        lw=1.8,
        ls="--",
        label="||tau_arm|| free-flying [N*m]",
    )
    ax3.set_ylabel("||tau_arm|| [N*m]")
    ax3.set_xlabel("Time [s]")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right")

    fig.tight_layout()
    return _save_figure(fig, output_path)


def _format_summary(telemetry: SimulationTelemetry) -> str:
    final_ee_err_cm = float(telemetry.ee_error_norm[-1] * 100.0)
    final_drift_mm = float(telemetry.base_pos_drift[-1] * 1000.0)
    max_drift_mm = float(np.max(telemetry.base_pos_drift) * 1000.0)
    final_ori_deg = float(telemetry.base_ori_error_deg[-1])
    max_lin_mom = float(np.max(telemetry.linear_momentum_norm))
    max_ang_mom = float(np.max(telemetry.angular_momentum_norm))
    com_drift_mm = float(
        np.max(np.linalg.norm(telemetry.com_pos - telemetry.com_pos[0], axis=1))
        * 1000.0
    )
    return (
        f"[{telemetry.mode}] duration={telemetry.time[-1]:.2f}s | "
        f"final_ee_err={final_ee_err_cm:.4f} cm | "
        f"base_drift(final/max)={final_drift_mm:.2f}/{max_drift_mm:.2f} mm | "
        f"base_rot_final={final_ori_deg:.3f} deg | "
        f"max_||P||={max_lin_mom:.2e} kg*m/s | "
        f"max_||L||={max_ang_mom:.2e} N*m*s | "
        f"com_drift={com_drift_mm:.4f} mm"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for ``space-manipulator``."""
    parser = argparse.ArgumentParser(
        prog="space-manipulator",
        description="Run MuJoCo space manipulator simulation (free-floating, free-flying, or compare).",
    )
    parser.add_argument(
        "--mode",
        choices=("free-floating", "free-flying", "compare"),
        default="free-floating",
        help="Control mode or side-by-side comparison (default: free-floating).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Simulation duration in seconds (default: 8.0).",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(1.15, 0.35, 0.25),
        metavar=("X", "Y", "Z"),
        help="3D target position in meters (default: 1.15 0.35 0.25).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo interactive 3D viewer.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace headless execution to wall-clock time.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output PNG path to save telemetry or comparison plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for running simulations and generating telemetry plots."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "compare":
        tel_floating = run_simulation(
            mode="free-floating",
            duration=args.duration,
            target_pos=args.target,
            headless=args.headless,
            realtime=args.realtime,
        )
        tel_flying = run_simulation(
            mode="free-flying",
            duration=args.duration,
            target_pos=args.target,
            headless=args.headless,
            realtime=args.realtime,
        )
        print(_format_summary(tel_floating))
        print(_format_summary(tel_flying))
        if args.plot is not None:
            saved = plot_comparison(tel_floating, tel_flying, args.plot)
            print(f"Saved comparison plot: {saved}")
    else:
        tel = run_simulation(
            mode=args.mode,
            duration=args.duration,
            target_pos=args.target,
            headless=args.headless,
            realtime=args.realtime,
        )
        print(_format_summary(tel))
        if args.plot is not None:
            saved = plot_telemetry(tel, args.plot)
            print(f"Saved telemetry plot: {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
