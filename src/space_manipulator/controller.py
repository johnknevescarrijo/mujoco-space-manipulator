"""Task-space controllers for free-floating and free-flying space manipulator modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from space_manipulator.kinematics import (
    compute_generalized_jacobian,
    compute_site_jacobian,
    damped_pseudoinverse,
    normalize_quaternion,
    orientation_error,
)
from space_manipulator.model_loader import ModelIndices, get_model_indices, get_site_pose

ControlMode = Literal["free-floating", "free-flying"]
VALID_CONTROL_MODES: tuple[str, ...] = ("free-floating", "free-flying")


@dataclass(frozen=True)
class ControllerGains:
    """Gain configuration for the 3-DoF arm and 6-DoF base controllers."""

    kp_task: float = 4.0
    max_ee_speed: float = 0.6
    dls_damping: float = 0.04
    kv_joint: float = 14.0
    kd_joint: float = 1.0
    kp_base_pos: float = 400.0
    kd_base_pos: float = 180.0
    kp_base_att: float = 120.0
    kd_base_att: float = 35.0


@dataclass(frozen=True)
class ControlOutput:
    """Actuator commands and instantaneous tracking diagnostics."""

    u_base: NDArray[np.float64]
    u_arm: NDArray[np.float64]
    ee_error: NDArray[np.float64]
    ee_error_norm: float
    base_pos_error: NDArray[np.float64]
    base_att_error: NDArray[np.float64]
    jacobian_used: NDArray[np.float64]


class SpaceManipulatorController:
    """Task-space controller supporting ``free-floating`` and ``free-flying`` modes."""

    def __init__(
        self,
        model: mujoco.MjModel,
        mode: ControlMode = "free-floating",
        gains: ControllerGains | None = None,
    ) -> None:
        if mode not in VALID_CONTROL_MODES:
            raise ValueError(
                f"Invalid control mode '{mode}'. Expected one of {VALID_CONTROL_MODES}."
            )
        self.model: mujoco.MjModel = model
        self.mode: ControlMode = mode
        self.gains: ControllerGains = gains if gains is not None else ControllerGains()
        self.indices: ModelIndices = get_model_indices(model)

    def _clip_actuators(
        self, actuator_ids: tuple[int, ...], raw: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        ids = list(actuator_ids)
        low = self.model.actuator_ctrlrange[ids, 0]
        high = self.model.actuator_ctrlrange[ids, 1]
        return np.clip(raw, low, high)

    def compute_control(
        self,
        data: mujoco.MjData,
        *,
        target_pos: ArrayLike | None = None,
        base_ref_pos: ArrayLike | None = None,
        base_ref_quat: ArrayLike | None = None,
    ) -> ControlOutput:
        """Compute base and arm control commands for the current state."""
        idx = self.indices
        g = self.gains

        # Ensure kinematics, Jacobians, and bias forces match current qpos/qvel
        mujoco.mj_forward(self.model, data)

        ee_pos, _ = get_site_pose(self.model, data, "ee_site")
        if target_pos is None:
            goal_pos, _ = get_site_pose(self.model, data, "target_site")
        else:
            goal_pos = np.asarray(target_pos, dtype=float)
            if goal_pos.shape != (3,):
                raise ValueError(
                    f"Expected target_pos shape (3,), got {goal_pos.shape}."
                )

        ref_pos = (
            np.zeros(3, dtype=float)
            if base_ref_pos is None
            else np.asarray(base_ref_pos, dtype=float)
        )
        if ref_pos.shape != (3,):
            raise ValueError(f"Expected base_ref_pos shape (3,), got {ref_pos.shape}.")

        ref_quat = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if base_ref_quat is None
            else normalize_quaternion(base_ref_quat)
        )

        ee_error = goal_pos - ee_pos
        ee_error_norm = float(np.linalg.norm(ee_error))
        v_ee_des = g.kp_task * ee_error
        v_norm = float(np.linalg.norm(v_ee_des))
        if v_norm > g.max_ee_speed and v_norm > 0.0:
            v_ee_des = v_ee_des * (g.max_ee_speed / v_norm)

        base_pos, base_rot = get_site_pose(self.model, data, "base_imu_site")
        base_quat = data.qpos[idx.base_qpos_adr + 3 : idx.base_qpos_adr + 7].copy()
        base_pos_error = ref_pos - base_pos
        base_att_error = orientation_error(base_quat, ref_quat, frame="body")

        if self.mode == "free-floating":
            gjm = compute_generalized_jacobian(
                self.model, data, site_name="ee_site", indices=idx
            )
            j_use = gjm.j_star
            u_base = np.zeros(len(idx.base_actuator_ids), dtype=float)
        else:
            _, j_m = compute_site_jacobian(
                self.model, data, site_name="ee_site", indices=idx
            )
            j_use = j_m

            base_linvel = data.qvel[idx.base_qvel_adr : idx.base_qvel_adr + 3]
            base_angvel_local = data.qvel[
                idx.base_qvel_adr + 3 : idx.base_qvel_adr + 6
            ]
            bias_lin = data.qfrc_bias[idx.base_qvel_adr : idx.base_qvel_adr + 3]
            bias_ang = data.qfrc_bias[idx.base_qvel_adr + 3 : idx.base_qvel_adr + 6]

            f_world = g.kp_base_pos * base_pos_error - g.kd_base_pos * base_linvel + bias_lin
            # base_fx..fz act along base_imu_site local axes
            f_local = base_rot.T @ f_world
            t_local = (
                g.kp_base_att * base_att_error
                - g.kd_base_att * base_angvel_local
                + bias_ang
            )
            u_base = self._clip_actuators(
                idx.base_actuator_ids, np.concatenate([f_local, t_local])
            )

        arm_qvel_idx = list(idx.arm_qvel_adrs)
        qd_arm = data.qvel[arm_qvel_idx]
        qfrc_bias_arm = data.qfrc_bias[arm_qvel_idx]

        j_pinv = damped_pseudoinverse(j_use, damping=g.dls_damping)
        qd_arm_des = j_pinv @ v_ee_des
        tau_arm_raw = (
            g.kv_joint * (qd_arm_des - qd_arm) - g.kd_joint * qd_arm + qfrc_bias_arm
        )
        u_arm = self._clip_actuators(idx.arm_actuator_ids, tau_arm_raw)

        return ControlOutput(
            u_base=u_base,
            u_arm=u_arm,
            ee_error=ee_error,
            ee_error_norm=ee_error_norm,
            base_pos_error=base_pos_error,
            base_att_error=base_att_error,
            jacobian_used=j_use,
        )

    def apply_control(self, data: mujoco.MjData, control: ControlOutput) -> None:
        """Write computed base and arm actuator commands into ``data.ctrl``."""
        data.ctrl[list(self.indices.base_actuator_ids)] = control.u_base
        data.ctrl[list(self.indices.arm_actuator_ids)] = control.u_arm

    def step(
        self,
        data: mujoco.MjData,
        *,
        target_pos: ArrayLike | None = None,
        base_ref_pos: ArrayLike | None = None,
        base_ref_quat: ArrayLike | None = None,
    ) -> ControlOutput:
        """Compute control, apply it to ``data.ctrl``, and advance one timestep."""
        control = self.compute_control(
            data,
            target_pos=target_pos,
            base_ref_pos=base_ref_pos,
            base_ref_quat=base_ref_quat,
        )
        self.apply_control(data, control)
        mujoco.mj_step(self.model, data)
        return control
