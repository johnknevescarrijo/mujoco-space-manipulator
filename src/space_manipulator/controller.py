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

        # Precompute cached indices and actuator limits for hot-path control loops
        self._base_act_ids: list[int] = list(self.indices.base_actuator_ids)
        self._arm_act_ids: list[int] = list(self.indices.arm_actuator_ids)
        self._arm_qvel_idx: list[int] = list(self.indices.arm_qvel_adrs)
        self._base_ctrl_low: NDArray[np.float64] = model.actuator_ctrlrange[
            self._base_act_ids, 0
        ].copy()
        self._base_ctrl_high: NDArray[np.float64] = model.actuator_ctrlrange[
            self._base_act_ids, 1
        ].copy()
        self._arm_ctrl_low: NDArray[np.float64] = model.actuator_ctrlrange[
            self._arm_act_ids, 0
        ].copy()
        self._arm_ctrl_high: NDArray[np.float64] = model.actuator_ctrlrange[
            self._arm_act_ids, 1
        ].copy()
        self._default_ref_pos: NDArray[np.float64] = np.zeros(3, dtype=float)
        self._default_ref_quat: NDArray[np.float64] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )
        self._zero_base_ctrl: NDArray[np.float64] = np.zeros(
            len(self._base_act_ids), dtype=float
        )

    def _clip_actuators(
        self, actuator_ids: tuple[int, ...], raw: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if actuator_ids == self.indices.base_actuator_ids:
            return np.clip(raw, self._base_ctrl_low, self._base_ctrl_high)
        if actuator_ids == self.indices.arm_actuator_ids:
            return np.clip(raw, self._arm_ctrl_low, self._arm_ctrl_high)
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
        forward: bool = True,
    ) -> ControlOutput:
        """Compute base and arm control commands for the current state.

        Args:
            data: Current MuJoCo simulation state.
            target_pos: Optional 3D target position overriding ``target_site``.
            base_ref_pos: Optional 3D base reference position.
            base_ref_quat: Optional base reference quaternion ``[w, x, y, z]``.
            forward: When True (default), runs ``mj_forward`` so kinematics and
                bias forces match ``qpos``/``qvel``. Set False only if the caller
                already ran ``mj_forward`` on the current state.
        """
        idx = self.indices
        g = self.gains

        # Ensure kinematics, Jacobians, and bias forces match current qpos/qvel
        if forward:
            mujoco.mj_forward(self.model, data)

        ee_pos, _ = get_site_pose(self.model, data, "ee_site", indices=idx)
        if target_pos is None:
            goal_pos, _ = get_site_pose(self.model, data, "target_site", indices=idx)
        else:
            goal_pos = np.asarray(target_pos, dtype=float)
            if goal_pos.shape != (3,):
                raise ValueError(
                    f"Expected target_pos shape (3,), got {goal_pos.shape}."
                )

        if base_ref_pos is None:
            ref_pos = self._default_ref_pos
        else:
            ref_pos = np.asarray(base_ref_pos, dtype=float)
            if ref_pos.shape != (3,):
                raise ValueError(
                    f"Expected base_ref_pos shape (3,), got {ref_pos.shape}."
                )

        ref_quat = (
            self._default_ref_quat
            if base_ref_quat is None
            else normalize_quaternion(base_ref_quat)
        )

        ee_error = goal_pos - ee_pos
        ee_error_norm = float(np.linalg.norm(ee_error))
        v_ee_des = g.kp_task * ee_error
        v_norm = float(np.linalg.norm(v_ee_des))
        if v_norm > g.max_ee_speed and v_norm > 0.0:
            v_ee_des = v_ee_des * (g.max_ee_speed / v_norm)

        base_pos, base_rot = get_site_pose(
            self.model, data, "base_imu_site", indices=idx
        )
        base_quat = data.qpos[idx.base_qpos_adr + 3 : idx.base_qpos_adr + 7].copy()
        base_pos_error = ref_pos - base_pos
        base_att_error = orientation_error(base_quat, ref_quat, frame="body")

        if self.mode == "free-floating":
            gjm = compute_generalized_jacobian(
                self.model, data, site_name="ee_site", indices=idx
            )
            j_use = gjm.j_star
            u_base = self._zero_base_ctrl.copy()
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
            u_base = np.clip(
                np.concatenate([f_local, t_local]),
                self._base_ctrl_low,
                self._base_ctrl_high,
            )

        qd_arm = data.qvel[self._arm_qvel_idx]
        qfrc_bias_arm = data.qfrc_bias[self._arm_qvel_idx]

        j_pinv = damped_pseudoinverse(j_use, damping=g.dls_damping)
        qd_arm_des = j_pinv @ v_ee_des
        tau_arm_raw = (
            g.kv_joint * (qd_arm_des - qd_arm) - g.kd_joint * qd_arm + qfrc_bias_arm
        )
        u_arm = np.clip(tau_arm_raw, self._arm_ctrl_low, self._arm_ctrl_high)

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
        data.ctrl[self._base_act_ids] = control.u_base
        data.ctrl[self._arm_act_ids] = control.u_arm

    def step(
        self,
        data: mujoco.MjData,
        *,
        target_pos: ArrayLike | None = None,
        base_ref_pos: ArrayLike | None = None,
        base_ref_quat: ArrayLike | None = None,
        forward: bool = True,
    ) -> ControlOutput:
        """Compute control, apply it to ``data.ctrl``, and advance one timestep."""
        control = self.compute_control(
            data,
            target_pos=target_pos,
            base_ref_pos=base_ref_pos,
            base_ref_quat=base_ref_quat,
            forward=forward,
        )
        self.apply_control(data, control)
        mujoco.mj_step(self.model, data)
        return control
