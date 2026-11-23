"""Gymnasium RL environments for the 3-DoF space manipulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from numpy.typing import NDArray

from space_manipulator.controller import SpaceManipulatorController
from space_manipulator.kinematics import orientation_error
from space_manipulator.model_loader import (
    ModelIndices,
    get_model_indices,
    load_model,
    reset_state,
    set_target_pose,
)

MissionMode = Literal["reach", "low_disturbance", "fuel_optimal", "moving_target"]
ActionMode = Literal["torque", "residual_gjm"]

VALID_MISSION_MODES: tuple[str, ...] = (
    "reach",
    "low_disturbance",
    "fuel_optimal",
    "moving_target",
)
VALID_ACTION_MODES: tuple[str, ...] = ("torque", "residual_gjm")
VALID_RENDER_MODES: tuple[str, ...] = ("rgb_array",)

OBS_DIM: int = 28
OBS_ORDER: tuple[str, ...] = (
    "arm_qpos[0:3]",
    "arm_qvel[3:6]",
    "base_quat[6:10]",
    "base_angvel_body[10:13]",
    "base_linvel_world[13:16]",
    "ee_pos_rel_base[16:19]",
    "ee_to_target_world[19:22]",
    "ee_vel_world[22:25]",
    "target_vel_world[25:28]",
)

_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class RewardWeights:
    """Reward component weights for space manipulator RL missions."""

    distance: float = 2.0
    progress: float = 10.0
    ee_vel: float = 0.0
    arm_effort: float = 0.01
    base_attitude: float = 0.0
    base_angvel: float = 0.0
    fuel_impulse: float = 0.0
    collision: float = 5.0
    success_bonus: float = 10.0

    @classmethod
    def for_mission(cls, mission: MissionMode) -> RewardWeights:
        """Return default reward weights tuned for the given mission."""
        if mission == "reach":
            return cls(
                distance=2.0,
                progress=10.0,
                ee_vel=0.0,
                arm_effort=0.01,
                base_attitude=0.0,
                base_angvel=0.0,
                fuel_impulse=0.0,
                collision=5.0,
                success_bonus=10.0,
            )
        if mission == "low_disturbance":
            return cls(
                distance=2.0,
                progress=10.0,
                ee_vel=0.0,
                arm_effort=0.01,
                base_attitude=3.0,
                base_angvel=1.5,
                fuel_impulse=0.0,
                collision=5.0,
                success_bonus=10.0,
            )
        if mission == "fuel_optimal":
            return cls(
                distance=2.0,
                progress=10.0,
                ee_vel=0.0,
                arm_effort=0.01,
                base_attitude=1.5,
                base_angvel=0.5,
                fuel_impulse=0.8,
                collision=5.0,
                success_bonus=10.0,
            )
        if mission == "moving_target":
            return cls(
                distance=2.5,
                progress=5.0,
                ee_vel=0.5,
                arm_effort=0.01,
                base_attitude=0.2,
                base_angvel=0.1,
                fuel_impulse=0.0,
                collision=5.0,
                success_bonus=0.5,
            )
        raise ValueError(
            f"Invalid mission '{mission}'. Expected one of {VALID_MISSION_MODES}."
        )


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for :class:`SpaceManipulatorEnv`."""

    mission: MissionMode = "reach"
    action_mode: ActionMode = "torque"
    model_path: str | Path | None = None
    frame_skip: int = 10
    max_episode_steps: int = 250
    success_threshold: float = 0.03
    hold_steps: int = 5
    terminate_on_success: bool | None = None
    residual_scale: float = 0.3
    randomize_init: bool = True
    init_base_angvel_std: float = 0.0
    reward_weights: RewardWeights | None = None
    render_mode: str | None = None
    render_width: int = 640
    render_height: int = 480
    camera_name: str | None = None


class SpaceManipulatorEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Modular Gymnasium environment for 3-DoF orbital manipulation.

    Missions:
        - ``reach``: 3-DoF free-floating arm reaching a static 3D target.
        - ``low_disturbance``: Free-floating reach with strong base attitude and
          angular-velocity penalties.
        - ``fuel_optimal``: 9-DoF base thrusters + arm joints with explicit base
          fuel-impulse penalty. In ``residual_gjm`` mode, residuals are added to
          the ``free-flying`` baseline controller.
        - ``moving_target``: Free-floating tracking of a smoothly orbiting
          target with continuous position/velocity rewards and no early success
          termination.

    Observation space (28-D ``float32``):
        ``[arm_qpos(3), arm_qvel(3), base_quat(4, w>=0), base_angvel_body(3),
        base_linvel_world(3), ee_pos_rel_base(3), ee_to_target_world(3),
        ee_vel_world(3), target_vel_world(3)]``

    Action space (``float32`` in ``[-1, 1]``):
        - ``shape=(3,)`` for ``reach``, ``low_disturbance``, ``moving_target``
          (arm torques; base thrusters zeroed).
        - ``shape=(9,)`` for ``fuel_optimal`` (6 base wrenches + 3 arm torques).
    """

    metadata: dict[str, Any] = {
        "render_modes": list(VALID_RENDER_MODES),
        "render_fps": 50,
    }

    def __init__(
        self,
        config: EnvConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        base_cfg = config if config is not None else EnvConfig()
        if kwargs:
            base_cfg = replace(base_cfg, **kwargs)

        if base_cfg.mission not in VALID_MISSION_MODES:
            raise ValueError(
                f"Invalid mission '{base_cfg.mission}'. Expected one of {VALID_MISSION_MODES}."
            )
        if base_cfg.action_mode not in VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode '{base_cfg.action_mode}'. "
                f"Expected one of {VALID_ACTION_MODES}."
            )
        if base_cfg.render_mode is not None and base_cfg.render_mode not in VALID_RENDER_MODES:
            raise ValueError(
                f"Invalid render_mode '{base_cfg.render_mode}'. "
                f"Expected None or one of {VALID_RENDER_MODES}."
            )
        if base_cfg.frame_skip < 1:
            raise ValueError("frame_skip must be >= 1.")
        if base_cfg.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1.")
        if base_cfg.hold_steps < 1:
            raise ValueError("hold_steps must be >= 1.")
        if base_cfg.residual_scale < 0.0:
            raise ValueError("residual_scale must be >= 0.0.")

        terminate_on_success = (
            base_cfg.terminate_on_success
            if base_cfg.terminate_on_success is not None
            else (base_cfg.mission != "moving_target")
        )
        reward_weights = (
            base_cfg.reward_weights
            if base_cfg.reward_weights is not None
            else RewardWeights.for_mission(base_cfg.mission)
        )
        self.config: EnvConfig = replace(
            base_cfg,
            terminate_on_success=terminate_on_success,
            reward_weights=reward_weights,
        )
        self.render_mode: str | None = self.config.render_mode
        self.weights: RewardWeights = reward_weights

        # Load MuJoCo model, main data, and auxiliary kinematics data
        self.model: mujoco.MjModel
        self.data: mujoco.MjData
        self.model, self.data = load_model(self.config.model_path)
        self._aux_data: mujoco.MjData = mujoco.MjData(self.model)
        self.indices: ModelIndices = get_model_indices(self.model)

        # Cache hot-path indices & actuator limits
        self._base_act_ids: list[int] = list(self.indices.base_actuator_ids)
        self._arm_act_ids: list[int] = list(self.indices.arm_actuator_ids)
        self._arm_qpos_idx: list[int] = list(self.indices.arm_qpos_adrs)
        self._arm_qvel_idx: list[int] = list(self.indices.arm_qvel_adrs)
        self._base_qpos_adr: int = self.indices.base_qpos_adr
        self._base_qvel_adr: int = self.indices.base_qvel_adr
        self._base_site_id: int = self.indices.site_ids["base_imu_site"]
        self._ee_site_id: int = self.indices.site_ids["ee_site"]
        self._target_site_id: int = self.indices.site_ids["target_site"]
        self._ee_linvel_adr, self._ee_linvel_dim = self.indices.sensor_slices["ee_linvel"]

        self._base_ctrl_low: NDArray[np.float64] = self.model.actuator_ctrlrange[
            self._base_act_ids, 0
        ].copy()
        self._base_ctrl_high: NDArray[np.float64] = self.model.actuator_ctrlrange[
            self._base_act_ids, 1
        ].copy()
        self._base_ctrl_max: NDArray[np.float64] = np.maximum(
            np.abs(self._base_ctrl_low), np.abs(self._base_ctrl_high)
        )
        self._arm_ctrl_low: NDArray[np.float64] = self.model.actuator_ctrlrange[
            self._arm_act_ids, 0
        ].copy()
        self._arm_ctrl_high: NDArray[np.float64] = self.model.actuator_ctrlrange[
            self._arm_act_ids, 1
        ].copy()
        self._arm_ctrl_max: NDArray[np.float64] = np.maximum(
            np.abs(self._arm_ctrl_low), np.abs(self._arm_ctrl_high)
        )

        # Arm joint limits for safe reachable target sampling
        self._arm_jnt_low: NDArray[np.float64] = np.array(
            [self.model.jnt_range[jid, 0] for jid in self.indices.arm_joint_ids],
            dtype=np.float64,
        )
        self._arm_jnt_high: NDArray[np.float64] = np.array(
            [self.model.jnt_range[jid, 1] for jid in self.indices.arm_joint_ids],
            dtype=np.float64,
        )

        self._env_dt: float = float(self.model.opt.timestep) * self.config.frame_skip
        self._has_base_actuation: bool = self.config.mission == "fuel_optimal"

        # Baseline controller for residual_gjm mode
        self._controller: SpaceManipulatorController | None = None
        if self.config.action_mode == "residual_gjm":
            ctrl_mode: Literal["free-floating", "free-flying"] = (
                "free-flying" if self._has_base_actuation else "free-floating"
            )
            self._controller = SpaceManipulatorController(self.model, mode=ctrl_mode)

        # Spaces
        act_dim = 9 if self._has_base_actuation else 3
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Episode state
        self._step_count: int = 0
        self._success_streak: int = 0
        self._prev_ee_error: float = 0.0
        self._cumulative_fuel_impulse: float = 0.0
        self._base_ref_pos: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._base_ref_quat: NDArray[np.float64] = _IDENTITY_QUAT.copy()
        self._target_center: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._target_vel: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        self._traj_amp: NDArray[np.float64] = np.array([0.08, 0.12, 0.08], dtype=np.float64)
        self._traj_omega: float = 0.8
        self._traj_phase: float = 0.0
        self._traj_rot_omega: float = 0.4

        # Lazy renderer
        self._renderer: mujoco.Renderer | None = None

    def _sample_arm_qpos(self, margin_frac: float = 0.15) -> NDArray[np.float64]:
        """Sample a collision-free arm joint configuration within a safe margin."""
        span = self._arm_jnt_high - self._arm_jnt_low
        low = self._arm_jnt_low + margin_frac * span
        high = self._arm_jnt_high - margin_frac * span
        return self.np_random.uniform(low, high).astype(np.float64)

    def _sample_reachable_target(
        self,
        base_pos: NDArray[np.float64],
        base_quat: NDArray[np.float64],
        initial_ee_pos: NDArray[np.float64],
        min_dist: float = 0.18,
    ) -> NDArray[np.float64]:
        """Sample a physically reachable 3D target via forward kinematics on aux data.

        Samples interior joint configurations (elbow bent forward, away from
        joint limits and full-extension singularities) so targets remain
        reachable even under free-floating base recoil.
        """
        aux = self._aux_data
        best_target: NDArray[np.float64] | None = None
        best_dist = -1.0

        for _ in range(24):
            q1 = float(self.np_random.uniform(-0.45, 0.45))
            q2 = float(self.np_random.uniform(-0.50, 0.50))
            sign = -1.0 if (q2 > 0.05 or (abs(q2) <= 0.05 and self.np_random.random() < 0.5)) else 1.0
            q3 = sign * float(self.np_random.uniform(0.45, 0.95))
            q_goal = np.array([q1, q2, q3], dtype=np.float64)
            aux.qpos[self._base_qpos_adr : self._base_qpos_adr + 3] = base_pos
            aux.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7] = base_quat
            for adr, val in zip(self._arm_qpos_idx, q_goal):
                aux.qpos[adr] = val
            mujoco.mj_kinematics(self.model, aux)
            candidate = aux.site_xpos[self._ee_site_id].copy()
            dist = float(np.linalg.norm(candidate - initial_ee_pos))
            if dist >= min_dist:
                return candidate
            if dist > best_dist:
                best_dist = dist
                best_target = candidate

        assert best_target is not None
        return best_target

    def _eval_moving_target(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate analytical position, quaternion, and linear velocity at time ``t``."""
        w = self._traj_omega
        phi = self._traj_phase
        ax, ay, az = self._traj_amp
        pos = self._target_center + np.array(
            [
                ax * math.cos(w * t + phi),
                ay * math.sin(w * t + phi),
                az * math.sin(0.5 * w * t + phi),
            ],
            dtype=np.float64,
        )
        vel = np.array(
            [
                -ax * w * math.sin(w * t + phi),
                ay * w * math.cos(w * t + phi),
                0.5 * az * w * math.cos(0.5 * w * t + phi),
            ],
            dtype=np.float64,
        )
        half_theta = 0.5 * self._traj_rot_omega * t
        quat = np.array(
            [math.cos(half_theta), 0.0, 0.0, math.sin(half_theta)],
            dtype=np.float64,
        )
        return pos, quat, vel

    def _get_obs(self) -> NDArray[np.float32]:
        """Build the 28-D compact float32 observation vector."""
        d = self.data
        arm_qpos = np.array([d.qpos[adr] for adr in self._arm_qpos_idx], dtype=np.float64)
        arm_qvel = np.array([d.qvel[adr] for adr in self._arm_qvel_idx], dtype=np.float64)

        base_quat = d.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7].copy()
        if base_quat[0] < 0.0:
            base_quat = -base_quat

        base_linvel = d.qvel[self._base_qvel_adr : self._base_qvel_adr + 3]
        base_angvel = d.qvel[self._base_qvel_adr + 3 : self._base_qvel_adr + 6]

        base_pos = d.site_xpos[self._base_site_id]
        base_rot = d.site_xmat[self._base_site_id].reshape(3, 3)
        ee_pos = d.site_xpos[self._ee_site_id]
        target_pos = d.site_xpos[self._target_site_id]

        ee_pos_rel_base = base_rot.T @ (ee_pos - base_pos)
        ee_to_target = target_pos - ee_pos
        ee_vel = d.sensordata[self._ee_linvel_adr : self._ee_linvel_adr + self._ee_linvel_dim]

        obs = np.concatenate(
            [
                arm_qpos,
                arm_qvel,
                base_quat,
                base_angvel,
                base_linvel,
                ee_pos_rel_base,
                ee_to_target,
                ee_vel,
                self._target_vel,
            ]
        )
        return obs.astype(np.float32)

    def _build_info(
        self,
        *,
        fuel_impulse_step: float,
        arm_effort: float,
        reward_terms: dict[str, float],
    ) -> dict[str, Any]:
        """Construct rich telemetry info dict for the current state."""
        d = self.data
        ee_pos = d.site_xpos[self._ee_site_id]
        target_pos = d.site_xpos[self._target_site_id]
        ee_error = float(np.linalg.norm(target_pos - ee_pos))
        ee_vel = d.sensordata[self._ee_linvel_adr : self._ee_linvel_adr + self._ee_linvel_dim]
        ee_speed = float(np.linalg.norm(ee_vel))

        base_quat = d.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7]
        att_err = orientation_error(base_quat, self._base_ref_quat, frame="body")
        base_attitude_rad = float(np.linalg.norm(att_err))
        base_attitude_deg = float(math.degrees(base_attitude_rad))

        base_angvel = d.qvel[self._base_qvel_adr + 3 : self._base_qvel_adr + 6]
        base_angvel_norm = float(np.linalg.norm(base_angvel))

        return {
            "ee_error": ee_error,
            "ee_speed": ee_speed,
            "base_attitude_deg": base_attitude_deg,
            "base_angvel_norm": base_angvel_norm,
            "fuel_impulse_step": float(fuel_impulse_step),
            "cumulative_fuel_impulse": float(self._cumulative_fuel_impulse),
            "arm_effort": float(arm_effort),
            "ncon": int(d.ncon),
            "is_success": bool(self._success_streak >= self.config.hold_steps),
            "reward_terms": reward_terms,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)

        if self.config.randomize_init:
            base_pos = self.np_random.uniform(-0.05, 0.05, size=3)
            axis = self.np_random.normal(size=3)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1e-8:
                axis = axis / axis_norm
                angle = float(self.np_random.uniform(-0.12, 0.12))
                half = 0.5 * angle
                base_quat = np.array(
                    [math.cos(half), *(math.sin(half) * axis)], dtype=np.float64
                )
            else:
                base_quat = _IDENTITY_QUAT.copy()
            arm_qpos = self._sample_arm_qpos(margin_frac=0.20)
        else:
            base_pos = np.zeros(3, dtype=np.float64)
            base_quat = _IDENTITY_QUAT.copy()
            arm_qpos = np.array([0.0, 0.25, -0.45], dtype=np.float64)

        if self.config.init_base_angvel_std > 0.0:
            base_angvel = self.np_random.normal(
                0.0, self.config.init_base_angvel_std, size=3
            )
        else:
            base_angvel = np.zeros(3, dtype=np.float64)

        self._base_ref_pos = base_pos.copy()
        self._base_ref_quat = base_quat.copy()

        # First reset state so we know initial ee_pos for minimum target distance
        reset_state(
            self.model,
            self.data,
            base_pos=base_pos,
            base_quat=base_quat,
            arm_qpos=arm_qpos,
            base_angvel=base_angvel,
            indices=self.indices,
        )
        initial_ee_pos = self.data.site_xpos[self._ee_site_id].copy()

        target_pos = self._sample_reachable_target(
            base_pos=base_pos,
            base_quat=base_quat,
            initial_ee_pos=initial_ee_pos,
        )
        self._target_center = target_pos.copy()
        target_quat = _IDENTITY_QUAT.copy()

        if self.config.mission == "moving_target":
            self._traj_phase = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            target_pos, target_quat, self._target_vel = self._eval_moving_target(0.0)
        else:
            self._target_vel = np.zeros(3, dtype=np.float64)

        set_target_pose(
            self.model,
            self.data,
            target_pos,
            target_quat,
            forward=True,
            indices=self.indices,
        )

        self._step_count = 0
        self._success_streak = 0
        self._cumulative_fuel_impulse = 0.0
        self._prev_ee_error = float(
            np.linalg.norm(
                self.data.site_xpos[self._target_site_id]
                - self.data.site_xpos[self._ee_site_id]
            )
        )

        obs = self._get_obs()
        info = self._build_info(
            fuel_impulse_step=0.0,
            arm_effort=0.0,
            reward_terms={"total": 0.0},
        )
        return obs, info

    def _compute_ctrl_substep(
        self,
        clipped_action: NDArray[np.float64],
        *,
        forward: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute ``(u_base, u_arm)`` for one physics substep."""
        if self.config.action_mode == "torque":
            if self._has_base_actuation:
                u_base = np.clip(
                    clipped_action[:6] * self._base_ctrl_max,
                    self._base_ctrl_low,
                    self._base_ctrl_high,
                )
                u_arm = np.clip(
                    clipped_action[6:9] * self._arm_ctrl_max,
                    self._arm_ctrl_low,
                    self._arm_ctrl_high,
                )
            else:
                u_base = np.zeros(6, dtype=np.float64)
                u_arm = np.clip(
                    clipped_action * self._arm_ctrl_max,
                    self._arm_ctrl_low,
                    self._arm_ctrl_high,
                )
            return u_base, u_arm

        # residual_gjm mode
        assert self._controller is not None
        ctrl_out = self._controller.compute_control(
            self.data,
            base_ref_pos=self._base_ref_pos,
            base_ref_quat=self._base_ref_quat,
            forward=forward,
        )
        scale = self.config.residual_scale
        if self._has_base_actuation:
            u_base = np.clip(
                ctrl_out.u_base + clipped_action[:6] * scale * self._base_ctrl_max,
                self._base_ctrl_low,
                self._base_ctrl_high,
            )
            u_arm = np.clip(
                ctrl_out.u_arm + clipped_action[6:9] * scale * self._arm_ctrl_max,
                self._arm_ctrl_low,
                self._arm_ctrl_high,
            )
        else:
            u_base = np.zeros(6, dtype=np.float64)
            u_arm = np.clip(
                ctrl_out.u_arm + clipped_action * scale * self._arm_ctrl_max,
                self._arm_ctrl_low,
                self._arm_ctrl_high,
            )
        return u_base, u_arm

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        act_arr = np.asarray(action, dtype=np.float64)
        if act_arr.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {act_arr.shape}."
            )
        clipped_action = np.clip(act_arr, -1.0, 1.0)

        substep_dt = float(self.model.opt.timestep)
        fuel_impulse_step = 0.0
        arm_effort_accum = 0.0

        # In torque mode, u_base and u_arm are constant across frame_skip substeps.
        # In residual_gjm mode, substep 0 reuses the mj_forward from reset/previous step.
        if self.config.action_mode == "torque":
            u_base, u_arm = self._compute_ctrl_substep(clipped_action, forward=False)
            self.data.ctrl[self._base_act_ids] = u_base
            self.data.ctrl[self._arm_act_ids] = u_arm
            fuel_impulse_step = (
                float(np.linalg.norm(u_base[:3]) + np.linalg.norm(u_base[3:6]))
                * self._env_dt
            )
            arm_effort_accum = float(np.sum((u_arm / self._arm_ctrl_max) ** 2))
            for _ in range(self.config.frame_skip):
                mujoco.mj_step(self.model, self.data)
        else:
            for i in range(self.config.frame_skip):
                u_base, u_arm = self._compute_ctrl_substep(
                    clipped_action, forward=(i > 0)
                )
                self.data.ctrl[self._base_act_ids] = u_base
                self.data.ctrl[self._arm_act_ids] = u_arm
                fuel_impulse_step += (
                    float(np.linalg.norm(u_base[:3]) + np.linalg.norm(u_base[3:6]))
                    * substep_dt
                )
                arm_effort_accum += float(np.sum((u_arm / self._arm_ctrl_max) ** 2))
                mujoco.mj_step(self.model, self.data)
            arm_effort_accum /= float(self.config.frame_skip)

        # Advance moving target after physics step so target matches new sim time
        if self.config.mission == "moving_target":
            t_now = float(self.data.time)
            t_pos, t_quat, self._target_vel = self._eval_moving_target(t_now)
            set_target_pose(
                self.model,
                self.data,
                t_pos,
                t_quat,
                forward=False,
                indices=self.indices,
            )

        # Single mj_forward after frame_skip so sensors, site poses, ncon, and
        # next step's forward=False controller are all synchronized.
        mujoco.mj_forward(self.model, self.data)

        self._step_count += 1
        self._cumulative_fuel_impulse += fuel_impulse_step

        # Compute reward & termination
        d = self.data
        ee_pos = d.site_xpos[self._ee_site_id]
        target_pos = d.site_xpos[self._target_site_id]
        ee_error = float(np.linalg.norm(target_pos - ee_pos))
        progress = self._prev_ee_error - ee_error
        self._prev_ee_error = ee_error

        ee_vel = d.sensordata[self._ee_linvel_adr : self._ee_linvel_adr + self._ee_linvel_dim]
        ee_vel_err = float(np.linalg.norm(ee_vel - self._target_vel))

        base_quat = d.qpos[self._base_qpos_adr + 3 : self._base_qpos_adr + 7]
        base_att_rad = float(
            np.linalg.norm(orientation_error(base_quat, self._base_ref_quat, frame="body"))
        )
        base_angvel_norm = float(
            np.linalg.norm(d.qvel[self._base_qvel_adr + 3 : self._base_qvel_adr + 6])
        )

        if ee_error <= self.config.success_threshold:
            self._success_streak += 1
        else:
            self._success_streak = 0

        is_success = self._success_streak >= self.config.hold_steps
        terminated = bool(self.config.terminate_on_success and is_success)
        truncated = bool(self._step_count >= self.config.max_episode_steps and not terminated)

        w = self.weights
        term_distance = -w.distance * ee_error
        term_progress = w.progress * progress
        term_ee_vel = -w.ee_vel * ee_vel_err
        term_arm_effort = -w.arm_effort * arm_effort_accum
        term_base_att = -w.base_attitude * base_att_rad
        term_base_angvel = -w.base_angvel * base_angvel_norm
        term_fuel = -w.fuel_impulse * fuel_impulse_step
        term_collision = -w.collision if d.ncon > 0 else 0.0

        if self.config.mission == "moving_target":
            term_success = w.success_bonus if ee_error <= self.config.success_threshold else 0.0
        else:
            term_success = w.success_bonus if is_success else 0.0

        reward = float(
            term_distance
            + term_progress
            + term_ee_vel
            + term_arm_effort
            + term_base_att
            + term_base_angvel
            + term_fuel
            + term_collision
            + term_success
        )

        reward_terms = {
            "distance": term_distance,
            "progress": term_progress,
            "ee_vel": term_ee_vel,
            "arm_effort": term_arm_effort,
            "base_attitude": term_base_att,
            "base_angvel": term_base_angvel,
            "fuel_impulse": term_fuel,
            "collision": term_collision,
            "success_bonus": term_success,
            "total": reward,
        }

        obs = self._get_obs()
        info = self._build_info(
            fuel_impulse_step=fuel_impulse_step,
            arm_effort=arm_effort_accum,
            reward_terms=reward_terms,
        )
        return obs, reward, terminated, truncated, info

    def render(self) -> NDArray[np.uint8] | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.render_height,
                width=self.config.render_width,
            )
        if self.config.camera_name is not None:
            self._renderer.update_scene(self.data, camera=self.config.camera_name)
        else:
            self._renderer.update_scene(self.data)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_config_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the environment configuration."""
        d = asdict(self.config)
        if d.get("model_path") is not None:
            d["model_path"] = str(d["model_path"])
        return d
