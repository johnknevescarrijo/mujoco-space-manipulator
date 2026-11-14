"""Utilities for loading, resetting, and inspecting the space manipulator MJCF model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_MODEL_PATH: Path = (
    Path(__file__).resolve().parents[2] / "models" / "space_manipulator.xml"
)

BASE_FREEJOINT_NAME: str = "base_freejoint"
TARGET_BODY_NAME: str = "target_satellite"

ARM_JOINT_NAMES: tuple[str, ...] = ("joint1", "joint2", "joint3")
BASE_ACTUATOR_NAMES: tuple[str, ...] = (
    "base_fx",
    "base_fy",
    "base_fz",
    "base_tx",
    "base_ty",
    "base_tz",
)
ARM_ACTUATOR_NAMES: tuple[str, ...] = (
    "act_joint1",
    "act_joint2",
    "act_joint3",
)
SITE_NAMES: tuple[str, ...] = ("base_imu_site", "ee_site", "target_site")
SENSOR_NAMES: tuple[str, ...] = (
    "base_pos",
    "base_quat",
    "base_linvel",
    "base_angvel",
    "ee_pos",
    "ee_quat",
    "ee_linvel",
    "ee_angvel",
    "joint1_pos",
    "joint2_pos",
    "joint3_pos",
    "joint1_vel",
    "joint2_vel",
    "joint3_vel",
)


@dataclass(frozen=True)
class ModelIndices:
    """Resolved MuJoCo IDs and state addresses for the space manipulator model."""

    base_freejoint_id: int
    base_qpos_adr: int
    base_qvel_adr: int
    arm_joint_ids: tuple[int, ...]
    arm_qpos_adrs: tuple[int, ...]
    arm_qvel_adrs: tuple[int, ...]
    base_actuator_ids: tuple[int, ...]
    arm_actuator_ids: tuple[int, ...]
    site_ids: dict[str, int]
    sensor_slices: dict[str, tuple[int, int]]
    target_mocap_id: int


def _require_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    """Return a MuJoCo object ID or raise ValueError if missing."""
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Required {obj_type.name} '{name}' not found in MJCF model.")
    return int(obj_id)


def load_model(
    model_path: Path | str | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the space manipulator MJCF model and initialize MjData.

    Args:
        model_path: Optional path to an MJCF XML file. Defaults to
            ``models/space_manipulator.xml``.

    Returns:
        Tuple of ``(MjModel, MjData)`` after an initial ``mj_forward`` pass.
    """
    resolved_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not resolved_path.is_file():
        raise FileNotFoundError(f"MJCF model file not found: {resolved_path}")

    model = mujoco.MjModel.from_xml_path(str(resolved_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def get_model_indices(model: mujoco.MjModel) -> ModelIndices:
    """Inspect and return validated joint, actuator, site, and sensor indices."""
    base_jnt_id = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, BASE_FREEJOINT_NAME)
    base_qpos_adr = int(model.jnt_qposadr[base_jnt_id])
    base_qvel_adr = int(model.jnt_dofadr[base_jnt_id])

    arm_joint_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES
    )
    arm_qpos_adrs = tuple(int(model.jnt_qposadr[jid]) for jid in arm_joint_ids)
    arm_qvel_adrs = tuple(int(model.jnt_dofadr[jid]) for jid in arm_joint_ids)

    base_actuator_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in BASE_ACTUATOR_NAMES
    )
    arm_actuator_ids = tuple(
        _require_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ARM_ACTUATOR_NAMES
    )

    site_ids = {
        name: _require_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in SITE_NAMES
    }

    sensor_slices: dict[str, tuple[int, int]] = {}
    for name in SENSOR_NAMES:
        sid = _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = int(model.sensor_adr[sid])
        dim = int(model.sensor_dim[sid])
        sensor_slices[name] = (adr, dim)

    target_body_id = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY_NAME)
    target_mocap_id = int(model.body_mocapid[target_body_id])
    if target_mocap_id < 0:
        raise ValueError(
            f"Body '{TARGET_BODY_NAME}' is expected to be a mocap body (mocap='true')."
        )

    return ModelIndices(
        base_freejoint_id=base_jnt_id,
        base_qpos_adr=base_qpos_adr,
        base_qvel_adr=base_qvel_adr,
        arm_joint_ids=arm_joint_ids,
        arm_qpos_adrs=arm_qpos_adrs,
        arm_qvel_adrs=arm_qvel_adrs,
        base_actuator_ids=base_actuator_ids,
        arm_actuator_ids=arm_actuator_ids,
        site_ids=site_ids,
        sensor_slices=sensor_slices,
        target_mocap_id=target_mocap_id,
    )


def set_target_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: ArrayLike,
    quat: ArrayLike | None = None,
    *,
    forward: bool = True,
) -> None:
    """Update the 3D pose of the mocap target satellite."""
    indices = get_model_indices(model)
    pos_arr = np.asarray(pos, dtype=float)
    if pos_arr.shape != (3,):
        raise ValueError(f"Expected target pos shape (3,), got {pos_arr.shape}.")
    data.mocap_pos[indices.target_mocap_id] = pos_arr

    if quat is not None:
        quat_arr = np.asarray(quat, dtype=float)
        if quat_arr.shape != (4,):
            raise ValueError(
                f"Expected target quat shape (4,), got {quat_arr.shape}."
            )
        norm = float(np.linalg.norm(quat_arr))
        if norm == 0.0:
            raise ValueError("Target quaternion cannot have zero norm.")
        data.mocap_quat[indices.target_mocap_id] = quat_arr / norm

    if forward:
        mujoco.mj_forward(model, data)


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    base_pos: ArrayLike | None = None,
    base_quat: ArrayLike | None = None,
    arm_qpos: ArrayLike | None = None,
    target_pos: ArrayLike | None = None,
    target_quat: ArrayLike | None = None,
) -> None:
    """Reset simulation state, optionally setting base, arm, and target poses."""
    mujoco.mj_resetData(model, data)
    indices = get_model_indices(model)

    if base_pos is not None:
        pos_arr = np.asarray(base_pos, dtype=float)
        if pos_arr.shape != (3,):
            raise ValueError(f"Expected base_pos shape (3,), got {pos_arr.shape}.")
        data.qpos[indices.base_qpos_adr : indices.base_qpos_adr + 3] = pos_arr

    if base_quat is not None:
        quat_arr = np.asarray(base_quat, dtype=float)
        if quat_arr.shape != (4,):
            raise ValueError(
                f"Expected base_quat shape (4,), got {quat_arr.shape}."
            )
        norm = float(np.linalg.norm(quat_arr))
        if norm == 0.0:
            raise ValueError("Base quaternion cannot have zero norm.")
        data.qpos[indices.base_qpos_adr + 3 : indices.base_qpos_adr + 7] = (
            quat_arr / norm
        )

    if arm_qpos is not None:
        arm_arr = np.asarray(arm_qpos, dtype=float)
        if arm_arr.shape != (len(ARM_JOINT_NAMES),):
            raise ValueError(
                f"Expected arm_qpos shape ({len(ARM_JOINT_NAMES)},), got {arm_arr.shape}."
            )
        for adr, val in zip(indices.arm_qpos_adrs, arm_arr):
            data.qpos[adr] = val

    if target_pos is not None or target_quat is not None:
        current_pos = (
            target_pos
            if target_pos is not None
            else data.mocap_pos[indices.target_mocap_id].copy()
        )
        set_target_pose(model, data, current_pos, target_quat, forward=False)

    mujoco.mj_forward(model, data)


def read_sensor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sensor_name: str,
) -> NDArray[np.float64]:
    """Read a single named sensor from ``data.sensordata``."""
    sid = _require_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return data.sensordata[adr : adr + dim].copy()


def read_all_sensors(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, NDArray[np.float64]]:
    """Read all standard space manipulator sensors as a dictionary."""
    indices = get_model_indices(model)
    return {
        name: data.sensordata[adr : adr + dim].copy()
        for name, (adr, dim) in indices.sensor_slices.items()
    }


def get_site_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return world position ``(3,)`` and rotation matrix ``(3, 3)`` for a site."""
    sid = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    pos = data.site_xpos[sid].copy()
    rotmat = data.site_xmat[sid].reshape(3, 3).copy()
    return pos, rotmat


def get_state_summary(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, Any]:
    """Return a structured summary of the current base, arm, and target state."""
    indices = get_model_indices(model)
    sensors = read_all_sensors(model, data)
    target_pos, _ = get_site_pose(model, data, "target_site")

    return {
        "time": float(data.time),
        "base_pos": sensors["base_pos"],
        "base_quat": sensors["base_quat"],
        "base_linvel": sensors["base_linvel"],
        "base_angvel": sensors["base_angvel"],
        "arm_qpos": np.array(
            [data.qpos[adr] for adr in indices.arm_qpos_adrs], dtype=float
        ),
        "arm_qvel": np.array(
            [data.qvel[adr] for adr in indices.arm_qvel_adrs], dtype=float
        ),
        "ee_pos": sensors["ee_pos"],
        "ee_linvel": sensors["ee_linvel"],
        "target_pos": target_pos,
    }
