"""Tests for MJCF loading, state management, and microgravity dynamics."""

from __future__ import annotations

from pathlib import Path
import warnings

import mujoco
import numpy as np

from space_manipulator import (
    ARM_ACTUATOR_NAMES,
    ARM_JOINT_NAMES,
    BASE_ACTUATOR_NAMES,
    SENSOR_NAMES,
    SITE_NAMES,
    get_model_indices,
    get_site_pose,
    get_state_summary,
    load_model,
    read_all_sensors,
    read_sensor,
    reset_state,
    set_target_pose,
)
from space_manipulator.mesh_generator import (
    DEFAULT_MESH_DIR,
    MESH_FILENAMES,
    generate_all_meshes,
)


def _spacecraft_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Compute center of mass of the dynamic spacecraft (excluding mocap target)."""
    dynamic_bodies = [
        model.body(name).id for name in ("base_sat", "link1", "link2", "link3")
    ]
    masses = np.array([model.body_mass[bid] for bid in dynamic_bodies])
    positions = np.array([data.xipos[bid] for bid in dynamic_bodies])
    return np.sum(masses[:, None] * positions, axis=0) / np.sum(masses)


def test_load_model_options_and_dimensions() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, data = load_model()

    assert len(caught) == 0
    assert np.allclose(model.opt.gravity, [0.0, 0.0, 0.0])
    assert np.isclose(model.opt.timestep, 0.002)
    assert model.opt.integrator == int(mujoco.mjtIntegrator.mjINT_RK4)
    assert model.nq == 10
    assert model.nv == 9
    assert model.nu == 9
    assert model.nsensor == len(SENSOR_NAMES)
    assert data.ncon == 0


def test_model_indices_and_sensors() -> None:
    model, data = load_model()
    indices = get_model_indices(model)

    assert len(indices.arm_joint_ids) == len(ARM_JOINT_NAMES) == 3
    assert len(indices.base_actuator_ids) == len(BASE_ACTUATOR_NAMES) == 6
    assert len(indices.arm_actuator_ids) == len(ARM_ACTUATOR_NAMES) == 3
    assert set(indices.site_ids.keys()) == set(SITE_NAMES)
    assert set(indices.sensor_slices.keys()) == set(SENSOR_NAMES)
    assert indices.target_mocap_id >= 0

    sensors = read_all_sensors(model, data)
    assert set(sensors.keys()) == set(SENSOR_NAMES)
    assert sensors["base_pos"].shape == (3,)
    assert sensors["base_quat"].shape == (4,)
    assert sensors["ee_pos"].shape == (3,)
    assert sensors["joint1_pos"].shape == (1,)
    assert np.allclose(sensors["base_pos"], [0.0, 0.0, 0.0])
    assert np.allclose(sensors["ee_pos"], read_sensor(model, data, "ee_pos"))


def test_reset_state_and_target_pose() -> None:
    model, data = load_model()
    reset_state(
        model,
        data,
        base_pos=[0.5, -0.2, 0.1],
        arm_qpos=[0.3, -0.4, 0.5],
        target_pos=[1.2, 0.4, -0.1],
    )

    summary = get_state_summary(model, data)
    assert np.allclose(summary["base_pos"], [0.5, -0.2, 0.1])
    assert np.allclose(summary["arm_qpos"], [0.3, -0.4, 0.5])
    assert np.allclose(summary["target_pos"], [1.2, 0.4, -0.1])

    set_target_pose(model, data, [1.4, -0.3, 0.25])
    target_pos, target_rot = get_site_pose(model, data, "target_site")
    assert np.allclose(target_pos, [1.4, -0.3, 0.25])
    assert target_rot.shape == (3, 3)


def test_free_floating_momentum_conservation() -> None:
    model, data = load_model()
    indices = get_model_indices(model)
    reset_state(model, data)

    initial_com = _spacecraft_com(model, data)

    # Zero base actuators (free-floating), apply torque on arm joints
    data.ctrl[:] = 0.0
    data.ctrl[indices.arm_actuator_ids[0]] = 5.0
    data.ctrl[indices.arm_actuator_ids[1]] = -3.0

    for _ in range(250):
        mujoco.mj_step(model, data)

    final_com = _spacecraft_com(model, data)
    base_pos = read_sensor(model, data, "base_pos")
    base_angvel = read_sensor(model, data, "base_angvel")

    # Total spacecraft COM stays fixed in 0g with zero external forces
    assert np.allclose(final_com, initial_com, atol=1e-6)
    # Base reacts to arm motion (base translation & rotation occur)
    assert np.linalg.norm(base_pos) > 1e-4
    assert np.linalg.norm(base_angvel) > 1e-3


def test_free_flying_base_thruster_actuation() -> None:
    model, data = load_model()
    indices = get_model_indices(model)
    reset_state(model, data)

    # Apply +X thruster force on base_imu_site
    data.ctrl[:] = 0.0
    data.ctrl[indices.base_actuator_ids[0]] = 20.0

    for _ in range(250):
        mujoco.mj_step(model, data)

    base_pos = read_sensor(model, data, "base_pos")
    base_linvel = read_sensor(model, data, "base_linvel")
    assert base_pos[0] > 0.04
    assert base_linvel[0] > 0.15


def test_cad_stl_meshes_loaded_in_mjmodel(tmp_path: Path) -> None:
    model, _ = load_model()

    assert model.nmesh >= 10
    assert model.nmesh == len(MESH_FILENAMES)

    for filename in MESH_FILENAMES:
        stl_path = DEFAULT_MESH_DIR / filename
        assert stl_path.is_file()
        assert stl_path.stat().st_size > 84

        mesh_name = stl_path.stem
        mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
        assert mesh_id >= 0
        assert model.mesh_facenum[mesh_id] > 0

        geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, f"vis_{mesh_name}"
        )
        assert geom_id >= 0
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0

    generated = generate_all_meshes(tmp_path)
    assert set(generated.keys()) == set(MESH_FILENAMES)
    for stats in generated.values():
        assert stats.path.is_file()
        assert stats.triangle_count > 0

