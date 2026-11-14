"""Tests for GJM kinematics, quaternion errors, and free-floating/free-flying control."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from space_manipulator import (
    SpaceManipulatorController,
    compute_generalized_jacobian,
    compute_mass_matrix,
    compute_site_jacobian,
    compute_system_momentum,
    get_model_indices,
    get_site_pose,
    load_model,
    orientation_error,
    read_sensor,
    reset_state,
)


def test_quaternion_orientation_error() -> None:
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(orientation_error(q_id, q_id), np.zeros(3))
    # Antipodal quaternion represents the same orientation (shortest-path = 0)
    assert np.allclose(orientation_error(q_id, -q_id), np.zeros(3))

    # 90-degree rotation around +Z
    half_angle = np.pi / 4.0
    q_z90 = np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)])
    err_body = orientation_error(q_id, q_z90, frame="body")
    assert np.allclose(err_body, [0.0, 0.0, np.pi / 2.0], atol=1e-7)

    # Rotating from q_z90 back to identity requires -pi/2 around Z
    err_back = orientation_error(q_z90, q_id, frame="body")
    assert np.allclose(err_back, [0.0, 0.0, -np.pi / 2.0], atol=1e-7)


def test_jacobians_mass_matrix_and_gjm_identity() -> None:
    model, data = load_model()
    indices = get_model_indices(model)
    reset_state(model, data, arm_qpos=[0.35, -0.45, 0.55])

    j_b, j_m = compute_site_jacobian(model, data, "ee_site", indices=indices)
    assert j_b.shape == (3, 6)
    assert j_m.shape == (3, 3)

    mass = compute_mass_matrix(model, data, indices=indices)
    assert mass.full.shape == (9, 9)
    assert mass.h_b.shape == (6, 6)
    assert mass.h_bm.shape == (6, 3)
    assert mass.h_m.shape == (3, 3)
    assert np.allclose(mass.full, mass.full.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(mass.full) > 0.0)

    # Apply arm torques from rest with zero base thrust -> zero total momentum
    data.ctrl[:] = 0.0
    data.ctrl[list(indices.arm_actuator_ids)] = [4.0, -2.5, 1.5]
    for _ in range(120):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    gjm = compute_generalized_jacobian(model, data, "ee_site", indices=indices)
    assert gjm.j_star.shape == (3, 3)
    assert not np.allclose(gjm.j_star, gjm.j_m)

    qd_m = data.qvel[list(indices.arm_qvel_adrs)]
    ee_vel_gjm = gjm.j_star @ qd_m
    ee_vel_sensor = read_sensor(model, data, "ee_linvel")
    assert np.allclose(ee_vel_gjm, ee_vel_sensor, atol=1e-7)


def test_free_floating_controller_converges_and_conserves_momentum() -> None:
    model, data = load_model()
    target = np.array([1.20, -0.35, 0.25])
    reset_state(model, data, arm_qpos=[0.0, 0.3, -0.5], target_pos=target)

    controller = SpaceManipulatorController(model, mode="free-floating")

    for _ in range(2500):
        out = controller.step(data)
        assert np.allclose(out.u_base, np.zeros(6))

    mujoco.mj_forward(model, data)
    ee_pos, _ = get_site_pose(model, data, "ee_site")
    base_pos, _ = get_site_pose(model, data, "base_imu_site")
    lin_mom, ang_mom = compute_system_momentum(model, data)

    final_error = float(np.linalg.norm(target - ee_pos))
    assert final_error < 0.02
    assert np.linalg.norm(lin_mom) < 1e-6
    assert np.linalg.norm(ang_mom) < 1e-6
    # Free-floating base shifts naturally to conserve momentum
    assert np.linalg.norm(base_pos) > 1e-3


def test_free_flying_controller_converges_and_holds_base() -> None:
    model, data = load_model()
    target = np.array([1.20, -0.35, 0.25])
    base_ref_pos = np.array([0.0, 0.0, 0.0])
    base_ref_quat = np.array([1.0, 0.0, 0.0, 0.0])

    reset_state(
        model,
        data,
        base_pos=base_ref_pos,
        base_quat=base_ref_quat,
        arm_qpos=[0.0, 0.3, -0.5],
        target_pos=target,
    )

    controller = SpaceManipulatorController(model, mode="free-flying")

    for _ in range(2500):
        controller.step(
            data,
            base_ref_pos=base_ref_pos,
            base_ref_quat=base_ref_quat,
        )

    mujoco.mj_forward(model, data)
    ee_pos, _ = get_site_pose(model, data, "ee_site")
    base_pos, _ = get_site_pose(model, data, "base_imu_site")
    base_quat = read_sensor(model, data, "base_quat")

    ee_error = float(np.linalg.norm(target - ee_pos))
    base_pos_err = float(np.linalg.norm(base_ref_pos - base_pos))
    base_att_err = float(np.linalg.norm(orientation_error(base_quat, base_ref_quat)))

    assert ee_error < 0.02
    assert base_pos_err < 0.005
    assert base_att_err < 0.005


def test_free_flying_holds_rotated_base_reference() -> None:
    model, data = load_model()
    angle = 0.4
    base_ref_pos = np.array([0.1, -0.1, 0.05])
    base_ref_quat = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    target = np.array([1.25, 0.45, 0.15])

    reset_state(
        model,
        data,
        base_pos=base_ref_pos,
        base_quat=base_ref_quat,
        arm_qpos=[0.0, 0.2, -0.4],
        target_pos=target,
    )

    controller = SpaceManipulatorController(model, mode="free-flying")
    for _ in range(2500):
        controller.step(
            data,
            base_ref_pos=base_ref_pos,
            base_ref_quat=base_ref_quat,
        )

    mujoco.mj_forward(model, data)
    ee_pos, _ = get_site_pose(model, data, "ee_site")
    base_pos, _ = get_site_pose(model, data, "base_imu_site")
    base_quat = read_sensor(model, data, "base_quat")

    assert float(np.linalg.norm(target - ee_pos)) < 0.02
    assert float(np.linalg.norm(base_ref_pos - base_pos)) < 0.005
    assert float(np.linalg.norm(orientation_error(base_quat, base_ref_quat))) < 0.005


def test_controller_rejects_invalid_mode() -> None:
    model, _ = load_model()
    with pytest.raises(ValueError, match="Invalid control mode"):
        SpaceManipulatorController(model, mode="invalid-mode")  # type: ignore[arg-type]
