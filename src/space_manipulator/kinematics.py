"""Kinematics, dynamics partitioning, GJM, quaternion error, and momentum telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import ArrayLike, NDArray

from space_manipulator.model_loader import ModelIndices, get_model_indices


@dataclass(frozen=True)
class PartitionedMassMatrix:
    """Block-partitioned mass matrix for the base-manipulator system."""

    full: NDArray[np.float64]
    h_b: NDArray[np.float64]
    h_bm: NDArray[np.float64]
    h_m: NDArray[np.float64]


@dataclass(frozen=True)
class GeneralizedJacobian:
    """Base, manipulator, and Generalized Jacobian Matrix (GJM) components."""

    j_b: NDArray[np.float64]
    j_m: NDArray[np.float64]
    j_star: NDArray[np.float64]
    base_coupling: NDArray[np.float64]
    mass_matrix: PartitionedMassMatrix


def normalize_quaternion(quat: ArrayLike) -> NDArray[np.float64]:
    """Normalize a scalar-first quaternion ``[w, x, y, z]``."""
    q = np.asarray(quat, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion shape (4,), got {q.shape}.")
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("Quaternion cannot have zero norm.")
    return q / norm


def quaternion_conjugate(quat: ArrayLike) -> NDArray[np.float64]:
    """Return the conjugate (inverse for unit quaternions) of ``[w, x, y, z]``."""
    q = normalize_quaternion(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quaternion_multiply(q1: ArrayLike, q2: ArrayLike) -> NDArray[np.float64]:
    """Multiply two scalar-first quaternions ``q1 * q2``."""
    w1, x1, y1, z1 = normalize_quaternion(q1)
    w2, x2, y2, z2 = normalize_quaternion(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def orientation_error(
    current_quat: ArrayLike,
    desired_quat: ArrayLike,
    *,
    frame: str = "body",
) -> NDArray[np.float64]:
    """Compute 3D axis-angle attitude error rotating ``current_quat`` to ``desired_quat``.

    Args:
        current_quat: Current orientation quaternion ``[w, x, y, z]``.
        desired_quat: Desired orientation quaternion ``[w, x, y, z]``.
        frame: ``"body"`` expresses the error vector in the current body frame
            (matching MuJoCo freejoint angular velocity and site torques);
            ``"world"`` expresses it in the inertial world frame.

    Returns:
        3D rotation-vector error ``theta * u`` in radians.
    """
    q_cur = normalize_quaternion(current_quat)
    q_des = normalize_quaternion(desired_quat)

    if frame == "body":
        q_err = quaternion_multiply(quaternion_conjugate(q_cur), q_des)
    elif frame == "world":
        q_err = quaternion_multiply(q_des, quaternion_conjugate(q_cur))
    else:
        raise ValueError(f"Unsupported frame '{frame}'; expected 'body' or 'world'.")

    # Shortest-path rotation: keep scalar component non-negative
    if q_err[0] < 0.0:
        q_err = -q_err

    vec = q_err[1:4]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-12:
        return 2.0 * vec

    angle = 2.0 * float(np.arctan2(vec_norm, np.clip(q_err[0], -1.0, 1.0)))
    return (angle / vec_norm) * vec


def compute_site_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "ee_site",
    *,
    include_rotation: bool = False,
    indices: ModelIndices | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the MuJoCo site Jacobian partitioned into base ``J_b`` and arm ``J_m``.

    Args:
        model: Loaded MuJoCo model.
        data: Current MuJoCo data (position kinematics must be up to date).
        site_name: Name of the target site (default ``"ee_site"``).
        include_rotation: When False (default), returns translational Jacobians
            ``J_b`` ``(3, 6)`` and ``J_m`` ``(3, 3)``. When True, stacks linear and
            angular rows into ``(6, 6)`` and ``(6, 3)``.
        indices: Optional precomputed ``ModelIndices``.

    Returns:
        Tuple ``(J_b, J_m)``.
    """
    idx = indices if indices is not None else get_model_indices(model)
    site_id = idx.site_ids.get(site_name)
    if site_id is None:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MJCF model.")

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    base_cols = slice(idx.base_qvel_adr, idx.base_qvel_adr + 6)
    arm_cols = list(idx.arm_qvel_adrs)

    if include_rotation:
        j_full = np.vstack([jacp, jacr])
        return j_full[:, base_cols].copy(), j_full[:, arm_cols].copy()

    return jacp[:, base_cols].copy(), jacp[:, arm_cols].copy()


def compute_mass_matrix(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    indices: ModelIndices | None = None,
) -> PartitionedMassMatrix:
    """Compute the full ``(nv, nv)`` mass matrix via ``mj_fullM`` and partition it.

    Returns:
        ``PartitionedMassMatrix`` containing ``full`` ``(9, 9)``, ``h_b`` ``(6, 6)``,
        ``h_bm`` ``(6, 3)``, and ``h_m`` ``(3, 3)``.
    """
    idx = indices if indices is not None else get_model_indices(model)
    full = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, data, full)

    b_slice = slice(idx.base_qvel_adr, idx.base_qvel_adr + 6)
    m_idx = list(idx.arm_qvel_adrs)

    h_b = full[b_slice, b_slice].copy()
    h_bm = full[b_slice, m_idx].copy()
    h_m = full[np.ix_(m_idx, m_idx)].copy()
    return PartitionedMassMatrix(full=full, h_b=h_b, h_bm=h_bm, h_m=h_m)


def compute_generalized_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str = "ee_site",
    *,
    include_rotation: bool = False,
    indices: ModelIndices | None = None,
) -> GeneralizedJacobian:
    """Compute the Generalized Jacobian Matrix ``J* = J_m - J_b H_b^{-1} H_bm``.

    In free-floating mode with zero initial momentum, ``J*`` maps arm joint
    velocities ``qd_m`` directly to inertial end-effector velocities ``x_dot_ee``
    while accounting for dynamic base reaction.
    """
    idx = indices if indices is not None else get_model_indices(model)
    j_b, j_m = compute_site_jacobian(
        model, data, site_name=site_name, include_rotation=include_rotation, indices=idx
    )
    mass = compute_mass_matrix(model, data, indices=idx)

    # Solve H_b * X = H_bm without explicitly inverting H_b
    h_b_inv_h_bm = np.linalg.solve(mass.h_b, mass.h_bm)
    base_coupling = -h_b_inv_h_bm
    j_star = j_m + j_b @ base_coupling

    return GeneralizedJacobian(
        j_b=j_b,
        j_m=j_m,
        j_star=j_star,
        base_coupling=base_coupling,
        mass_matrix=mass,
    )


def damped_pseudoinverse(
    jacobian: ArrayLike,
    damping: float = 0.04,
) -> NDArray[np.float64]:
    """Compute the Damped Least Squares (DLS) pseudoinverse of a Jacobian."""
    if damping < 0.0:
        raise ValueError("Damping factor must be non-negative.")
    j = np.asarray(jacobian, dtype=float)
    rows = j.shape[0]
    gram = j @ j.T + (damping**2) * np.eye(rows, dtype=float)
    return j.T @ np.linalg.inv(gram)


def compute_system_momentum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_body_name: str = "base_sat",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute total linear momentum and COM angular momentum of the spacecraft.

    Excludes uncoupled mocap targets by querying MuJoCo's subtree velocity/momentum
    rooted at ``base_sat``.

    Returns:
        Tuple ``(linear_momentum, angular_momentum)`` in world frame.
    """
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_id < 0:
        raise ValueError(f"Body '{root_body_name}' not found in MJCF model.")

    mujoco.mj_subtreeVel(model, data)
    subtree_mass = float(model.body_subtreemass[root_id])
    linear_momentum = subtree_mass * data.subtree_linvel[root_id].copy()
    angular_momentum = data.subtree_angmom[root_id].copy()
    return linear_momentum, angular_momentum
