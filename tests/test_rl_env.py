"""Automated tests for SpaceManipulatorEnv and Gymnasium registration."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from space_manipulator.envs import (
    ENV_IDS,
    OBS_DIM,
    VALID_ACTION_MODES,
    VALID_MISSION_MODES,
    EnvConfig,
    SpaceManipulatorEnv,
    register_envs,
)
from space_manipulator.kinematics import compute_system_momentum


@pytest.mark.parametrize("mission", VALID_MISSION_MODES)
@pytest.mark.parametrize("action_mode", VALID_ACTION_MODES)
def test_gymnasium_check_env_all_missions_and_modes(
    mission: str, action_mode: str
) -> None:
    env = SpaceManipulatorEnv(
        EnvConfig(mission=mission, action_mode=action_mode, max_episode_steps=40)  # type: ignore[arg-type]
    )
    try:
        check_env(env, skip_render_check=True)
        assert env.observation_space.shape == (OBS_DIM,)
        expected_act_dim = 9 if mission == "fuel_optimal" else 3
        assert env.action_space.shape == (expected_act_dim,)
    finally:
        env.close()


@pytest.mark.parametrize("mission", ["reach", "moving_target"])
@pytest.mark.parametrize("action_mode", VALID_ACTION_MODES)
def test_seed_determinism_50_steps(mission: str, action_mode: str) -> None:
    def rollout(seed: int) -> tuple[list[np.ndarray], list[float], list[float]]:
        env = SpaceManipulatorEnv(
            mission=mission, action_mode=action_mode, max_episode_steps=60
        )
        obs, _ = env.reset(seed=seed)
        obs_list = [obs]
        rew_list: list[float] = []
        err_list: list[float] = []
        rng = np.random.default_rng(1234)
        for _ in range(50):
            act = rng.uniform(-0.5, 0.5, size=env.action_space.shape).astype(np.float32)
            obs, rew, term, trunc, info = env.step(act)
            obs_list.append(obs)
            rew_list.append(rew)
            err_list.append(info["ee_error"])
            if term or trunc:
                break
        env.close()
        return obs_list, rew_list, err_list

    obs1, rew1, err1 = rollout(42)
    obs2, rew2, err2 = rollout(42)

    assert len(obs1) == len(obs2)
    for o1, o2 in zip(obs1, obs2):
        assert np.array_equal(o1, o2)
    assert rew1 == rew2
    assert err1 == err2


def test_residual_gjm_zero_action_converges_reach() -> None:
    env = SpaceManipulatorEnv(
        mission="reach",
        action_mode="residual_gjm",
        max_episode_steps=250,
        randomize_init=False,
    )
    try:
        obs, info = env.reset(seed=7)
        assert obs.shape == (OBS_DIM,)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        terminated = False
        truncated = False
        for _ in range(250):
            obs, _, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        assert terminated, f"Expected convergence termination, got info={info}"
        assert info["is_success"] is True
        assert info["ee_error"] < 0.03
        assert info["fuel_impulse_step"] == 0.0
        assert info["cumulative_fuel_impulse"] == 0.0
    finally:
        env.close()


def test_free_floating_momentum_conservation_under_random_actions() -> None:
    env = SpaceManipulatorEnv(
        mission="reach",
        action_mode="torque",
        max_episode_steps=100,
        randomize_init=True,
    )
    try:
        env.reset(seed=99)
        p0, l0 = compute_system_momentum(env.model, env.data, indices=env.indices)
        rng = np.random.default_rng(99)

        for _ in range(50):
            act = rng.uniform(-0.8, 0.8, size=env.action_space.shape).astype(np.float32)
            _, _, term, trunc, info = env.step(act)
            if info["ncon"] == 0 and env.data.nefc == 0:
                p_cur, l_cur = compute_system_momentum(
                    env.model, env.data, indices=env.indices
                )
                assert float(np.linalg.norm(p_cur - p0)) < 1e-5
                assert float(np.linalg.norm(l_cur - l0)) < 1e-5
            if term or trunc:
                break
    finally:
        env.close()


def test_moving_target_trajectory_and_velocity() -> None:
    env = SpaceManipulatorEnv(
        mission="moving_target",
        action_mode="torque",
        max_episode_steps=50,
    )
    try:
        obs0, _ = env.reset(seed=11)
        pos0 = env.data.mocap_pos[env.indices.target_mocap_id].copy()
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

        dt = env.model.opt.timestep * env.config.frame_skip
        terminated = False
        for _ in range(10):
            obs1, _, terminated, _, info = env.step(zero_action)
            pos1 = env.data.mocap_pos[env.indices.target_mocap_id].copy()
            fd_vel = (pos1 - pos0) / dt
            reported_vel = obs1[25:28].astype(np.float64)
            assert np.linalg.norm(pos1 - pos0) > 1e-4
            # Midpoint/endpoint finite difference matches analytical velocity within O(dt)
            assert np.allclose(fd_vel, reported_vel, atol=5e-3)
            pos0 = pos1
            assert not terminated
            assert info["fuel_impulse_step"] == 0.0
        assert obs0.shape == (OBS_DIM,)
    finally:
        env.close()


def test_fuel_optimal_impulse_accounting() -> None:
    env = SpaceManipulatorEnv(
        mission="fuel_optimal",
        action_mode="torque",
        max_episode_steps=50,
    )
    try:
        env.reset(seed=5)
        zero_base_act = np.array([0, 0, 0, 0, 0, 0, 0.2, -0.2, 0.1], dtype=np.float32)
        _, _, _, _, info0 = env.step(zero_base_act)
        assert info0["fuel_impulse_step"] == 0.0
        assert info0["cumulative_fuel_impulse"] == 0.0

        nonzero_base_act = np.array(
            [0.5, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        _, _, _, _, info1 = env.step(nonzero_base_act)
        assert info1["fuel_impulse_step"] > 0.0
        assert np.isclose(info1["cumulative_fuel_impulse"], info1["fuel_impulse_step"])
    finally:
        env.close()


def test_register_envs_and_gymnasium_make() -> None:
    ids1 = register_envs()
    ids2 = register_envs()  # idempotent
    assert ids1 == tuple(ENV_IDS.keys())
    assert ids2 == tuple(ENV_IDS.keys())

    for env_id, expected_mission in ENV_IDS.items():
        env = gym.make(env_id)
        try:
            assert env.unwrapped.config.mission == expected_mission
            obs, info = env.reset(seed=0)
            assert obs.shape == (OBS_DIM,)
            assert "reward_terms" in info
        finally:
            env.close()


def test_invalid_config_and_action_shape_raise() -> None:
    with pytest.raises(ValueError, match="Invalid mission"):
        SpaceManipulatorEnv(mission="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid action_mode"):
        SpaceManipulatorEnv(action_mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid render_mode"):
        SpaceManipulatorEnv(render_mode="human")

    env = SpaceManipulatorEnv(mission="reach", action_mode="torque")
    try:
        env.reset(seed=0)
        assert env.render() is None
        with pytest.raises(ValueError, match="Expected action shape"):
            env.step(np.zeros(9, dtype=np.float32))
    finally:
        env.close()
