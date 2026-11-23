"""Gymnasium RL environments and registration helpers for ``space_manipulator``."""

from __future__ import annotations

import gymnasium as gym

from space_manipulator.envs.space_manipulator_env import (
    OBS_DIM,
    OBS_ORDER,
    VALID_ACTION_MODES,
    VALID_MISSION_MODES,
    VALID_RENDER_MODES,
    ActionMode,
    EnvConfig,
    MissionMode,
    RewardWeights,
    SpaceManipulatorEnv,
)

ENV_IDS: dict[str, MissionMode] = {
    "SpaceManipulatorReach-v0": "reach",
    "SpaceManipulatorLowDisturbance-v0": "low_disturbance",
    "SpaceManipulatorFuelOptimal-v0": "fuel_optimal",
    "SpaceManipulatorMovingTarget-v0": "moving_target",
}


def register_envs() -> tuple[str, ...]:
    """Idempotently register all ``SpaceManipulator*-v0`` Gymnasium environments."""
    for env_id, mission in ENV_IDS.items():
        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point="space_manipulator.envs:SpaceManipulatorEnv",
                kwargs={"mission": mission},
            )
    return tuple(ENV_IDS.keys())


__all__ = [
    "ENV_IDS",
    "OBS_DIM",
    "OBS_ORDER",
    "VALID_ACTION_MODES",
    "VALID_MISSION_MODES",
    "VALID_RENDER_MODES",
    "ActionMode",
    "EnvConfig",
    "MissionMode",
    "RewardWeights",
    "SpaceManipulatorEnv",
    "register_envs",
]
