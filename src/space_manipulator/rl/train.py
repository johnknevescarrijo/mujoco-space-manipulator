"""Stable-Baselines3 training script and Python API for SpaceManipulatorEnv."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Literal

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from space_manipulator.envs import (
    VALID_ACTION_MODES,
    VALID_MISSION_MODES,
    ActionMode,
    EnvConfig,
    MissionMode,
    SpaceManipulatorEnv,
)

AlgoName = Literal["ppo", "sac"]
VALID_ALGOS: tuple[str, ...] = ("ppo", "sac")


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for SB3 training on :class:`SpaceManipulatorEnv`."""

    algo: AlgoName = "ppo"
    mission: MissionMode = "reach"
    action_mode: ActionMode = "torque"
    total_timesteps: int = 200_000
    n_envs: int = 4
    seed: int = 42
    frame_skip: int = 10
    max_episode_steps: int = 250
    init_base_angvel_std: float = 0.0
    eval_freq: int = 10_000
    eval_episodes: int = 10
    normalize_obs: bool = False
    learning_rate: float = 3e-4
    device: str = "auto"
    output_dir: str | Path | None = None
    # Optional algorithm hyperparameter overrides (useful for tuning and tests)
    n_steps: int | None = None
    batch_size: int | None = None
    n_epochs: int | None = None
    learning_starts: int | None = None

    def validate(self) -> None:
        """Validate training configuration parameters."""
        if self.algo not in VALID_ALGOS:
            raise ValueError(f"Invalid algo '{self.algo}'. Expected one of {VALID_ALGOS}.")
        if self.mission not in VALID_MISSION_MODES:
            raise ValueError(
                f"Invalid mission '{self.mission}'. Expected one of {VALID_MISSION_MODES}."
            )
        if self.action_mode not in VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode '{self.action_mode}'. "
                f"Expected one of {VALID_ACTION_MODES}."
            )
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps must be >= 1.")
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1.")
        if self.eval_freq < 1:
            raise ValueError("eval_freq must be >= 1.")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.0.")

    @property
    def resolved_output_dir(self) -> Path:
        """Return the concrete output directory for this training run."""
        if self.output_dir is not None:
            return Path(self.output_dir)
        return Path(
            f"runs/{self.mission}_{self.action_mode}_{self.algo}_seed{self.seed}"
        )

    def to_env_config(self) -> EnvConfig:
        """Build the corresponding :class:`EnvConfig`."""
        return EnvConfig(
            mission=self.mission,
            action_mode=self.action_mode,
            frame_skip=self.frame_skip,
            max_episode_steps=self.max_episode_steps,
            init_base_angvel_std=self.init_base_angvel_std,
        )


def _make_env_factory(
    env_config: EnvConfig,
    seed: int,
    monitor_file: str | None = None,
) -> Callable[[], Monitor]:
    """Return a zero-argument callable that builds a seeded, Monitored env."""

    def _init() -> Monitor:
        env = SpaceManipulatorEnv(env_config)
        env.reset(seed=seed)
        return Monitor(env, filename=monitor_file)

    return _init


def build_vec_env(
    config: TrainConfig,
    *,
    for_eval: bool = False,
    monitor_dir: Path | None = None,
) -> VecEnv:
    """Create a vectorized (and optionally normalized) environment."""
    config.validate()
    env_cfg = config.to_env_config()
    num_envs = 1 if for_eval else config.n_envs
    seed_offset = 10_000 if for_eval else 0

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    factories = [
        _make_env_factory(
            env_cfg,
            seed=config.seed + seed_offset + rank,
            monitor_file=(
                str(monitor_dir / f"env_{rank}") if monitor_dir is not None else None
            ),
        )
        for rank in range(num_envs)
    ]

    if not for_eval and num_envs > 1:
        vec_env: VecEnv = SubprocVecEnv(factories)
    else:
        vec_env = DummyVecEnv(factories)

    if config.normalize_obs:
        vec_env = VecNormalize(
            vec_env,
            training=not for_eval,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )

    return vec_env


def _build_sb3_model(config: TrainConfig, train_env: VecEnv) -> PPO | SAC:
    """Instantiate the requested Stable-Baselines3 algorithm."""
    if config.algo == "ppo":
        kwargs: dict[str, Any] = {
            "policy": "MlpPolicy",
            "env": train_env,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "device": config.device,
            "verbose": 1,
        }
        if config.n_steps is not None:
            kwargs["n_steps"] = config.n_steps
        if config.batch_size is not None:
            kwargs["batch_size"] = config.batch_size
        if config.n_epochs is not None:
            kwargs["n_epochs"] = config.n_epochs
        return PPO(**kwargs)

    kwargs = {
        "policy": "MlpPolicy",
        "env": train_env,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
        "device": config.device,
        "verbose": 1,
    }
    if config.batch_size is not None:
        kwargs["batch_size"] = config.batch_size
    if config.learning_starts is not None:
        kwargs["learning_starts"] = config.learning_starts
    return SAC(**kwargs)


def train_agent(config: TrainConfig) -> dict[str, Any]:
    """Train a PPO or SAC agent on :class:`SpaceManipulatorEnv` and save artifacts."""
    config.validate()
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env: VecEnv | None = None
    eval_env: VecEnv | None = None
    try:
        train_env = build_vec_env(
            config, for_eval=False, monitor_dir=output_dir / "monitor"
        )
        eval_env = build_vec_env(config, for_eval=True)

        # Extract canonical env configuration dict for reproducibility
        probe_env = SpaceManipulatorEnv(config.to_env_config())
        env_config_dict = probe_env.get_config_dict()
        probe_env.close()

        config_dict = asdict(config)
        config_dict["output_dir"] = str(output_dir)
        config_dict["env_config"] = env_config_dict

        config_path = output_dir / "config.json"
        config_path.write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

        model = _build_sb3_model(config, train_env)

        eval_freq_calls = max(1, config.eval_freq // config.n_envs)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir),
            eval_freq=eval_freq_calls,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
        )

        model.learn(total_timesteps=config.total_timesteps, callback=eval_callback)

        final_model_stem = output_dir / "final_model"
        model.save(str(final_model_stem))
        final_model_path = output_dir / "final_model.zip"

        vecnormalize_path: Path | None = None
        if config.normalize_obs and isinstance(train_env, VecNormalize):
            vecnormalize_path = output_dir / "vecnormalize.pkl"
            train_env.save(str(vecnormalize_path))

        best_model_path = output_dir / "best_model.zip"
        evaluations_path = output_dir / "evaluations.npz"

        return {
            "output_dir": output_dir,
            "config_path": config_path,
            "final_model_path": final_model_path,
            "best_model_path": best_model_path if best_model_path.exists() else None,
            "vecnormalize_path": vecnormalize_path,
            "evaluations_path": evaluations_path if evaluations_path.exists() else None,
        }
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for ``space-manipulator-train``."""
    parser = argparse.ArgumentParser(
        description="Train PPO or SAC on SpaceManipulatorEnv using Stable-Baselines3."
    )
    parser.add_argument("--algo", choices=VALID_ALGOS, default="ppo")
    parser.add_argument("--mission", choices=VALID_MISSION_MODES, default="reach")
    parser.add_argument("--action-mode", choices=VALID_ACTION_MODES, default="torque")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=250)
    parser.add_argument("--init-base-angvel-std", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--learning-starts", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for training SB3 agents."""
    args = parse_args(argv)
    config = TrainConfig(
        algo=args.algo,
        mission=args.mission,
        action_mode=args.action_mode,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_episode_steps,
        init_base_angvel_std=args.init_base_angvel_std,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        normalize_obs=args.normalize_obs,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_starts=args.learning_starts,
    )
    result = train_agent(config)
    print(f"Training complete. Artifacts saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
