"""Train SAC for six-tendon residual reaching after dataset-reference control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from rope_arm_reach_env import RopeArmEnvConfig, RopeArmReachEnv, RopePIDConfig


@dataclass(slots=True)
class CurriculumStage:
    start_step: int
    reference_error_min: float
    reference_error_max: float
    success_tol: float
    name: str


class CurriculumCallback(BaseCallback):
    def __init__(self, stages: list[CurriculumStage], verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.stages = sorted(stages, key=lambda s: s.start_step)
        self.current_stage_idx = -1

    def _on_step(self) -> bool:
        stage_idx = 0
        for i, stage in enumerate(self.stages):
            if self.num_timesteps >= stage.start_step:
                stage_idx = i
        if stage_idx != self.current_stage_idx:
            stage = self.stages[stage_idx]
            self.training_env.env_method(
                "set_curriculum",
                reference_error_min=stage.reference_error_min,
                reference_error_max=stage.reference_error_max,
                success_tol=stage.success_tol,
            )
            self.current_stage_idx = stage_idx
            if self.verbose:
                print(
                    f"[Curriculum] step={self.num_timesteps} -> {stage.name}: "
                    f"reference_error=[{stage.reference_error_min:.3f}, {stage.reference_error_max:.3f}], "
                    f"success_tol={stage.success_tol:.3f}"
                )
        return True


def build_curriculum(total_timesteps: int) -> list[CurriculumStage]:
    # Curriculum is based on the residual error after dataset-reference control:
    #     reference_dist = ||target_xyz - reference_tip_xyz||
    # Stage 1 avoids trivial samples where reference_dist is almost zero, but
    # also avoids very hard samples. Later stages gradually widen the range.
    return [
        CurriculumStage(0, 0.020, 0.060, 0.030, "moderate reference residuals"),
        # CurriculumStage(int(total_timesteps * 0.25), 0.015, 0.090, 0.030, "wider residuals"),
        # CurriculumStage(int(total_timesteps * 0.55), 0.010, 0.130, 0.028, "medium residual workspace"),
        # CurriculumStage(int(total_timesteps * 0.80), 0.000, 0.180, 0.025, "large residual workspace"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC residual correction after dataset reference control.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--dataset", type=str, default="dataset/workspace_6rope.csv")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument(
        "--vec-env",
        choices=("auto", "dummy", "subproc"),
        default="auto",
        help="Use subproc for true parallel rollout when n-envs > 1.",
    )

    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--success-hold-steps", type=int, default=5)
    parser.add_argument("--initial-reference-error-min", type=float, default=0.020)
    parser.add_argument("--initial-reference-error-max", type=float, default=0.060)
    parser.add_argument("--initial-success-tol", type=float, default=0.030)

    parser.add_argument("--action-scale", type=float, default=0.0005)
    parser.add_argument("--command-rate-limit", type=float, default=0.20)
    parser.add_argument("--max-contract", type=float, default=0.60)
    parser.add_argument("--max-extend", type=float, default=0.03)

    parser.add_argument("--passive-settle-steps", type=int, default=500)
    parser.add_argument("--reference-steps", type=int, default=500)
    parser.add_argument("--reference-ramp-steps", type=int, default=200)
    parser.add_argument("--reference-hold-steps", type=int, default=40, help="Minimum hold steps after reference ramp before stability checking can pass.")
    parser.add_argument("--reference-stable-max-steps", type=int, default=300, help="Extra hold steps allowed while waiting for qvel/tip stability.")
    parser.add_argument("--reference-stable-hold-steps", type=int, default=20, help="Consecutive stable steps required before starting the RL episode.")
    parser.add_argument("--reference-qvel-tol", type=float, default=0.05, help="Max abs qvel threshold for reference stability.")
    parser.add_argument("--reference-tip-delta-tol", type=float, default=1e-4, help="Per-step tip displacement threshold for reference stability.")
    parser.add_argument("--max-reference-retries", type=int, default=40)
    parser.add_argument(
        "--strict-reference-sampling",
        action="store_true",
        help="Fail reset if no row satisfies reference-error range after retries; default accepts nearest candidate.",
    )
    parser.add_argument(
        "--passive-cache",
        type=Path,
        default=Path("cache/passive_settled_state.npz"),
        help="Cache file for passive-settled MuJoCo state. Delete it after changing scene.xml or plate setup.",
    )

    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--force-max", type=float, default=15.0)
    parser.add_argument("--force-rate-limit", type=float, default=150.0)
    parser.add_argument("--pid-target-rate-limit", type=float, default=1.2)

    parser.add_argument("--model-dir", type=Path, default=Path("rl_models"))
    parser.add_argument("--log-dir", type=Path, default=Path("rl_logs"))
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--save-replay-buffer", action="store_true")
    parser.add_argument("--debug-done", action="store_true")
    parser.add_argument("--debug-reset", action="store_true")
    return parser.parse_args()


def make_env(root: Path, seed: int, args: argparse.Namespace):
    def _factory():
        cfg = RopeArmEnvConfig(
            root=root,
            xml_path=root / args.xml,
            dataset_csv=root / args.dataset,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_episode_steps,
            reference_error_min=args.initial_reference_error_min,
            reference_error_max=args.initial_reference_error_max,
            success_tol=args.initial_success_tol,
            success_hold_steps=args.success_hold_steps,
            action_scale=args.action_scale,
            command_rate_limit=args.command_rate_limit,
            max_contract=args.max_contract,
            max_extend=args.max_extend,
            passive_settle_steps=args.passive_settle_steps,
            reference_steps=args.reference_steps,
            reference_ramp_steps=args.reference_ramp_steps,
            reference_hold_steps=args.reference_hold_steps,
            reference_stable_max_steps=args.reference_stable_max_steps,
            reference_stable_hold_steps=args.reference_stable_hold_steps,
            reference_qvel_tol=args.reference_qvel_tol,
            reference_tip_delta_tol=args.reference_tip_delta_tol,
            max_reference_retries=args.max_reference_retries,
            accept_nearest_reference_on_retry_failure=not args.strict_reference_sampling,
            passive_cache_path=args.passive_cache,
            pid=RopePIDConfig(
                kp=args.kp,
                ki=0.0,
                kd=args.kd,
                force_max=args.force_max,
                force_rate_limit=args.force_rate_limit,
                target_rate_limit=args.pid_target_rate_limit,
            ),
            debug_done=args.debug_done,
            debug_reset=args.debug_reset,
        )
        env = RopeArmReachEnv(cfg)
        env.reset(seed=seed)
        return env
    return _factory


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    model_dir = args.model_dir.resolve()
    log_dir = args.log_dir.resolve()
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(root, args.seed + i, args) for i in range(args.n_envs)]
    use_subproc = args.vec_env == "subproc" or (args.vec_env == "auto" and args.n_envs > 1)
    if use_subproc:
        vec_env = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
        print(f"Using SubprocVecEnv with {args.n_envs} environments.")
    else:
        vec_env = VecMonitor(DummyVecEnv(env_fns))
        print(f"Using DummyVecEnv with {args.n_envs} environment(s).")

    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        buffer_size=300_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(log_dir),
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    callbacks = [
        CurriculumCallback(build_curriculum(args.total_timesteps)),
        CheckpointCallback(
            save_freq=max(10_000 // max(1, args.n_envs), 1),
            save_path=str(checkpoint_dir),
            name_prefix="sac_rope_arm_reference_then_rl",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
    ]

    final_path = model_dir / "sac_rope_arm_reference_then_rl.zip"
    interrupted_path = model_dir / "sac_rope_arm_reference_then_rl_interrupted.zip"

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=not args.no_progress_bar,
            log_interval=20,
        )
    except KeyboardInterrupt:
        model.save(interrupted_path)
        print(f"\nTraining interrupted. Saved current model to: {interrupted_path}")
        if args.save_replay_buffer:
            replay_path = model_dir / "sac_rope_arm_reference_then_rl_interrupted_replay_buffer.pkl"
            model.save_replay_buffer(str(replay_path))
            print(f"Saved replay buffer to: {replay_path}")
        return

    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    if args.save_replay_buffer:
        replay_path = model_dir / "sac_rope_arm_reference_then_rl_replay_buffer.pkl"
        model.save_replay_buffer(str(replay_path))
        print(f"Saved replay buffer to: {replay_path}")


if __name__ == "__main__":
    main()
