from pathlib import Path
import argparse
import numpy as np

from stable_baselines3 import SAC

from rope_arm_reach_env import RopeArmEnvConfig, RopeArmReachEnv, RopePIDConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("rl_models/sac_rope_arm_reference_then_rl_SAC8.zip"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--xml", type=str, default="scene.xml")
    parser.add_argument("--dataset", type=str, default="dataset/workspace_6rope.csv")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--passive-cache", type=Path, default=Path("cache/passive_settled_state.npz"))

    parser.add_argument("--reference-error-min", type=float, default=0.015)
    parser.add_argument("--reference-error-max", type=float, default=0.08)
    parser.add_argument("--success-tol", type=float, default=0.03)

    parser.add_argument("--action-scale", type=float, default=0.0005)
    parser.add_argument("--command-rate-limit", type=float, default=0.20)
    parser.add_argument("--kp", type=float, default=60.0)
    parser.add_argument("--kd", type=float, default=1.5)
    parser.add_argument("--force-max", type=float, default=15.0)
    parser.add_argument("--force-rate-limit", type=float, default=150.0)
    parser.add_argument("--pid-target-rate-limit", type=float, default=1.2)

    parser.add_argument("--passive-settle-steps", type=int, default=500)
    parser.add_argument("--reference-steps", type=int, default=500)
    parser.add_argument("--reference-ramp-steps", type=int, default=200)
    parser.add_argument("--reference-hold-steps", type=int, default=40)
    parser.add_argument("--max-reference-retries", type=int, default=40)

    # New reference stabilization parameters. These must match the updated
    # RopeArmEnvConfig fields in rope_arm_reach_env.py.
    parser.add_argument("--reference-stable-max-steps", type=int, default=300)
    parser.add_argument("--reference-stable-hold-steps", type=int, default=20)
    parser.add_argument("--reference-qvel-tol", type=float, default=0.05)
    parser.add_argument("--reference-tip-delta-tol", type=float, default=1e-4)

    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--debug-done", action="store_true")
    parser.add_argument("--debug-reset", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()

    cfg = RopeArmEnvConfig(
        root=root,
        xml_path=root / args.xml,
        dataset_csv=root / args.dataset,
        reference_error_min=args.reference_error_min,
        reference_error_max=args.reference_error_max,
        success_tol=args.success_tol,
        action_scale=args.action_scale,
        command_rate_limit=args.command_rate_limit,
        passive_settle_steps=args.passive_settle_steps,
        reference_steps=args.reference_steps,
        reference_ramp_steps=args.reference_ramp_steps,
        reference_hold_steps=args.reference_hold_steps,
        max_reference_retries=args.max_reference_retries,
        reference_stable_max_steps=args.reference_stable_max_steps,
        reference_stable_hold_steps=args.reference_stable_hold_steps,
        reference_qvel_tol=args.reference_qvel_tol,
        reference_tip_delta_tol=args.reference_tip_delta_tol,
        passive_cache_path=args.passive_cache,
        pid=RopePIDConfig(
            kp=args.kp,
            kd=args.kd,
            force_max=args.force_max,
            force_rate_limit=args.force_rate_limit,
            target_rate_limit=args.pid_target_rate_limit,
        ),
        debug_done=args.debug_done,
        debug_reset=args.debug_reset,
    )

    env = RopeArmReachEnv(cfg)

    model = None
    if not args.zero_action:
        model = SAC.load(args.model, device="cpu")

    success_count = 0
    timeout_count = 0
    unstable_count = 0
    final_dists = []
    reference_dists = []
    episode_lengths = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        done = False
        step_count = 0
        final_info = {}

        while not done:
            if args.zero_action:
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1
            final_info = info

        reason = final_info.get("done_reason", "unknown")
        final_dist = float(final_info.get("dist", np.nan))
        reference_dist = float(final_info.get("reference_dist", np.nan))

        final_dists.append(final_dist)
        reference_dists.append(reference_dist)
        episode_lengths.append(step_count)

        if reason == "success":
            success_count += 1
        elif reason == "unstable":
            unstable_count += 1
        elif reason == "timeout":
            timeout_count += 1

        print(
            f"EP {ep + 1:03d} | "
            f"reason={reason:8s} | "
            f"steps={step_count:3d} | "
            f"ref_dist={reference_dist:.4f} | "
            f"final_dist={final_dist:.4f}"
        )

    final_dists = np.asarray(final_dists, dtype=np.float64)
    reference_dists = np.asarray(reference_dists, dtype=np.float64)
    episode_lengths = np.asarray(episode_lengths, dtype=np.float64)

    improved = final_dists < reference_dists
    improvement = reference_dists - final_dists
    reference_success = reference_dists <= args.success_tol

    print("\n========== Evaluation Summary ==========")
    print(f"Mode:              {'zero-action' if args.zero_action else 'SAC policy'}")
    print(f"Episodes:          {args.episodes}")
    print(f"Success rate:      {success_count / args.episodes * 100:.1f}%")
    print(f"Success count:     {success_count}")
    print(f"Timeout count:     {timeout_count}")
    print(f"Unstable count:    {unstable_count}")
    print(f"Reference success: {np.mean(reference_success) * 100:.1f}%")
    print(f"Improved rate:     {np.mean(improved) * 100:.1f}%")
    print(f"Mean improvement:  {np.nanmean(improvement):.4f}")
    print(f"Mean ref dist:     {np.nanmean(reference_dists):.4f}")
    print(f"Mean final dist:   {np.nanmean(final_dists):.4f}")
    print(f"Median final:      {np.nanmedian(final_dists):.4f}")
    print(f"Mean ep length:    {np.mean(episode_lengths):.1f}")
    print("========================================")

    env.close()


if __name__ == "__main__":
    main()
