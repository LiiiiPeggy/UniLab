"""Goal-sampler distribution validation for RangerBoxReach (Task 15).

Usage:
  uv run scripts/manip_loco/validate_ranger_box_goal_sampler.py \
      [--num-envs 32] [--rounds 313] [--output sampler_stats.json]

Repeatedly resets EE goals (no policy, no physics rollout) and aggregates the
sampled-goal distribution over >= 10k goals:

  - local/extended mix vs ``local_fraction``
  - LOCAL: EE-to-goal radial distance in ``local_radius_range``
  - EXTENDED: r_xy in ``extended_xy_radius_range``, dz in ``extended_z_range``
  - base-only-impossible fraction: |goal_z - ready_EE_z| >= ``capture_outer``
    (must be ~0 for the planar EXTENDED sampler)
  - world z clamp hits (goal z == 0.15 floor), fallback-goal usage

Writes JSON stats to ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_ranger_box_reach import (  # noqa: E402
    CLEAN_OVERRIDES,
    CONF_DIR,
    ROOT_DIR,
)
from hydra import compose, initialize_config_dir  # noqa: E402

from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402


def rng_stats(a: np.ndarray) -> dict:
    return {
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "mean": float(a.mean()),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=313, help="goals = num_envs * rounds")
    ap.add_argument("--output", default="sampler_stats.json")
    ap.add_argument(
        "--local-fraction",
        type=float,
        default=None,
        help="override env.goal_ee.local_fraction",
    )
    args = ap.parse_args()

    ensure_registries()
    overrides = ["task=ranger_box_reach/mujoco", f"algo.num_envs={args.num_envs}"]
    if args.local_fraction is not None:
        overrides.append(f"env.goal_ee.local_fraction={args.local_fraction}")
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides + CLEAN_OVERRIDES)
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=args.num_envs, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)

    goal_cfg = env._cfg.goal_ee
    capture_outer = float(goal_cfg.capture_outer)
    n = args.num_envs
    total = n * args.rounds

    loc_arr, radial_arr, rxy_arr, dz_arr = [], [], [], []
    floor_arr, z_arr, fb_arr = [], [], []
    fallback_delta = np.array([0.10, 0.0, -0.05])  # reset_ee_goals fallback offset

    for rnd in range(args.rounds):
        env.reset_ee_goals(np.arange(n))
        ee_local, _ = env.get_ee_local_pose()
        ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        goal = env.world_ee_goal
        d = np.linalg.norm(goal - ee_world, axis=1)
        gl = env._world_goal_to_armbase(goal, env.armbase_pos_world, env.armbase_quat_world)

        loc_arr.append(env._goal_is_local.copy())
        radial_arr.append(d)
        rxy_arr.append(np.linalg.norm(goal[:, :2] - ee_world[:, :2], axis=1))
        dz_arr.append(goal[:, 2] - ee_world[:, 2])
        floor_arr.append(np.abs(goal[:, 2] - ee_world[:, 2]))
        z_arr.append(goal[:, 2].copy())
        # The fallback goal is written when all resample attempts fail; detect
        # the exact EE-relative offset [0.10, 0, -0.05] in the armbase frame.
        fb_arr.append((np.linalg.norm(gl - (ee_local + fallback_delta[None, :]), axis=1) < 1e-6))
        if (rnd + 1) % 50 == 0:
            print(f"  round {rnd + 1}/{args.rounds} ({(rnd + 1) * n} goals)")

    loc = np.concatenate(loc_arr).astype(bool)
    radial = np.concatenate(radial_arr)
    r_xy = np.concatenate(rxy_arr)
    dz = np.concatenate(dz_arr)
    floor_dz = np.concatenate(floor_arr)
    z = np.concatenate(z_arr)
    fb = np.concatenate(fb_arr)
    assert radial.shape[0] == total, f"expected {total} goals, got {radial.shape[0]}"

    ext = ~loc
    stats = {
        "num_goals": int(total),
        "local_fraction_config": float(goal_cfg.local_fraction),
        "local_fraction_observed": float(loc.mean()),
        "extended_impossible_fraction": float((floor_dz >= capture_outer).mean()),
        "extended_impossible_fraction_ext_only": (
            float((floor_dz[ext] >= capture_outer).mean()) if ext.any() else None
        ),
        "local_radial_dist": rng_stats(radial[loc]),
        "extended_r_xy": rng_stats(r_xy[ext]),
        "extended_dz": rng_stats(dz[ext]),
        "extended_vertical_floor": rng_stats(floor_dz[ext]),
        "goal_z": rng_stats(z),
        "goal_z_clamp_hits": float((z <= 0.15 + 1e-9).mean()),
        "fallback_used_fraction": float(fb.mean()),
        "capture_inner": float(goal_cfg.capture_inner),
        "capture_outer": capture_outer,
        "extended_xy_radius_range": [float(v) for v in goal_cfg.extended_xy_radius_range],
        "extended_z_range": [float(v) for v in goal_cfg.extended_z_range],
    }

    print(json.dumps(stats, indent=2))
    out_path = Path(args.output)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path}")

    env.close()


if __name__ == "__main__":
    main()
