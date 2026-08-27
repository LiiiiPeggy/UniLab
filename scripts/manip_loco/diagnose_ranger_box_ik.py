"""Deterministic IK windup diagnostic for RangerBoxReach.

Run:  uv run scripts/manip_loco/diagnose_ranger_box_ik.py

Tests a grid of fixed goals at varying distance / direction from the armbase,
recording whether the persistent IK target stays bounded (no windup) or rails
to the soft joint limits.  This is the gate before any PPO training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402

from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")


def _make_env():
    ensure_registries()
    overrides = [
        "task=ranger_box_reach/mujoco",
        "algo.num_envs=1",
        "algo.max_iterations=10",
        "env.control_config.arm_action_scale=0.0",
        "env.control_config.arm_max_delta_per_step=0.01",
        "env.ik.gain=0.2",
        "env.ik.dq_clip=0.03",
        "env.noise_config.level=0.0",
        "env.base_velocity_controller.enable_latency=false",
        "env.base_velocity_controller.enable_noise=false",
        "env.base_velocity_controller.enable_wheel_visualization=false",
        "env.domain_rand.randomize_kp=false",
        "env.domain_rand.randomize_kd=false",
        "env.domain_rand.randomize_body_mass=false",
        "env.domain_rand.random_com=false",
        "env.domain_rand.randomize_dof_armature=false",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _run_goal(env, goal_local: np.ndarray, n_steps: int = 300):
    env.reset(np.array([0]))
    env.init_state()
    armbase_pos = env.armbase_pos_world[0]
    armbase_quat = env.armbase_quat_world[0]
    goal_world = armbase_pos + np_quat_apply_batched(armbase_quat[None], goal_local[None])[0]
    env.world_ee_goal[:] = goal_world
    dists = []
    max_target_err = 0.0
    min_soft_margin = 1e9
    for _ in range(n_steps):
        env.step(np.zeros((1, 9)))
        ee_local, _ = env.get_ee_local_pose()
        ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        dists.append(np.linalg.norm(ee_world[0] - goal_world))
        max_target_err = max(
            max_target_err, np.abs(env._ik_target[0] - env.get_arm_dof_pos()[0]).max()
        )
        # min distance from target to soft limits
        min_soft_margin = min(
            min_soft_margin,
            float(
                np.minimum(
                    env._ik_target[0] - env._arm_soft_lower, env._arm_soft_upper - env._ik_target[0]
                ).min()
            ),
        )
    return {
        "d0": dists[0],
        "d_final": dists[-1],
        "d_min": min(dists),
        "max_target_err": max_target_err,
        "min_soft_margin": min_soft_margin,
    }


def main() -> None:
    env = _make_env()
    print("=== RangerBoxReach IK windup diagnostic ===\n")
    print(f"{'goal':>14} {'d0':>6} {'d_final':>8} {'d_min':>7} {'max|t-q|':>9} {'soft_margin':>12}")

    # A: clearly reachable (front, ~0.3m)
    goals = {
        "A front 0.3": np.array([0.30, 0.0, 0.10]),
        "A front 0.4": np.array([0.40, 0.0, 0.10]),
        "B high 0.4": np.array([0.30, 0.0, 0.45]),
        "B side 0.35": np.array([0.20, 0.30, 0.10]),
        "C far 1.0": np.array([1.00, 0.0, 0.10]),
        "C far 1.5": np.array([1.50, 0.0, 0.10]),
        "D behind -0.3": np.array([-0.30, 0.0, 0.10]),
        "D below 0.2": np.array([0.30, 0.0, -0.30]),
    }
    for name, g in goals.items():
        r = _run_goal(env, g)
        flag = "WINDU P" if r["min_soft_margin"] < 0.01 else "ok"
        print(
            f"{name:>14} {r['d0']:6.2f} {r['d_final']:8.2f} {r['d_min']:7.2f} "
            f"{r['max_target_err']:9.3f} {r['min_soft_margin']:12.3f}  {flag}"
        )

    env.close()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
