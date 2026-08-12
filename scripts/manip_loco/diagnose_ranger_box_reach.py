"""Deterministic smoke tests for RangerBoxReach (Test A/B/C).

Run with:  uv run python scripts/manip_loco/diagnose_ranger_box_reach.py

Test A — arm static hold:   base fixed, action=0, IK=0, home target, 10 s.
Test B — pure-IK reach:     base fixed, RL residual=0, IK on, one reachable goal.
Test C — base-only:         arm fixed home, only 3D base action.

These are NOT training runs; they verify the low-level controller/env
behaviour is stable before any PPO training.
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


def _clean_overrides() -> list[str]:
    return [
        "task=ranger_box_reach/mujoco",
        "algo.num_envs=1",
        "algo.max_iterations=10",
        "env.control_config.arm_action_scale=0.0",
        "env.control_config.arm_max_delta_per_step=0.01",
        "env.ik.gain=0.0",
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


def _make_env(overrides: list[str]):
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env, cfg


def _ee_world(env) -> np.ndarray:
    ee_local, _ = env.get_ee_local_pose()
    return env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)


def test_a_arm_static_hold():
    env, cfg = _make_env(_clean_overrides())
    env.reset(np.array([0]))
    env.init_state()
    arm0 = env.get_arm_dof_pos()[0].copy()
    ee_z0 = _ee_world(env)[0, 2]
    max_drift = np.zeros(6)
    max_target_err = np.zeros(6)
    for _ in range(500):  # 10 s
        state = env.step(np.zeros((1, 9)))
        arm = env.get_arm_dof_pos()[0]
        max_drift = np.maximum(max_drift, np.abs(arm - arm0))
        max_target_err = np.maximum(max_target_err, np.abs(env._ik_target[0] - arm))
    ee_z1 = _ee_world(env)[0, 2]
    print("[Test A] arm static hold")
    print(f"  max joint drift:    {np.round(max_drift, 4)}")
    print(f"  max |target-actual|: {np.round(max_target_err, 4)}")
    print(f"  EE z drift: {ee_z0:.3f} -> {ee_z1:.3f}  (delta {ee_z1 - ee_z0:+.4f})")
    print(f"  terminated: {bool(state.terminated[0])}")
    env.close()
    return float(max_drift.max())


def test_b_pure_ik_reach():
    overrides = _clean_overrides()
    overrides[overrides.index("env.ik.gain=0.0")] = "env.ik.gain=0.2"
    env, cfg = _make_env(overrides)
    env.reset(np.array([0]))
    env.init_state()
    # Place a fixed reachable goal ~0.35 m in front of the armbase.
    armbase_pos = env.armbase_pos_world[0]
    armbase_quat = env.armbase_quat_world[0]
    goal_local = np.array([0.35, 0.0, 0.15])
    goal_world = armbase_pos + np_quat_apply_batched(armbase_quat[None], goal_local[None])[0]
    env.world_ee_goal[:] = goal_world
    dist0 = np.linalg.norm(_ee_world(env)[0] - goal_world)
    dists = [dist0]
    for _ in range(300):
        state = env.step(np.zeros((1, 9)))
        dists.append(np.linalg.norm(_ee_world(env)[0] - goal_world))
        if state.terminated[0]:
            break
    print("[Test B] pure IK reach")
    print(f"  EE dist: {dists[0]:.3f} -> {dists[-1]:.3f}  (min {min(dists):.3f})")
    print(f"  terminated: {bool(state.terminated[0])}")
    env.close()
    return float(dists[-1])


def test_c_base_only():
    env, cfg = _make_env(_clean_overrides())
    env.reset(np.array([0]))
    env.init_state()
    base0 = env._backend.get_base_pos()[0, :2].copy()
    # Constant forward command via base action
    for _ in range(100):
        act = np.zeros((1, 9))
        act[0, 0] = 0.5  # vx
        state = env.step(act)
    base1 = env._backend.get_base_pos()[0, :2].copy()
    disp = np.linalg.norm(base1 - base0)
    print("[Test C] base-only")
    print(f"  base displacement: {disp:.3f} m")
    print(f"  terminated: {bool(state.terminated[0])}")
    env.close()
    return float(disp)


def main() -> None:
    print("=== RangerBoxReach deterministic smoke tests ===\n")
    test_a_arm_static_hold()
    print()
    test_b_pure_ik_reach()
    print()
    test_c_base_only()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
