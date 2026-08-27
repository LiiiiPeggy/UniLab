"""Pure-IK reaching benchmark for RangerBoxReach — no policy, no RL residual.

Run:  uv run scripts/manip_loco/benchmark_ranger_box_ik_reach.py

GO/NO-GO gate before PPO.  Tests the redesigned task:
  - arm resets to the manipulation-ready folded pose (EE ~0.45 m from armbase,
    Jacobian condition ~1.6, gravcomp disabled → drift < 0.02 rad)
  - goals are sampled in the LOCAL-EE workspace (delta box x/y [-0.3,0.3],
    z [-0.2,0.3]) and filtered for IK feasibility at reset
  - the arm-engagement gate is feasibility-based (arm fully engaged)

Conditions: num_envs 16, RL residual OFF (arm_action_scale=0), noise OFF,
latency OFF, DR OFF, base stationary, max 10 s per episode, 100 episodes.

PASS criteria (user-specified):  success_10cm > 0.80  AND  final EE error < 0.05 m.
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

from unilab.envs.common.rotation import (  # noqa: E402
    np_quat_apply_batched,
    np_quat_conjugate_batched,
)
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")

NUM_ENVS = 16
EPISODES_TARGET = 100
MAX_STEPS = 500  # 10 s at ctrl_dt 0.02
SUCCESS_10 = 0.10
SUCCESS_05 = 0.05
PASS_SUCCESS = 0.80
PASS_FINAL = 0.05

# Same base-box geometry as the env reward (reach_env.py).
_BASE_BOX_HALF = np.array([0.55, 0.38, 0.20], dtype=np.float64)
_BOX_CENTRE_OFFSET = np.array([-0.1262, 0.0, -0.0965], dtype=np.float64)


def _overrides() -> list[str]:
    return [
        "task=ranger_box_reach/mujoco",
        f"algo.num_envs={NUM_ENVS}",
        "algo.max_iterations=10",
        "env.control_config.arm_action_scale=0.0",
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


def build_env():
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=_overrides())
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=NUM_ENVS, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _min_arm_signed_dist(env, ee_world) -> np.ndarray:
    """Min signed distance of arm links (Link2-6) + EE to the base box (<0 = collided)."""
    n = env._num_envs
    base_quat_w = env.armbase_quat_world
    box_centre_w = env.armbase_pos_world + np_quat_apply_batched(
        base_quat_w, np.broadcast_to(_BOX_CENTRE_OFFSET, (n, 3))
    )

    def _one(p_w):
        p_local = np_quat_apply_batched(np_quat_conjugate_batched(base_quat_w), p_w - box_centre_w)
        q = np.abs(p_local) - _BASE_BOX_HALF
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    pts = [ee_world] + [env._backend.get_sensor_data(name) for name in env._cfg.sensor.link_pos]
    sd = np.stack([_one(p) for p in pts], axis=1)
    return np.min(sd, axis=1)


def main() -> None:
    env = build_env()

    ep_goal_dist: list[float] = []
    ep_success10: list[bool] = []
    ep_success5: list[bool] = []
    ep_final: list[float] = []
    ep_min: list[float] = []
    ep_tts: list[int] = []
    ep_joint_lim: list[bool] = []
    ep_collision: list[bool] = []
    dist_curve: list[np.ndarray] = []

    n_episodes = 0
    while n_episodes < EPISODES_TARGET:
        env.reset(np.arange(NUM_ENVS))
        ee_local, _ = env.get_ee_local_pose()
        ee_world0 = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        g0 = np.linalg.norm(ee_world0 - env.world_ee_goal, axis=1)

        per_env_min = np.full(NUM_ENVS, 1e9)
        ep_success_once = np.zeros(NUM_ENVS, dtype=bool)
        ep_success_once5 = np.zeros(NUM_ENVS, dtype=bool)
        ep_tts_steps = np.zeros(NUM_ENVS, dtype=np.int32)
        per_env_jl = np.zeros(NUM_ENVS, dtype=bool)
        per_env_col = np.zeros(NUM_ENVS, dtype=bool)
        step_means: list[float] = []

        for step in range(MAX_STEPS):
            env.step(np.zeros((NUM_ENVS, 9)))  # zero action: base stops, IK-only arm
            ee_local, _ = env.get_ee_local_pose()
            ee_world = env.armbase_pos_world + np_quat_apply_batched(
                env.armbase_quat_world, ee_local
            )
            d = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
            step_means.append(float(np.mean(d)))
            per_env_min = np.minimum(per_env_min, d)
            new10 = (~ep_success_once) & (d < SUCCESS_10)
            ep_success_once |= d < SUCCESS_10
            ep_success_once5 |= d < SUCCESS_05
            ep_tts_steps = np.where(new10, step + 1, ep_tts_steps)
            arm = env.get_arm_dof_pos()
            per_env_jl |= ((arm > env._arm_joint_upper) | (arm < env._arm_joint_lower)).any(axis=1)
            per_env_col |= _min_arm_signed_dist(env, ee_world) < 0.0
            if np.all(ep_success_once):
                break

        ee_local, _ = env.get_ee_local_pose()
        ee_world_f = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        d_final = np.linalg.norm(ee_world_f - env.world_ee_goal, axis=1)

        for i in range(NUM_ENVS):
            ep_goal_dist.append(float(g0[i]))
            ep_success10.append(bool(ep_success_once[i]))
            ep_success5.append(bool(ep_success_once5[i]))
            ep_final.append(float(d_final[i]))
            ep_min.append(float(per_env_min[i]))
            ep_tts.append(int(ep_tts_steps[i]) if ep_success_once[i] else -1)
            ep_joint_lim.append(bool(per_env_jl[i]))
            ep_collision.append(bool(per_env_col[i]))
        dist_curve.append(np.asarray(step_means, dtype=np.float64))
        n_episodes += NUM_ENVS

    env.close()
    n_episodes = min(n_episodes, len(ep_goal_dist))

    def _arr(name):
        return np.asarray(name, dtype=np.float64)[:n_episodes]

    suc10 = _arr(ep_success10).astype(bool)
    suc5 = _arr(ep_success5).astype(bool)
    final = _arr(ep_final)
    mine = _arr(ep_min)
    tts = np.asarray(ep_tts, dtype=np.int64)[:n_episodes]
    jl = _arr(ep_joint_lim).astype(bool)
    col = _arr(ep_collision).astype(bool)

    print("=" * 72)
    print(
        f"Pure-IK reaching (redesigned task)  ({n_episodes} episodes, zero action, max {MAX_STEPS} steps)"
    )
    print("=" * 72)
    print(
        f"  initial goal-EE dist: mean {np.mean(_arr(ep_goal_dist)):.3f}  p90 {np.percentile(_arr(ep_goal_dist), 90):.3f}"
    )
    print(f"  success_10cm : {suc10.mean():.3f}")
    print(f"  success_5cm  : {suc5.mean():.3f}")
    print(
        f"  final EE dist: mean {final.mean():.3f}  p50 {np.percentile(final, 50):.3f}  p90 {np.percentile(final, 90):.3f}"
    )
    print(f"  min  EE dist : mean {mine.mean():.3f}  p50 {np.percentile(mine, 50):.3f}")
    tts_ok = tts[tts >= 0]
    if len(tts_ok):
        print(f"  time_to_success: mean {tts_ok.mean():.0f} steps ({tts_ok.mean() * 0.02:.2f}s)")
    else:
        print("  time_to_success: none")
    print(f"  joint-limit viol rate : {jl.mean():.3f}")
    print(f"  arm-base collision rate: {col.mean():.3f}")

    maxlen = max(len(c) for c in dist_curve)
    full = np.zeros(maxlen)
    cnt = np.zeros(maxlen)
    for curve in dist_curve:
        k = len(curve)
        full[:k] += curve
        cnt[:k] += 1
    with np.errstate(invalid="ignore"):
        mean_curve = full / np.maximum(cnt, 1)
    print("\n  mean EE distance over steps:")
    ticks = [t for t in list(range(0, maxlen, 50)) + [maxlen - 1] if t < maxlen]
    for t in ticks:
        print(f"    t={t * 0.02:4.1f}s  EE={mean_curve[t]:.3f}m")
    diff = np.diff(mean_curve)
    valid = cnt[1:] >= 1
    mono = float(np.mean(diff[valid])) if valid.any() else 0.0
    print(
        f"\n  mean d(EE)/step = {mono:.6f}  ({'steady decrease' if mono < 0 else 'NOT decreasing'})"
    )

    pass_ok = bool(suc10.mean() > PASS_SUCCESS and np.percentile(final, 50) < PASS_FINAL)
    print("\n" + "=" * 72)
    print(f"  PASS criteria: success_10cm > {PASS_SUCCESS}  AND  final EE p50 < {PASS_FINAL}")
    print(f"  RESULT: {'PASS — ready for PPO' if pass_ok else 'FAIL — keep fixing task design'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
