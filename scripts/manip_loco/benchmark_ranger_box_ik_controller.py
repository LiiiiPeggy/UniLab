"""IK target-controller ablation for RangerBoxReach — pure IK, no policy.

Run:  uv run scripts/manip_loco/benchmark_ranger_box_ik_controller.py

Compares three target-controller structures on the same local-EE task
(num_envs 16, zero action, no noise/latency/DR, 100 episodes):

  A  integrated   — OLD chase-current _ik_target integration (overshoots).
  B  q + gain*dq  — resolved-rate position target, no velocity damping.
  C  q + gain*(dq - kv*qdot) — resolved-rate + joint velocity damping (target).

Reports per mode: success_10cm / success_5cm, final EE p50, EE overshoot
(peak-to-peak swing), and mean |joint velocity|.

PASS criterion for the recommended mode C: success_10cm > 0.80.
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
MAX_STEPS = 500
SUCCESS_10 = 0.10
SUCCESS_05 = 0.05
PASS_SUCCESS = 0.80

_BASE_BOX_HALF = np.array([0.55, 0.38, 0.20], dtype=np.float64)
_BOX_CENTRE_OFFSET = np.array([-0.1262, 0.0, -0.0965], dtype=np.float64)

MODES = {
    "A integrated": ("integrated", 0.0),
    "B q+gain*dq": ("resolved_rate", 0.0),
    "C q+gain*(dq-kv*qdot)": ("resolved_rate_damped", 0.05),
}


def build_env(mode: str, kv: float):
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
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
                f"env.ik.controller_mode={mode}",
                f"env.ik.velocity_damping={kv}",
            ],
        )
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=NUM_ENVS, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _min_arm_signed_dist(env, ee_world) -> np.ndarray:
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


def run_mode(label: str, mode: str, kv: float) -> dict:
    env = build_env(mode, kv)

    ep_success10: list[bool] = []
    ep_success5: list[bool] = []
    ep_final: list[float] = []
    ep_swing: list[float] = []
    ep_vel: list[float] = []
    ep_jl: list[bool] = []
    ep_col: list[bool] = []

    n_episodes = 0
    while n_episodes < EPISODES_TARGET:
        env.reset(np.arange(NUM_ENVS))
        per10 = np.zeros(NUM_ENVS, dtype=bool)
        per5 = np.zeros(NUM_ENVS, dtype=bool)
        per_jl = np.zeros(NUM_ENVS, dtype=bool)
        per_col = np.zeros(NUM_ENVS, dtype=bool)
        d_min = np.full(NUM_ENVS, 1e9)
        d_max = np.full(NUM_ENVS, -1e9)
        vel_sum = np.zeros(NUM_ENVS)
        vel_cnt = np.zeros(NUM_ENVS)

        for _ in range(MAX_STEPS):
            env.step(np.zeros((NUM_ENVS, 9)))
            ee_local, _ = env.get_ee_local_pose()
            ee_world = env.armbase_pos_world + np_quat_apply_batched(
                env.armbase_quat_world, ee_local
            )
            d = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
            d_min = np.minimum(d_min, d)
            d_max = np.maximum(d_max, d)
            per10 |= d < SUCCESS_10
            per5 |= d < SUCCESS_05
            arm = env.get_arm_dof_pos()
            per_jl |= ((arm > env._arm_joint_upper) | (arm < env._arm_joint_lower)).any(axis=1)
            per_col |= _min_arm_signed_dist(env, ee_world) < 0.0
            vel_sum += np.abs(env.get_arm_dof_vel()).sum(axis=1)
            vel_cnt += 1
            if per10.all():
                break

        ee_local, _ = env.get_ee_local_pose()
        ee_world_f = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        d_final = np.linalg.norm(ee_world_f - env.world_ee_goal, axis=1)

        for i in range(NUM_ENVS):
            ep_success10.append(bool(per10[i]))
            ep_success5.append(bool(per5[i]))
            ep_final.append(float(d_final[i]))
            ep_swing.append(float(d_max[i] - d_min[i]))
            ep_vel.append(float(vel_sum[i] / max(vel_cnt[i], 1)))
            ep_jl.append(bool(per_jl[i]))
            ep_col.append(bool(per_col[i]))
        n_episodes += NUM_ENVS

    env.close()
    n = min(n_episodes, len(ep_success10))

    def _arr(name):
        return np.asarray(name, dtype=np.float64)[:n]

    suc10 = _arr(ep_success10).astype(bool)
    suc5 = _arr(ep_success5).astype(bool)
    final = _arr(ep_final)
    swing = _arr(ep_swing)
    vel = _arr(ep_vel)
    jl = _arr(ep_jl).astype(bool)
    col = _arr(ep_col).astype(bool)

    print("=" * 72)
    print(f"{label}  (n={n})")
    print("=" * 72)
    print(f"  success_10cm : {suc10.mean():.3f}")
    print(f"  success_5cm  : {suc5.mean():.3f}")
    print(f"  final EE p50 : {np.percentile(final, 50):.3f}   p90 {np.percentile(final, 90):.3f}")
    print(
        f"  EE overshoot : mean {swing.mean():.3f}   p50 {np.percentile(swing, 50):.3f}  (max-min swing)"
    )
    print(f"  mean |qdot|  : {vel.mean():.3f} rad/s")
    print(f"  joint-limit / collision rate: {jl.mean():.3f} / {col.mean():.3f}")
    ok = bool(suc10.mean() > PASS_SUCCESS)
    print(f"  -> {'PASS' if ok else 'FAIL'} (criterion success_10cm > {PASS_SUCCESS})")
    print()
    return {"success": suc10.mean(), "final": np.percentile(final, 50), "swing": swing.mean()}


def main() -> None:
    results = {}
    for label, (mode, kv) in MODES.items():
        results[label] = run_mode(label, mode, kv)
    print("=" * 72)
    print("SUMMARY")
    for label, r in results.items():
        print(
            f"  {label:28s}: success_10cm={r['success']:.3f}  final_p50={r['final']:.3f}  "
            f"swing={r['swing']:.3f}"
        )
    print("=" * 72)
    print("\n=== done ===")


if __name__ == "__main__":
    main()
