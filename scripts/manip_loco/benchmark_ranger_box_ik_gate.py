"""Pure-IK capture gate for RangerBoxReach — the Run-3 GO/NO-GO gate.

Run:  uv run scripts/manip_loco/benchmark_ranger_box_ik_gate.py

Uses the env's REAL local goal sampler (local_fraction=1.0, local_radius_range
[0.10, 0.15], with the IK-feasibility filter), zero action, base stationary,
no noise/latency/DR.  200 episodes.  The radius range is the measured reliable
capture radius (see benchmark_ranger_box_ik_capture_radius.py).

Answers: "once the arm is inside its reliable capture region, can it complete
the final reaching alone?"

PASS criteria (user-specified):
  success_once_10cm > 0.90
  success_hold_10cm > 0.85
  collision rate = 0
  joint-limit violation ~ 0
Also reports success_5cm, final p50/p90.
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
EPISODES_TARGET = 208
MAX_STEPS = 500
SUCCESS_10 = 0.10
SUCCESS_05 = 0.05
HOLD_STEPS = int(0.5 / 0.02)

PASS_ONCE = 0.90
PASS_HOLD = 0.85

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
        "env.goal_ee.local_fraction=1.0",  # all LOCAL (capture-region) goals
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
    ep_once = []
    ep_hold = []
    ep_5 = []
    ep_final = []
    ep_min = []
    ep_tts = []
    ep_jl = []
    ep_col = []
    ep_g0 = []

    n_episodes = 0
    while n_episodes < EPISODES_TARGET:
        env.reset(np.arange(NUM_ENVS))
        ee_local, _ = env.get_ee_local_pose()
        ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        g0 = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
        once = np.zeros(NUM_ENVS, dtype=bool)
        hold = np.zeros(NUM_ENVS, dtype=bool)
        once5 = np.zeros(NUM_ENVS, dtype=bool)
        hold_timer = np.zeros(NUM_ENVS, dtype=np.int32)
        tts = np.zeros(NUM_ENVS, dtype=np.int32)
        jl = np.zeros(NUM_ENVS, dtype=bool)
        col = np.zeros(NUM_ENVS, dtype=bool)
        dmin = np.full(NUM_ENVS, 1e9)

        for step in range(MAX_STEPS):
            env.step(np.zeros((NUM_ENVS, 9)))
            ee_local, _ = env.get_ee_local_pose()
            ee_world = env.armbase_pos_world + np_quat_apply_batched(
                env.armbase_quat_world, ee_local
            )
            d = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
            dmin = np.minimum(dmin, d)
            within = d < SUCCESS_10
            new_once = (~once) & within
            once |= within
            once5 |= d < SUCCESS_05
            tts = np.where(new_once & (tts == 0), step + 1, tts)
            hold_timer = np.where(within, hold_timer + 1, 0)
            hold |= hold_timer >= HOLD_STEPS
            arm = env.get_arm_dof_pos()
            jl |= ((arm > env._arm_joint_upper) | (arm < env._arm_joint_lower)).any(axis=1)
            col |= _min_arm_signed_dist(env, ee_world) < 0.0
            if np.all(hold):  # all envs held → done (never break on once only)
                break

        ee_local, _ = env.get_ee_local_pose()
        ee_world_f = env.armbase_pos_world + np_quat_apply_batched(
            env.armbase_quat_world, ee_local
        )
        d_final = np.linalg.norm(ee_world_f - env.world_ee_goal, axis=1)

        for i in range(NUM_ENVS):
            ep_g0.append(float(g0[i]))
            ep_once.append(bool(once[i]))
            ep_hold.append(bool(hold[i]))
            ep_5.append(bool(once5[i]))
            ep_final.append(float(d_final[i]))
            ep_min.append(float(dmin[i]))
            ep_tts.append(int(tts[i]) if once[i] else -1)
            ep_jl.append(bool(jl[i]))
            ep_col.append(bool(col[i]))
        n_episodes += NUM_ENVS

    env.close()
    n = min(n_episodes, len(ep_once))

    def _arr(x):
        return np.asarray(x, dtype=np.float64)[:n]

    once = _arr(ep_once).astype(bool)
    hold = _arr(ep_hold).astype(bool)
    once5 = _arr(ep_5).astype(bool)
    final = _arr(ep_final)
    mine = _arr(ep_min)
    tts = np.asarray(ep_tts, dtype=np.int64)[:n]
    jl = _arr(ep_jl).astype(bool)
    col = _arr(ep_col).astype(bool)
    tts_ok = tts[tts >= 0]

    print("=" * 72)
    print(f"Pure-IK capture gate  (local goals {_arr(ep_g0).mean():.3f} m mean, "
          f"{n} episodes, zero action)")
    print("=" * 72)
    print(f"  success_once_10cm : {once.mean():.3f}   (PASS > {PASS_ONCE})")
    print(f"  success_hold_10cm : {hold.mean():.3f}   (PASS > {PASS_HOLD})")
    print(f"  success_5cm       : {once5.mean():.3f}")
    print(f"  final EE   p50    : {np.percentile(final, 50):.3f}   p90 {np.percentile(final, 90):.3f}")
    print(f"  min EE     p50    : {np.percentile(mine, 50):.3f}")
    if len(tts_ok):
        print(f"  time_to_success   : mean {tts_ok.mean():.0f} steps ({tts_ok.mean()*0.02:.2f}s)")
    else:
        print("  time_to_success   : none")
    print(f"  joint-limit rate  : {jl.mean():.3f}")
    print(f"  collision rate    : {col.mean():.3f}")
    pass_ok = bool(once.mean() > PASS_ONCE and hold.mean() > PASS_HOLD and col.mean() == 0.0 and jl.mean() < 0.01)
    print("=" * 72)
    print(f"  GATE: {'PASS — local arm controller qualified' if pass_ok else 'FAIL — not qualified'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
