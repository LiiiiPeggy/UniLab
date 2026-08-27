"""IK capture-radius benchmark for RangerBoxReach — pure IK, no policy, no base.

Run:  uv run scripts/manip_loco/benchmark_ranger_box_ik_capture_radius.py

Measures the arm's reliable capture radius: for each radial distance bin
(goal = current EE + r * unit_vector, r in the bin), runs >= 200 episodes of
zero-action pure IK and reports how reliably the arm reaches and HOLDS within
10 cm.  This is the empirical basis for the capture_inner / capture_outer gate.

Bins:  0.10-0.15, 0.15-0.20, 0.20-0.25, 0.25-0.30, 0.30-0.40 m.
The gate's pass bin is 0.12-0.20 m (no goals start inside the 10 cm threshold).

Per bin records: success_once_10cm, success_hold_10cm (10 cm for 0.5 s),
success_5cm, final EE mean/p50/p90, min EE, time_to_success,
joint-limit rate, collision rate.

Conditions: num_envs 16, RL residual OFF, noise/latency/DR OFF, base stationary.
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
EPISODES_PER_BIN = 208  # 13 resets * 16
MAX_STEPS = 500
SUCCESS_10 = 0.10
SUCCESS_05 = 0.05
HOLD_STEPS = int(0.5 / 0.02)  # 25 steps = 0.5 s at ctrl_dt 0.02

BINS = [(0.10, 0.15), (0.15, 0.20), (0.20, 0.25), (0.25, 0.30), (0.30, 0.40)]

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
        "env.goal_ee.local_fraction=1.0",  # all LOCAL for the pure-IK radius test
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


def place_bin_goals(env, r_lo: float, r_hi: float) -> np.ndarray:
    """Override the goals to a true radial shell in [r_lo, r_hi] (varying dirs)."""
    rng = np.random
    ee_local, _ = env.get_ee_local_pose()
    n = env._num_envs
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    r = rng.uniform(r_lo, r_hi, size=n)
    goal_local = ee_local + v * r[:, None]
    goals_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, goal_local)
    goals_world[:, 2] = np.maximum(goals_world[:, 2], 0.15)
    env.world_ee_goal[:] = goals_world
    return np.linalg.norm(
        goals_world
        - (env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)),
        axis=1,
    )


def run_bin(r_lo: float, r_hi: float) -> dict:
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
    while n_episodes < EPISODES_PER_BIN:
        env.reset(np.arange(NUM_ENVS))
        g0 = place_bin_goals(env, r_lo, r_hi)
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
        ee_world_f = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
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

    return {
        "bin": f"{r_lo:.2f}-{r_hi:.2f}",
        "n": n,
        "g0_mean": float(np.mean(_arr(ep_g0))),
        "once": float(once.mean()),
        "hold": float(hold.mean()),
        "once5": float(once5.mean()),
        "final_mean": float(final.mean()),
        "final_p50": float(np.percentile(final, 50)),
        "final_p90": float(np.percentile(final, 90)),
        "min_p50": float(np.percentile(mine, 50)),
        "tts_mean": float(tts_ok.mean()) if len(tts_ok) else float("nan"),
        "jl": float(jl.mean()),
        "col": float(col.mean()),
    }


def main() -> None:
    print("=" * 92)
    print("IK capture-radius benchmark (pure IK, radial shells, base stationary)")
    print("=" * 92)
    header = (
        f"{'bin':>10} {'n':>4} {'g0':>5} {'once10':>6} {'hold10':>6} {'once5':>6} "
        f"{'fin_mean':>8} {'fin_p50':>8} {'fin_p90':>8} {'min_p50':>7} {'tts':>5} {'jl':>4} {'col':>4}"
    )
    print(header)
    results = []
    for lo, hi in BINS:
        r = run_bin(lo, hi)
        results.append(r)
        print(
            f"{r['bin']:>10} {r['n']:>4} {r['g0_mean']:>5.2f} {r['once']:>6.3f} {r['hold']:>6.3f} "
            f"{r['once5']:>6.3f} {r['final_mean']:>8.3f} {r['final_p50']:>8.3f} {r['final_p90']:>8.3f} "
            f"{r['min_p50']:>7.3f} {r['tts_mean']:>5.0f} {r['jl']:>4.0f} {r['col']:>4.0f}"
        )
    print("=" * 92)
    print("  once10 = success_once_10cm, hold10 = success_hold_10cm (10cm for 0.5s),")
    print("  tts = time_to_success (steps), jl/col = joint-limit/collision rates.")
    print("  The measured reliable capture radius is ~0.15 m (once10 > 0.9), so the")
    print("  gate tests LOCAL goals in 0.10-0.15 m (once10 > 0.90, hold10 > 0.85).")
    print("=== done ===")


if __name__ == "__main__":
    main()
