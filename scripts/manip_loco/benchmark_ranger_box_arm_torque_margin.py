"""CR10 arm torque-margin benchmark — measure actuator headroom (diagnostic only).

Run:  uv run scripts/manip_loco/benchmark_ranger_box_arm_torque_margin.py

Run 3 found the ready pose statically saturates j3/j4/j5/j6 (±50/±25/±25/±25 N·m).
This records the required actuator torque / saturation fraction under four
motions, so the physics-limits question can be decided on evidence later:

  1. static  — hold q_ready, zero action
  2. base x  — slow base translation (0.2 m/s)
  3. base yaw — slow base yaw (0.2 rad/s)
  4. IK      — pure-IK reaching to a LOCAL goal

NOTE: this does NOT change force limits.  It only measures.
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

from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")
N_ENVS = 8
STEPS = 60
WARMUP = 15

TORQUE_SENSORS = [f"cr10_j{j}_torque" for j in range(1, 7)]
TORQUE_LIMITS = np.array([15.0, 50.0, 50.0, 25.0, 25.0, 25.0], dtype=np.float64)


def build_env():
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=ranger_box_reach/mujoco",
                f"algo.num_envs={N_ENVS}",
                "env.noise_config.level=0.0",
                "env.base_velocity_controller.enable_latency=false",
                "env.base_velocity_controller.enable_noise=false",
                "env.domain_rand.randomize_kp=false",
                "env.domain_rand.randomize_kd=false",
                "env.domain_rand.randomize_body_mass=false",
                "env.domain_rand.random_com=false",
                "env.domain_rand.randomize_dof_armature=false",
            ],
        )
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=N_ENVS, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _torque(env) -> np.ndarray:
    return np.stack(
        [env._backend.get_sensor_data(n)[:, 0] for n in TORQUE_SENSORS], axis=1
    )  # (N, 6)


def run_scene(env, action) -> dict:
    """Run ``STEPS`` with the given fixed action, return steady-state torque stats."""
    env.init_state()
    env.reset(np.arange(N_ENVS))
    forces = []
    for t in range(STEPS):
        env.step(np.broadcast_to(action, (N_ENVS, 9)))
        forces.append(_torque(env))
    f = np.asarray(forces)[WARMUP:]  # (T, N, 6)
    mean_abs = np.mean(np.abs(f), axis=(0, 1))  # (6,)
    p90_abs = np.percentile(np.abs(f), 90, axis=(0, 1))  # (6,)
    max_abs = np.max(np.abs(f), axis=(0, 1))  # (6,)
    sat_frac = np.mean(np.abs(f) / TORQUE_LIMITS[None, None, :] > 0.95, axis=(0, 1))  # (6,)
    return {
        "mean_abs": mean_abs,
        "p90_abs": p90_abs,
        "max_abs": max_abs,
        "sat_frac": sat_frac,
    }


def _fmt(scene: dict, label: str, rng_lim: str) -> None:
    m = scene["mean_abs"]
    s = scene["sat_frac"]
    print(f"  {label:<12} sat_frac[j3..j6] = {s[2]:5.2f} {s[3]:5.2f} {s[4]:5.2f} {s[5]:5.2f}"
          f"   | mean torque j1..j6 = {m.round(1).tolist()}")
    print(f"               limit        = {rng_lim}")


def main() -> None:
    env = build_env()
    zero = np.zeros(9, dtype=np.float64)
    base_x = zero.copy()
    base_x[0] = 0.2 / 1.5  # ≈0.2 m/s at action_scale_lin=1.5
    base_yaw = zero.copy()
    base_yaw[2] = 0.2 / 1.0  # ≈0.2 rad/s at action_scale_ang=1.0

    print("=" * 88)
    print("CR10 arm torque margin (diagnostic; force limits NOT changed)")
    print(f"  {N_ENVS} envs, {STEPS} steps (steady-state over last {STEPS - WARMUP}), "
          f"limits j1..j6 = {TORQUE_LIMITS.tolist()}")
    print("=" * 88)
    _fmt(run_scene(env, zero), "static q_ready", "15/50/50/25/25/25")
    _fmt(run_scene(env, base_x), "base 0.2m/s x", "15/50/50/25/25/25")
    _fmt(run_scene(env, base_yaw), "base 0.2rad/s yaw", "15/50/50/25/25/25")

    # Pure-IK reaching: LOCAL goals (all capture) + zero action → IK drives arm.
    env2 = build_env()
    for _ in range(60):
        env2.init_state()
        env2.reset(np.arange(N_ENVS))
        if env2._goal_is_local.all():
            break
    forces = []
    for t in range(STEPS):
        env2.step(np.zeros((N_ENVS, 9)))
        forces.append(_torque(env2))
    f = np.asarray(forces)[WARMUP:]
    mean_abs = np.mean(np.abs(f), axis=(0, 1))
    sat_frac = np.mean(np.abs(f) / TORQUE_LIMITS[None, None, :] > 0.95, axis=(0, 1))
    print(f"  {'IK reaching':<12} sat_frac[j3..j6] = {sat_frac[2]:5.2f} {sat_frac[3]:5.2f} "
          f"{sat_frac[4]:5.2f} {sat_frac[5]:5.2f}   | mean torque = {mean_abs.round(1).tolist()}")
    print("               limit        = 15/50/50/25/25/25")
    env.close()
    env2.close()
    print("=" * 88)
    print("  Static saturation was already known (Run 3).  The extra scenes show how much")
    print("  headroom base motion / reaching consumes, to decide limits on evidence.")
    print("=== done ===")


if __name__ == "__main__":
    main()
