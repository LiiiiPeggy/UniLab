"""CR10 passive-stability benchmark — gravcomp / joint-damping / actuator-kd ablation.

Run:  uv run python scripts/manip_loco/benchmark_ranger_box_physics.py

Pure physics test: base fixed, arm at home, NO policy / IK / noise / latency /
DR.  Runs 500 steps and reports per-joint overshoot, settling time,
oscillation amplitude, steady-state error, and stability.

Sweeps:
  gravcomp: 1.0 / 0.8 / 0.5 / 0.0
  joint damping: 1 / 5 / 10 / 20 N·m·s/rad (gravcomp=1)
  actuator kd: 1x / 2x / 4x (gravcomp=1, damping=0)

Uses temporary modified copies of robot.xml + scene_flat.xml — the committed
assets are NOT touched.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402

from unilab.base.scene import SceneCfg  # noqa: E402
from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402

ASSET_DIR = ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "ranger_box"
CONF_DIR = str(ROOT_DIR / "conf" / "ppo")
HOME = np.array([0.0, -0.3, 0.75, 0.0, 0.45, 0.0])
TORQUE_SENSORS = [f"cr10_j{j}_torque" for j in range(1, 7)]


def _base_overrides() -> list[str]:
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


def _make_temp_scene(gravcomp: float | None = None, damping: float | None = None) -> str:
    """Create a temp asset dir with modified robot.xml; return scene path."""
    tmp = Path(tempfile.mkdtemp(prefix="rbx_phys_"))
    robot = (ASSET_DIR / "robot.xml").read_text()
    if gravcomp is not None:
        robot = robot.replace('gravcomp="1"', f'gravcomp="{gravcomp}"')
    if damping is not None:
        for j in range(1, 7):
            robot = robot.replace(
                f'<joint name="cr10_joint{j}" pos',
                f'<joint name="cr10_joint{j}" damping="{damping}" pos',
            )
    (tmp / "robot.xml").write_text(robot)
    shutil.copy(ASSET_DIR / "scene_flat.xml", tmp / "scene_flat.xml")
    os.symlink(ASSET_DIR / "meshes", tmp / "meshes")
    return str(tmp / "scene_flat.xml")


def _build_env(scene_path: str, kd_mult: float = 1.0):
    ensure_registries()
    overrides = _base_overrides()
    if kd_mult != 1.0:
        base_kd = [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
        new_kd = [f"{v * kd_mult:.2f}" for v in base_kd]
        overrides.append(f"env.control_config.arm_kd=[{','.join(new_kd)}]")
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=overrides)
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env_cfg_override["scene"] = SceneCfg(model_file=scene_path)
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _run_stability(env, n_steps: int = 500, hold_target: bool = False) -> dict:
    """Run passive hold and return per-joint stability metrics.

    If ``hold_target`` is True, the persistent IK target is forced back to home
    every step — this isolates PURE physics (can the position actuators hold
    home?) from controller target drift.
    """
    env.reset(np.array([0]))
    env.init_state()
    # Critical: neutralise IK — put the goal exactly at the current EE world
    # position so dq_ik ≈ 0 and the arm simply holds home (pure passive test).
    ee_local, _ = env.get_ee_local_pose()
    ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
    env.world_ee_goal[:] = ee_world
    q_hist = np.zeros((n_steps, 6))
    qvel_hist = np.zeros((n_steps, 6))
    torque_hist = np.zeros((n_steps, 6))
    for i in range(n_steps):
        env.step(np.zeros((1, 9)))
        if hold_target:
            env._ik_target[:] = env._default_arm_angles
        q_hist[i] = env.get_arm_dof_pos()[0]
        qvel_hist[i] = env.get_arm_dof_vel()[0]
        torque_hist[i] = np.array([env._backend.get_sensor_data(n)[0, 0] for n in TORQUE_SENSORS])
    err = np.abs(q_hist - HOME)
    # metrics per joint
    overshoot = err.max(axis=0)
    final_err = err[-1]
    # steady-state error = mean |err| over last 100 steps
    ss_err = err[-100:].mean(axis=0)
    # oscillation amplitude = peak-to-peak over last 200 steps
    osc = q_hist[-200:].max(axis=0) - q_hist[-200:].min(axis=0)
    # settling time: first step after which |err| stays < 0.05 rad
    settle = np.full(6, n_steps)
    for j in range(6):
        for t in range(n_steps):
            if np.all(err[t:, j] < 0.05):
                settle[j] = t
                break
    # "stable" = NOT oscillating (peak-to-peak over last 200 steps < 0.1 rad).
    # A static offset from home (gravcomp sag) is tolerable — the RL policy
    # compensates it with its instantaneous residual.  Oscillation is fatal.
    stable = osc < 0.1
    return {
        "overshoot": overshoot,
        "settle": settle,
        "osc": osc,
        "ss_err": ss_err,
        "final_err": final_err,
        "stable": stable,
        "q_hist": q_hist,
    }


def _fmt_metrics(r: dict, j: int) -> str:
    return (
        f"over={r['overshoot'][j]:7.3f}  settle={r['settle'][j]:4d}  "
        f"osc={r['osc'][j]:7.3f}  ss_err={r['ss_err'][j]:6.3f}  "
        f"{'stable' if r['stable'][j] else 'UNSTABLE'}"
    )


def gravcomp_sweep():
    print("=" * 78)
    print("gravcomp ablation (kp/kd/damping identical, 500 steps, home hold)")
    print("=" * 78)
    for gc in [1.0, 0.8, 0.5, 0.0]:
        scene = _make_temp_scene(gravcomp=gc)
        env = _build_env(scene)
        r = _run_stability(env, hold_target=True)
        env.close()
        shutil.rmtree(Path(scene).parent, ignore_errors=True)
        print(f"\n[gravcomp={gc}]")
        for j, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
            print(f"  {name}: {_fmt_metrics(r, j)}")


def damping_sweep():
    print("\n" + "=" * 78)
    print("joint damping sweep (gravcomp=1, actuator kd unchanged)")
    print("=" * 78)
    for d in [1.0, 5.0, 10.0, 20.0]:
        scene = _make_temp_scene(damping=d)
        env = _build_env(scene)
        r = _run_stability(env, hold_target=True)
        env.close()
        shutil.rmtree(Path(scene).parent, ignore_errors=True)
        print(f"\n[damping={d}]")
        for j, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
            print(f"  {name}: {_fmt_metrics(r, j)}")


def kd_sweep():
    print("\n" + "=" * 78)
    print("actuator kd sweep (gravcomp=1, joint damping=0)")
    print("=" * 78)
    for mult in [1.0, 2.0, 4.0]:
        scene = _make_temp_scene(gravcomp=1.0)
        env = _build_env(scene, kd_mult=mult)
        r = _run_stability(env, hold_target=True)
        env.close()
        shutil.rmtree(Path(scene).parent, ignore_errors=True)
        print(f"\n[kd x{mult}]")
        for j, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
            print(f"  {name}: {_fmt_metrics(r, j)}")


def main() -> None:
    gravcomp_sweep()
    damping_sweep()
    kd_sweep()
    print("\n=== done ===")


if __name__ == "__main__":
    main()
