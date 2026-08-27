"""Gravity-settle calibration for RangerBoxReach — deterministic, no PPO/IK/policy.

Run:  uv run scripts/manip_loco/calibrate_ranger_box_settle.py

Phase 1 — measure the CURRENT physics equilibrium under position-hold at home:
    reset at home, force the persistent IK target to home every step
    (so arm_ctrl == home, bypassing the anti-windup anchor), run 600 steps.
    The settled joint config q* is where kp*(home - q*) == residual gravity.

Phase 2 — validate the proposed fix:  reset AT q*, still hold target = home.
    If the arm stays at q* (drift < threshold), then moving only the reset
    keyframe to q* makes the arm hold still with zero policy.

Phase 3 — validate the alternative "move home too":  reset AT q*, target = q*.
    Shows whether moving the controller target also moves the resting pose.

Phase 4 — gravcomp ablation: repeat Phase 1 with gravcomp forced to 0, to
    quantify how much of the sag gravcomp actually removes at steady state.

Output: per-joint  initial / settled / error / |final velocity| / torque /
        saturation flag, plus the suggested new keyframe qpos for Phase 2.
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

# Current home (default_angles[:6]).
HOME = np.array([0.0, -0.3, 0.75, 0.0, 0.45, 0.0], dtype=np.float64)
ARM_NAMES = [f"cr10_joint{j}" for j in range(1, 7)]
TORQUE_SENSORS = [f"cr10_j{j}_torque" for j in range(1, 7)]
TORQUE_LIMITS = np.array([15.0, 50.0, 50.0, 25.0, 25.0, 25.0], dtype=np.float64)
# Joint convergence criterion: |q - target| steady within this (rad).
DRIFT_OK = 0.02


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


def _make_temp_scene(gravcomp: float) -> str:
    """Create a temp asset dir with robot.xml gravcomp overridden; return scene path."""
    tmp = Path(tempfile.mkdtemp(prefix="rbx_settle_"))
    robot = (ASSET_DIR / "robot.xml").read_text()
    robot = robot.replace('gravcomp="1"', f'gravcomp="{gravcomp}"')
    (tmp / "robot.xml").write_text(robot)
    shutil.copy(ASSET_DIR / "scene_flat.xml", tmp / "scene_flat.xml")
    os.symlink(ASSET_DIR / "meshes", tmp / "meshes")
    return str(tmp / "scene_flat.xml")


def build_env(scene_path: str | None = None):
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=_base_overrides())
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    if scene_path is not None:
        env_cfg_override["scene"] = SceneCfg(model_file=scene_path)
    env = create_env(cfg, num_envs=1, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def neutralize_goal(env) -> None:
    """Put the goal exactly at the current EE world position (dq_ik ~ 0)."""
    ee_local, _ = env.get_ee_local_pose()
    ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
    env.world_ee_goal[:] = ee_world


def run_hold(
    env, target: np.ndarray, *, n_steps: int = 600, reset_arm: np.ndarray | None = None
) -> dict:
    env.reset(np.array([0]))
    env.init_state()
    neutralize_goal(env)
    if reset_arm is not None:
        env._backend.set_joint_qpos(
            ARM_NAMES, np.asarray(reset_arm, dtype=np.float64).reshape(1, 6)
        )
        env._backend.set_joint_qvel(ARM_NAMES, np.zeros((1, 6), dtype=np.float64))
    # Target must be in place before the FIRST physics step too.
    env._ik_target[:] = target
    q_hist = np.zeros((n_steps, 6), dtype=np.float64)
    qvel_hist = np.zeros((n_steps, 6), dtype=np.float64)
    torque_hist = np.zeros((n_steps, 6), dtype=np.float64)
    for i in range(n_steps):
        env.step(np.zeros((1, 9)))
        env._ik_target[:] = target
        q_hist[i] = env.get_arm_dof_pos()[0]
        qvel_hist[i] = env.get_arm_dof_vel()[0]
        torque_hist[i] = np.array(
            [env._backend.get_sensor_data(n)[0, 0] for n in TORQUE_SENSORS], dtype=np.float64
        )
    return {
        "q_hist": q_hist,
        "qvel": qvel_hist,
        "torque": torque_hist,
        "q_final": q_hist[-100:].mean(axis=0),
    }


def run_idle(env, *, n_steps: int = 600, reset_arm: np.ndarray, default: np.ndarray) -> dict:
    """Simulate the REAL config the policy sees at reset: arm placed at
    reset_arm, default_arm_angles = default, zero action, and the controller
    left to run naturally (anti-windup + home-return active, NO _ik_target
    forcing). Reports arm drift + EE world drift over the episode."""
    env._default_arm_angles[:] = default
    env.reset(np.array([0]))
    env.init_state()
    neutralize_goal(env)
    env._backend.set_joint_qpos(ARM_NAMES, np.asarray(reset_arm, dtype=np.float64).reshape(1, 6))
    env._backend.set_joint_qvel(ARM_NAMES, np.zeros((1, 6), dtype=np.float64))
    env._ik_target[:] = default
    q_hist = np.zeros((n_steps, 6), dtype=np.float64)
    ee_hist = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(n_steps):
        env.step(np.zeros((1, 9)))
        q_hist[i] = env.get_arm_dof_pos()[0]
        ee_local, _ = env.get_ee_local_pose()
        ee_world = env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)
        ee_hist[i] = ee_world[0]
    return {"q_hist": q_hist, "ee_hist": ee_hist, "q_final": q_hist[-100:].mean(axis=0)}


def _fmt(arr: np.ndarray) -> str:
    return "[" + " ".join(f"{v:8.4f}" for v in arr) + "]"


def report_phase(title: str, r: dict, *, ref: np.ndarray, target: np.ndarray) -> None:
    q_final = r["q_final"]
    q_last = r["q_hist"][-200:]
    osc = q_last.max(axis=0) - q_last.min(axis=0)
    drift = q_final - ref
    max_drift = np.abs(q_last - ref).max(axis=0)
    v_final = np.abs(r["qvel"][-100:]).mean(axis=0)
    torque = np.abs(r["torque"][-100:]).mean(axis=0)
    sat = (torque / TORQUE_LIMITS) > 0.95
    print(f"\n=== {title} ===")
    print("  joint       j1      j2      j3      j4      j5      j6")
    print(f"  target   {_fmt(target)}")
    print(f"  ref      {_fmt(ref)}")
    print(f"  settled  {_fmt(q_final)}")
    print(f"  drift    {_fmt(drift)}   (settled - ref)")
    print(f"  maxdrift {_fmt(max_drift)}   (peak |q - ref|, last 200 steps)")
    print(f"  osc      {_fmt(osc)}")
    print(f"  |vel|    {_fmt(v_final)}")
    print(f"  torque   {_fmt(torque)}")
    print(f"  saturated {sat.astype(int)}")
    ok = np.all(max_drift < DRIFT_OK)
    print(f"  STABLE {'OK' if ok else 'FAIL'}  (maxdrift < {DRIFT_OK})")


def main() -> None:
    # ── Phase 1: current keyframe, reset=home, target=home → measure q* ──
    env = build_env()
    p1 = run_hold(env, HOME, reset_arm=None)
    env.close()
    q_star = p1["q_final"]
    report_phase("Phase 1  reset=home  target=home  (CURRENT behavior)", p1, ref=HOME, target=HOME)
    print(f"\n  settled config q* = {_fmt(q_star)}   (this is the natural hold pose)")
    print(f"  settled error     = {_fmt(q_star - HOME)}")

    # ── Phase 2: reset=q*, target=home (proposed: move keyframe only) ──
    p2 = run_hold(env, HOME, reset_arm=q_star)
    env.close()
    report_phase(
        "Phase 2  reset=q*  target=home  (move keyframe ONLY)", p2, ref=q_star, target=HOME
    )

    # ── Phase 3: reset=q*, target=q* (alternative: also move default_angles) ──
    p3 = run_hold(env, q_star, reset_arm=q_star)
    env.close()
    report_phase(
        "Phase 3  reset=q*  target=q*  (also move default_angles)", p3, ref=q_star, target=q_star
    )

    # ── Phase 4: gravcomp off, reset=home, target=home (quantify gravcomp) ──
    scene0 = _make_temp_scene(gravcomp=0.0)
    env0 = build_env(scene0)
    p0 = run_hold(env0, HOME)
    env0.close()
    shutil.rmtree(Path(scene0).parent, ignore_errors=True)
    report_phase("Phase 4  gravcomp=0  reset=home  target=home", p0, ref=HOME, target=HOME)
    print("\n  gravcomp removes (settle error):")
    for j, name in enumerate(["j1", "j2", "j3", "j4", "j5", "j6"]):
        with_gc = q_star[j] - HOME[j]
        without_gc = p0["q_final"][j] - HOME[j]
        frac = (1 - with_gc / without_gc) * 100 if abs(without_gc) > 1e-6 else 0.0
        print(f"    {name}: without {without_gc:+.4f}  with {with_gc:+.4f}  reduced {frac:5.1f}%")

    # ── Phase 5/6: real-config idle (natural controller, no target forcing) ──
    # These simulate what the policy actually sees at reset with the proposed
    # configs:  reset=q*, default_angles = q*  vs  reset=q*, default_angles=home.
    print("\n=== Real-config idle (controller runs naturally, zero action) ===")
    for label, default in (("default=q*  (reset=q*)", q_star), ("default=home (reset=q*)", HOME)):
        env = build_env()
        r5 = run_idle(env, reset_arm=q_star, default=default)
        env.close()
        q_final = r5["q_final"]
        q_last = r5["q_hist"][-200:]
        osc = q_last.max(axis=0) - q_last.min(axis=0)
        drift = q_final - q_star
        maxdrift = np.abs(q_last - q_star).max(axis=0)
        ee_start = r5["ee_hist"][0]
        ee_end = r5["ee_hist"][-1]
        ee_drift = np.linalg.norm(ee_end - ee_start)
        print(f"\n[{label}]")
        print(f"  arm drift   {_fmt(drift)}")
        print(f"  arm maxdrift {_fmt(maxdrift)}")
        print(f"  osc         {_fmt(osc)}")
        print(f"  EE drift (start→end, 12s) = {ee_drift:.4f} m   (threshold 0.10)")
        ok = bool(np.all(maxdrift < DRIFT_OK) and ee_drift < 0.03)
        print(f"  STABLE {'OK' if ok else 'FAIL'}  (arm maxdrift < {DRIFT_OK} AND EE drift < 0.03)")

    # ── Suggested keyframe update for Phase 2 (move ONLY the reset pose) ──
    print("\n=== Suggested keyframe update (Phase 2) ===")
    print("  arm qpos (q*)      = " + _fmt(q_star))
    print("  keyframe ctrl KEEP = " + _fmt(HOME) + "   (controller target stays home)")
    print(
        "  → reset the arm AT the settled pose while the controller still "
        "holds toward home: kp*(home - q*) exactly balances gravity."
    )
    print("\n=== done ===")


if __name__ == "__main__":
    main()
