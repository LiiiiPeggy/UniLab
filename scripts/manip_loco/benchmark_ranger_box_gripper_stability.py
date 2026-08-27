"""AG95 gripper stability benchmark — is the jitter from the passive linkage?

Run:  uv run scripts/manip_loco/benchmark_ranger_box_gripper_stability.py

Native MuJoCo, single env.  Base fixed, arm held at q_ready (keyframe ctrl),
IK/RL disabled, noise/latency/DR off.  For each (passive-joint damping,
gripper master target) pair, runs 1000 physics steps and records every gripper
joint's q / qdot.  Reports max |qdot|, peak-to-peak q, and late-phase
oscillation so we can pick (a) a stable fixed opening and (b) the smallest
damping that removes the chatter.

Damping sweep:  0, 0.01, 0.05, 0.1, 0.5  (all 8 gripper joints).
Target sweep:   master joint (gripper_finger1_joint, range [0, 0.65]) at
                0, 0.1, 0.2, 0.3, 0.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mujoco  # noqa: E402

MODEL = ROOT_DIR / "src/unilab/assets/robots/ranger_box/scene_flat.xml"

GRIPPER_JOINTS = [
    "gripper_finger1_joint",
    "gripper_finger2_joint",
    "gripper_finger1_finger_joint",
    "gripper_finger2_finger_joint",
    "gripper_finger1_inner_knuckle_joint",
    "gripper_finger2_inner_knuckle_joint",
    "gripper_finger1_finger_tip_joint",
    "gripper_finger2_finger_tip_joint",
]

DAMPINGS = [0.0, 0.01, 0.05, 0.1, 0.5]
TARGETS = [0.0, 0.1, 0.2, 0.3, 0.4]
STEPS = 1000
WARMUP = 200  # ignore initial transient when measuring steady-state oscillation

# Dynamic-perturbation scene: the arm is driven through a slow 0.05-rad
# sinusoidal swing around q_ready (grip target fixed at 0), mimicking what
# happens while the base/arm move during reaching.  Measures how much the
# passive gripper linkage chatters under arm motion.
PERT_AMPL = 0.05
PERT_FREQ = 1.0  # Hz
DYNAMIC_DAMPINGS = [0.0, 0.01, 0.05, 0.1]


def main() -> None:
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    assert m.nkey >= 1, "scene_flat.xml must ship the 'home' keyframe"

    mujoco.mj_resetDataKeyframe(m, d, 0)
    key_ctrl = d.ctrl.copy()
    key_qpos = d.qpos.copy()

    J = mujoco.mjtObj.mjOBJ_JOINT
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    # qpos slots lag dof slots by 1 after a freejoint (quat occupies 4 qpos
    # slots vs 3 qvel); qpos uses jnt_qposadr, qvel/damping use jnt_dofadr.
    jids = [mujoco.mj_name2id(m, J, j) for j in GRIPPER_JOINTS]
    qpos_ids = np.array([m.jnt_qposadr[j] for j in jids], dtype=np.int64)
    dof_ids = np.array([m.jnt_dofadr[j] for j in jids], dtype=np.int64)
    grip_act = mujoco.mj_name2id(m, A, "gripper_finger1_joint_act")

    print("=" * 96)
    print("AG95 gripper stability (native MuJoCo, base/arm fixed, IK/RL off)")
    print(f"  dt={m.opt.timestep:.4f}s  steps={STEPS}  joints={len(GRIPPER_JOINTS)}")
    print("=" * 96)
    hdr = (f"{'damp':>5} {'tgt':>4} {'max|qd|':>8} {'rms|qd|':>8} {'pp q':>7} "
           f"{'ss pp q':>8} {'ss rms qd':>9} {'fin q0':>6}")
    print(hdr)

    results = []
    for damp in DAMPINGS:
        for tgt in TARGETS:
            mujoco.mj_resetDataKeyframe(m, d, 0)
            d.qpos[:] = key_qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            m.dof_damping[dof_ids] = damp
            # Keep the arm at ready; drive only the gripper master.
            d.ctrl[:] = key_ctrl
            d.ctrl[grip_act] = tgt

            q_hist = np.zeros((STEPS, len(GRIPPER_JOINTS)))
            qd_hist = np.zeros((STEPS, len(GRIPPER_JOINTS)))
            for t in range(STEPS):
                mujoco.mj_step(m, d)
                q_hist[t] = d.qpos[qpos_ids]
                qd_hist[t] = d.qvel[dof_ids]

            q = q_hist[WARMUP:]
            qd = qd_hist[WARMUP:]
            pp_ss = float(np.max(q.max(axis=0) - q.min(axis=0)))
            rms_qd = float(np.sqrt(np.mean(qd**2)))
            max_qd = float(np.abs(qd_hist).max())
            pp_all = float(np.max(q_hist.max(axis=0) - q_hist.min(axis=0)))
            fin0 = float(q_hist[-1, 0])
            results.append(
                (damp, tgt, max_qd, rms_qd, pp_all, pp_ss, rms_qd, fin0)
            )
            print(
                f"{damp:>5.2f} {tgt:>4.1f} {max_qd:>8.3f} {rms_qd:>8.4f} {pp_all:>7.4f} "
                f"{pp_ss:>8.4f} {rms_qd:>9.4f} {fin0:>6.3f}"
            )

    print("=" * 96)
    print("  max|qd| = worst instantaneous joint speed (rad/s) over all gripper joints,")
    print("  pp q    = peak-to-peak q over the whole run,  ss = steady-state (post-warmup).")
    print("  A stable fixed opening has small pp_ss, small rms_qd, and fin_q0 ≈ target.")

    # Best stable candidate: smallest steady-state oscillation with low damping.
    best = min(results, key=lambda r: (r[5] + 10 * r[6]))
    print(f"\n  best static (damp={best[0]:.2f}, target={best[1]:.2f}): "
          f"ss_pp_q={best[5]:.4f} rad, rms_qd={best[6]:.4f} rad/s")

    # ── Dynamic scene: arm swinging, grip target fixed at 0 ──────────────────
    print("\n" + "-" * 96)
    print("Arm-swing perturbation (base fixed, arm ±0.05 rad @ 1 Hz, grip target=0):")
    hdr2 = (f"{'damp':>5} {'grip_rms_qd':>12} {'grip_max_qd':>12} {'grip_pp_q':>9}")
    print(hdr2)
    for damp in DYNAMIC_DAMPINGS:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[:] = key_qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        m.dof_damping[dof_ids] = damp
        d.ctrl[:] = key_ctrl
        d.ctrl[grip_act] = 0.0

        q_hist = np.zeros((STEPS, len(GRIPPER_JOINTS)))
        qd_hist = np.zeros((STEPS, len(GRIPPER_JOINTS)))
        for t in range(STEPS):
            # slow sinusoidal swing on all 6 arm joints around ready
            d.ctrl[:6] = key_ctrl[:6] + PERT_AMPL * np.sin(
                2 * np.pi * PERT_FREQ * m.opt.timestep * t
            )
            mujoco.mj_step(m, d)
            q_hist[t] = d.qpos[qpos_ids]
            qd_hist[t] = d.qvel[dof_ids]

        q = q_hist[WARMUP:]
        qd = qd_hist[WARMUP:]
        rms_qd = float(np.sqrt(np.mean(qd**2)))
        max_qd = float(np.abs(qd).max())
        pp_q = float(np.max(q.max(axis=0) - q.min(axis=0)))
        print(f"{damp:>5.2f} {rms_qd:>12.4f} {max_qd:>12.3f} {pp_q:>9.4f}")

    print("=== done ===")


if __name__ == "__main__":
    main()
