"""Search for a CR10 manipulation-ready pose — folded, well-conditioned.

Run:  uv run scripts/manip_loco/find_ranger_box_ready_pose.py

Randomly samples arm joint configs (within limits) and keeps those that are:
  - EE distance from armbase in [0.30, 0.60] m   (folded, near the base)
  - Jacobian condition number < 20                (well-conditioned IK)
  - joint-limit margin >= 0.2 rad from soft limits
  - no arm-base collision and no non-adjacent link self-collision

Outputs the top 20 candidates, then runs a short physics settle (hold the
candidate, 150 steps) on the top ones and reports the actual joint drift so
the most gravity-stable folded pose can be chosen.

The search is batched (256 envs) for speed.
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
    np_matrix_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
)
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")

N_BATCH = 256
N_BATCHES = 800  # 204 800 candidates
ARM_NAMES = [f"cr10_joint{j}" for j in range(1, 7)]

# Filters
DIST_LO, DIST_HI = 0.30, 0.60
COND_MAX = 20.0
LIMIT_MARGIN = 0.20  # rad from soft limit
BASE_CLEAR_MIN = 0.05  # m clearance to the base box

# Base box geometry (matches reach_env reward).
_BASE_BOX_HALF = np.array([0.55, 0.38, 0.20], dtype=np.float64)
_BOX_CENTRE_OFFSET = np.array([-0.1262, 0.0, -0.0965], dtype=np.float64)
# Conservative per-link bounding radii (m) for a light self-collision sphere
# test.  Chosen small (half of the smallest proxy dimension) so only genuine
# non-adjacent overlaps are flagged — folded links may legitimately come close.
LINK_R = np.array([0.05, 0.09, 0.08, 0.05, 0.05, 0.06], dtype=np.float64)


def build_env(num_envs: int):
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["task=ranger_box_reach/mujoco",
                                                       f"algo.num_envs={num_envs}",
                                                       "algo.max_iterations=10"])
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    env = create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override)
    env.set_autoreset(False)
    return env


def _arm_point_signed_dist(p_w, box_centre_w, base_quat_w):
    """Signed distance from world points to the base box in base frame."""
    p_local = np_quat_apply_batched(np_quat_conjugate_batched(base_quat_w), p_w - box_centre_w)
    q = np.abs(p_local) - _BASE_BOX_HALF
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def sample_batch(env, rng):
    """Sample one batch; return candidate metrics arrays (N_BATCH,).

    j1 is restricted to [-0.3, 0.3] so the arm points forward (natural base
    frame), and j2 to [-0.15, 0.15] because j2's gravity equilibrium is ~0:
    a ready pose with j2 far from 0 sags on j2 regardless of gravcomp.
    """
    lo = env._arm_joint_lower.copy()
    hi = env._arm_joint_upper.copy()
    lo[0], hi[0] = -0.30, 0.30
    lo[1], hi[1] = -0.15, 0.15
    q = rng.uniform(lo[None, :], hi[None, :], size=(N_BATCH, 6))
    env._backend.set_joint_qpos(ARM_NAMES, q)
    env._backend.set_joint_qvel(ARM_NAMES, np.zeros_like(q))
    env._backend.forward_sensors()

    ee_local, _ = env.get_ee_local_pose()
    dist = np.linalg.norm(ee_local, axis=1)

    # Jacobian condition (position block, armbase frame).
    jacp, _ = env._backend.get_site_jacobian_w(env._ee_site_id, env._arm_jacobian_dof_indices)
    ref_rot = np_matrix_from_quat(env._backend.get_sensor_data(env._cfg.sensor.arm_ref_world_quat))
    jacp_b = np.matmul(np.swapaxes(ref_rot, 1, 2), jacp)  # (N,3,6)
    sv = np.linalg.svd(jacp_b, compute_uv=False)
    cond = sv[:, 0] / np.maximum(sv[:, -1], 1e-9)

    # Joint limit margin (min distance to soft limits).
    soft_lo = env._arm_joint_lower + 0.05 * (env._arm_joint_upper - env._arm_joint_lower)
    soft_hi = env._arm_joint_upper - 0.05 * (env._arm_joint_upper - env._arm_joint_lower)
    margin = np.min(np.minimum(q - soft_lo, soft_hi - q), axis=1)

    # Arm-base clearance (min signed distance of links + EE to base box).
    base_quat_w = env.armbase_quat_world
    box_centre_w = env.armbase_pos_world + np_quat_apply_batched(
        base_quat_w, np.broadcast_to(_BOX_CENTRE_OFFSET, (N_BATCH, 3))
    )
    pts = [ee_local, env.armbase_pos_world + np_quat_apply_batched(base_quat_w, ee_local)]
    for name in env._cfg.sensor.link_pos:
        pts.append(env._backend.get_sensor_data(name))
    sd = np.stack([_arm_point_signed_dist(p, box_centre_w, base_quat_w) for p in pts], axis=1)
    base_clear = sd.min(axis=1)

    # Link self-collision (non-adjacent pairs, sphere approx on body origins).
    link_pos = np.stack([env._backend.get_sensor_data(name) for name in env._cfg.sensor.link_pos],
                        axis=1)  # (N,5,3) for Link2..6
    self_col = np.zeros(N_BATCH, dtype=bool)
    for i in range(5):
        for j in range(i + 2, 5):  # skip adjacent (i+1)
            d = np.linalg.norm(link_pos[:, i] - link_pos[:, j], axis=1)
            self_col |= d < (LINK_R[i + 1] + LINK_R[j + 1])
    return q, dist, cond, margin, base_clear, self_col


def settle_drift(env, q_candidate):
    """Hold the candidate as the IK target for 150 steps; report drift + collision.

    Returns (drift_joints, min_base_clear) — a negative min clearance means the
    arm physically collided with the base box during the settle.
    """
    env.reset(np.array([0]))
    env.init_state()
    env._backend.set_joint_qpos(ARM_NAMES, np.asarray(q_candidate, dtype=np.float64).reshape(1, 6))
    env._backend.set_joint_qvel(ARM_NAMES, np.zeros((1, 6)))
    env._ik_target[:] = q_candidate
    q_hist = np.zeros((150, 6))
    min_clear = 1e9
    for i in range(150):
        env.step(np.zeros((1, 9)))
        env._ik_target[:] = q_candidate
        q_hist[i] = env.get_arm_dof_pos()[0]
        ee_local, _ = env.get_ee_local_pose()
        ee_world = env.armbase_pos_world + np_quat_apply_batched(
            env.armbase_quat_world, ee_local
        )
        box_centre_w = env.armbase_pos_world + np_quat_apply_batched(
            env.armbase_quat_world, np.broadcast_to(_BOX_CENTRE_OFFSET, (1, 3))
        )
        pts = [ee_world] + [env._backend.get_sensor_data(n) for n in env._cfg.sensor.link_pos]
        sd = np.stack(
            [_arm_point_signed_dist(p, box_centre_w, env.armbase_quat_world) for p in pts], axis=1
        )
        min_clear = min(min_clear, float(sd.min()))
    drift = np.abs(q_hist[-80:].mean(axis=0) - q_candidate)
    return drift, min_clear


def main() -> None:
    env = build_env(N_BATCH)
    rng = np.random.default_rng(7)

    keep_q, keep_dist, keep_cond, keep_margin, keep_clear = [], [], [], [], []
    for _ in range(N_BATCHES):
        q, dist, cond, margin, base_clear, self_col = sample_batch(env, rng)
        ok = (
            (dist >= DIST_LO)
            & (dist <= DIST_HI)
            & (cond < COND_MAX)
            & (margin >= LIMIT_MARGIN)
            & (base_clear > BASE_CLEAR_MIN)
            & (~self_col)
        )
        if ok.any():
            keep_q.append(q[ok])
            keep_dist.append(dist[ok])
            keep_cond.append(cond[ok])
            keep_margin.append(margin[ok])
            keep_clear.append(base_clear[ok])
    env.close()

    if not keep_q:
        print("No candidates passed the mechanical filters.")
        return
    q = np.concatenate(keep_q)
    dist = np.concatenate(keep_dist)
    cond = np.concatenate(keep_cond)
    margin = np.concatenate(keep_margin)
    clear = np.concatenate(keep_clear)
    print(f"candidates passing filters: {len(q)}")

    # Score: prefer low condition, comfortable distance, ample margin.
    score = cond / 20.0 + np.abs(dist - 0.45) / 0.15 + (0.3 - np.minimum(margin, 0.3)) / 0.3
    order = np.argsort(score)[:20]
    print("\n=== top 20 candidates (by score) ===")
    print(f"{'#':>2} {'cond':>6} {'dist':>5} {'margin':>6} {'clear':>5}  "
          f"{'q1..q6':>40}  EE")
    for rank, i in enumerate(order):
        print(f"{rank + 1:2d} {cond[i]:6.1f} {dist[i]:5.2f} {margin[i]:6.2f} {clear[i]:5.2f}  "
              f"[{', '.join(f'{v:.3f}' for v in q[i])}]")

    # Physics settle for the top 10 to find gravity-stable folded poses.
    print("\n=== physics settle drift (top 10, 150 steps hold) ===")
    env = build_env(1)
    print(f"{'#':>2} {'cond':>6} {'drift j1..j6':>50} {'max':>6} {'minClear':>8}")
    best = None
    for rank, i in enumerate(order[:10]):
        drift, min_clear = settle_drift(env, q[i])
        mx = float(np.max(drift))
        coll = " COLLIDE" if min_clear < 0.0 else ""
        print(f"{rank + 1:2d} {cond[i]:6.1f} [{', '.join(f'{v:.3f}' for v in drift)}]  "
              f"{mx:6.3f} {min_clear:8.3f}{coll}")
        if (min_clear >= 0.0) and (best is None or mx < best[0]):
            best = (mx, rank, i)
    env.close()

    if best is not None:
        _, rank, i = best
        print(f"\nmost gravity-stable (collision-free): candidate #{rank + 1}  max drift {best[0]:.3f} rad")
        print("  q = " + str([round(float(v), 4) for v in q[i]]))
        print("  cond = %.1f  dist = %.2f  margin = %.2f  base_clear = %.2f"
              % (cond[i], dist[i], margin[i], clear[i]))
    else:
        print("\nno collision-free stable candidate — folded poses sag toward the "
              "extended equilibrium under the current physics.")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
