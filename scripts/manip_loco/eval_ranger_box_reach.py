"""Deterministic eval for RangerBoxReach — hard metrics, not just reward.

Usage:
  uv run python scripts/manip_loco/eval_ranger_box_reach.py \
      --load-run <run_dir> --num-envs 64 --episodes 200

Loads a checkpoint, runs the actor with deterministic MEAN actions (no
exploration noise), and reports success rates / EE distance / collision /
joint-limit / actuator-saturation metrics, split by reachable vs extended.

Output is printed and also written to <run_dir>/eval_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402
from unilab.training.sim2sim import policy_load_dim_guard  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")
TORQUE_SENSORS = [f"cr10_j{j}_torque" for j in range(1, 7)]
TORQUE_LIMITS = np.array([15, 50, 50, 25, 25, 25], dtype=np.float64)


def _build_env(num_envs: int):
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(config_name="config", overrides=["task=ranger_box_reach/mujoco"])
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-run", required=True)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--checkpoint", default="model_999.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, cfg = _build_env(args.num_envs)
    wrapped = RslRlVecEnvWrapper(env, device=device)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
    ckpt = Path(args.load_run) / args.checkpoint
    with policy_load_dim_guard(
        env_obs_dim=getattr(wrapped, "num_obs", None),
        env_action_dim=getattr(wrapped, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Accumulators, split by LOCAL vs EXTENDED goal type + all.
    stats = {"local": _new_stats(), "extended": _new_stats(), "all": _new_stats()}
    n_episodes = 0
    capture_outer = float(getattr(env._cfg.goal_ee, "capture_outer", 0.18))
    obs = wrapped.reset()[0]

    with torch.inference_mode():
        while n_episodes < args.episodes:
            obs = wrapped.reset()[0]
            ep_ee_dists = []
            ep_min_dist = 1e9
            ep_success_once = np.zeros(args.num_envs, dtype=bool)
            ep_success_hold = np.zeros(args.num_envs, dtype=bool)
            ep_joint_limit = np.zeros(args.num_envs, dtype=bool)
            ep_sat_count = np.zeros(args.num_envs)
            ep_steps = 0
            base_start = env._backend.get_base_pos()[:, :2].copy()
            base_hist: list[np.ndarray] = [base_start.copy()]

            # LOCAL vs EXTENDED from the sampler (set at reset).
            is_local = env._goal_is_local.copy()

            for _ in range(500):
                act = policy(obs)
                obs, rew, dones, infos = wrapped.step(act)
                ep_steps += 1
                base_hist.append(env._backend.get_base_pos()[:, :2].copy())
                ee_world = env.armbase_pos_world + np_quat_apply_batched(
                    env.armbase_quat_world, env.get_ee_local_pose()[0]
                )
                d = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
                ep_ee_dists.append(d.copy())
                ep_min_dist = np.minimum(ep_min_dist, d)
                ep_success_once = ep_success_once | (d < 0.10)
                ep_success_hold = ep_success_hold | env._success_hold.copy()
                ep_joint_limit = ep_joint_limit | _joint_limit_hit(env)
                forces = np.stack(
                    [env._backend.get_sensor_data(n)[:, 0] for n in TORQUE_SENSORS], axis=1
                )
                sat = np.abs(forces) / TORQUE_LIMITS > 0.95
                ep_sat_count += sat.sum(axis=1)
                if dones.any():
                    break

            base_end = env._backend.get_base_pos()[:, :2].copy()
            base_disp = np.linalg.norm(base_end - base_start, axis=1)
            # base path length (sum of per-step base displacement)
            base_arr = np.stack(base_hist, axis=0)  # (steps+1, N, 2)
            base_path = np.linalg.norm(np.diff(base_arr, axis=0), axis=2).sum(axis=0)
            ep_dist_final = ep_ee_dists[-1] if ep_ee_dists else np.zeros(args.num_envs)
            ep_dist_arr = np.stack(ep_ee_dists, axis=0)  # (steps, N)

            # Capture entry: first step where EE-goal <= capture_outer.
            in_capture = ep_dist_arr <= capture_outer
            capture_entry = np.full(args.num_envs, -1, dtype=np.int32)
            tts = np.full(args.num_envs, -1, dtype=np.int32)
            for i in range(args.num_envs):
                hit = np.nonzero(in_capture[:, i])[0]
                if hit.size:
                    capture_entry[i] = int(hit[0]) + 1
                succ = np.nonzero(ep_dist_arr[:, i] < 0.10)[0]
                if succ.size:
                    tts[i] = int(succ[0]) + 1

            for i in range(args.num_envs):
                key = "local" if is_local[i] else "extended"
                _accum(
                    stats[key],
                    ep_success_once[i],
                    ep_success_hold[i],
                    ep_dist_arr[:, i],
                    ep_dist_final[i],
                    ep_min_dist[i],
                    base_disp[i],
                    base_path[i],
                    capture_entry[i],
                    tts[i],
                    ep_joint_limit[i],
                    ep_sat_count[i],
                    ep_steps,
                )
                _accum(
                    stats["all"],
                    ep_success_once[i],
                    ep_success_hold[i],
                    ep_dist_arr[:, i],
                    ep_dist_final[i],
                    ep_min_dist[i],
                    base_disp[i],
                    base_path[i],
                    capture_entry[i],
                    tts[i],
                    ep_joint_limit[i],
                    ep_sat_count[i],
                    ep_steps,
                )
            n_episodes += args.num_envs

    _report(stats, args.num_envs)
    out = {k: _finalize(v) for k, v in stats.items()}
    out_path = Path(args.load_run) / "eval_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")
    env.close()


def _new_stats():
    return {
        "n": 0,
        "success_once": 0,
        "success_hold": 0,
        "dist_sum": 0.0,
        "dist_sq_sum": 0.0,
        "dist_p50": [],
        "dist_p90": [],
        "dist_final_sum": 0.0,
        "dist_min_sum": 0.0,
        "base_disp_sum": 0.0,
        "base_path_sum": 0.0,
        "capture_entry_steps": [],
        "capture_to_success_steps": [],
        "joint_limit": 0,
        "sat_sum": 0.0,
        "steps_sum": 0,
    }


def _accum(s, once, hold, dists, final, mind, base_disp, base_path, cap_entry, tts,
           joint_lim, sat, steps):
    s["n"] += 1
    s["success_once"] += int(once)
    s["success_hold"] += int(hold)
    s["dist_sum"] += float(np.mean(dists))
    s["dist_sq_sum"] += float(np.mean(dists**2))
    s["dist_p50"].append(float(np.percentile(dists, 50)))
    s["dist_p90"].append(float(np.percentile(dists, 90)))
    s["dist_final_sum"] += float(final)
    s["dist_min_sum"] += float(mind)
    s["base_disp_sum"] += float(base_disp)
    s["base_path_sum"] += float(base_path)
    if cap_entry >= 0:
        s["capture_entry_steps"].append(float(cap_entry))
    if cap_entry >= 0 and tts >= cap_entry:
        s["capture_to_success_steps"].append(float(tts - cap_entry))
    s["joint_limit"] += int(joint_lim)
    s["sat_sum"] += float(sat)
    s["steps_sum"] += int(steps)


def _finalize(s):
    n = max(s["n"], 1)
    entry_n = len(s["capture_entry_steps"])
    c2s = s["capture_to_success_steps"]
    return {
        "n": s["n"],
        "success_once_rate": s["success_once"] / n,
        "success_hold_rate": s["success_hold"] / n,
        "ee_dist_mean": s["dist_sum"] / n,
        "ee_dist_p50": float(np.mean(s["dist_p50"])),
        "ee_dist_p90": float(np.mean(s["dist_p90"])),
        "ee_dist_final": s["dist_final_sum"] / n,
        "ee_dist_min": s["dist_min_sum"] / n,
        "base_disp_mean": s["base_disp_sum"] / n,
        "base_path_mean": s["base_path_sum"] / n,
        "capture_entry_rate": entry_n / n,
        "time_to_capture_mean": float(np.mean(s["capture_entry_steps"])) if entry_n else float("nan"),
        "time_capture_to_success": float(np.mean(c2s)) if c2s else float("nan"),
        "joint_limit_rate": s["joint_limit"] / n,
        "actuator_sat_rate": s["sat_sum"] / max(s["steps_sum"], 1),
    }


def _joint_limit_hit(env):
    arm = env.get_arm_dof_pos()
    return ((arm > env._arm_joint_upper) | (arm < env._arm_joint_lower)).any(axis=1)


def _report(stats, num_envs):
    print(f"\n=== RangerBoxReach deterministic eval ({num_envs} envs) ===")
    for key in ("local", "extended", "all"):
        f = _finalize(stats[key])
        print(f"\n[{key}]  n={f['n']}")
        print(
            f"  success_once: {f['success_once_rate']:.3f}   "
            f"success_hold: {f['success_hold_rate']:.3f}"
        )
        print(
            f"  ee_dist  mean {f['ee_dist_mean']:.3f}  p50 {f['ee_dist_p50']:.3f}  "
            f"p90 {f['ee_dist_p90']:.3f}  final {f['ee_dist_final']:.3f}  "
            f"min {f['ee_dist_min']:.3f}"
        )
        print(
            f"  base_disp mean {f['base_disp_mean']:.3f} m   "
            f"base_path mean {f['base_path_mean']:.3f} m"
        )
        print(
            f"  capture_entry_rate {f['capture_entry_rate']:.3f}   "
            f"time_to_capture {f['time_to_capture_mean']:.1f} steps   "
            f"capture→success {f['time_capture_to_success']:.1f} steps"
        )
        print(
            f"  joint_limit_rate {f['joint_limit_rate']:.3f}   "
            f"actuator_sat_rate {f['actuator_sat_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
