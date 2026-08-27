"""EXTENDED-goal vertical-geometry benchmark for RangerBoxReach.

Usage:
  uv run scripts/manip_loco/benchmark_ranger_box_extended_geometry.py \
      --load-run logs/rsl_rl_ppo/RangerBoxReach/<run_dir> \
      --num-envs 16 --rounds 63 --checkpoints 150 199

Question answered (data first — sampler changes are gated on this):
  Run 4A sampled EXTENDED goals as TRUE 3D radial vectors (unit-sphere
  direction x r~U(extended_radius_range)) from the current EE in the armbase
  frame. The wheeled base under SE(2) lock can only change x/y/yaw — its
  height is constant. A goal whose |delta_z| exceeds ``capture_outer`` can
  therefore never be brought into the capture ring by base translation alone
  unless the arm itself closes the vertical gap.

  NOTE: this benchmark must run against the PRE-planar sampler (git history /
  run 2026-08-27_18-36-07_mujoco era). The committed sampler now bounds dz to
  extended_z_range, which drives the impossible fraction to ~0 by construction.

Per episode we record, at reset:
  ready_ee_z                  EE height at the ready/init pose (arm uncommitted)
  vertical_floor              = |goal_z - ready_ee_z|   (base-only lower bound)
  impossible                  = vertical_floor >= capture_outer

and during rollout: hard success metrics (identical definitions to
eval_ranger_box_reach) plus Task-17/18 drive diagnostics — goal-vs-velocity
heading agreement, forward/lateral velocity split, base action magnitudes,
grouped by hold10 outcome.

All goals are forced EXTENDED via ``env.goal_ee.local_fraction=0.0``.

Writes <load_run>/extended_geometry_iter{N}.json and prints a per-|dz|-bin table:
  bins (m): [0,0.05) [0.05,0.10) [0.10,0.15) [0.15,0.20) [0.20,0.30) [0.30,+)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_ranger_box_reach import (  # noqa: E402
    CLEAN_OVERRIDES,
    CONF_DIR,
    HOLD_STEPS,
    MAX_EP_STEPS,
    ROOT_DIR,
    SUCCESS_10,
    _ee_world,
)
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402
from unilab.training.sim2sim import policy_load_dim_guard  # noqa: E402

DZ_BIN_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, math.inf]
DZ_BIN_LABELS = ["0.00-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", ">0.30"]
SERIES_CAP_DEFAULT = 24  # dump full per-step time-series for the first N episodes only
MOVE_SPEED_EPS = 0.08  # m/s; below this the heading direction is noise


def build_ext_env(num_envs: int):
    """Same clean setup as eval_ranger_box_reach but ALL goals EXTENDED."""
    ensure_registries()
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=ranger_box_reach/mujoco",
                f"algo.num_envs={num_envs}",
                "env.goal_ee.local_fraction=0.0",
            ]
            + CLEAN_OVERRIDES,
        )
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override), cfg


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    x, y, z, w = q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _new_episode(
    env,
    i: int,
    d0: float,
    base_start: np.ndarray,
    ready_ee_z: float,
    capture_outer: float,
    keep_series: bool,
) -> dict:
    goal = env.world_ee_goal[i]
    gd = goal[:2] - base_start[:2]
    ep = {
        "g0": float(d0),
        "goal_world": goal.copy(),
        "base_start": base_start.copy(),
        "ready_ee_z": float(ready_ee_z),
        "goal_z": float(goal[2]),
        "vertical_floor": abs(float(goal[2]) - ready_ee_z),
        "impossible": abs(float(goal[2]) - ready_ee_z) >= capture_outer,
        "goal_dir_xy": (
            float(math.atan2(gd[1], gd[0])) if float(np.linalg.norm(gd)) > 1e-6 else 0.0
        ),
        "d": [],
        "base_hist": [],
        "series": [] if keep_series else None,
        "sum_abs_vfwd": 0.0,
        "sum_abs_vlat": 0.0,
        "sum_abs_act_wz": 0.0,
        "sum_lat_ratio_moving": 0.0,
        "n_moving": 0,
        "sum_heading_cos": 1.0,  # starts at neutral so blank rounds stay defined
        "n_head": 1,
        "once": False,
        "hold_timer": 0,
        "hold": False,
        "tts": -1,
        "steps": 0,
        "done_reason": "timeout",
    }
    return ep


def run_geometry_checkpoint(
    env,
    wrapped,
    policy,
    *,
    num_envs: int,
    rounds: int,
    capture_inner: float,
    capture_outer: float,
    series_cap: int,
) -> list[dict]:
    """Deterministic all-EXTENDED eval with geometry + drive diagnostics."""
    finished: list[dict] = []
    kept_series = 0

    for _round in range(rounds):
        obs = wrapped.reset()[0]
        # Public reset does NOT zero the truncation counter and init_state()
        # randomizes it — explicit zero keeps every round running full length.
        env.state.info["steps"][:] = 0
        ee0 = _ee_world(env)
        d0 = np.linalg.norm(ee0 - env.world_ee_goal, axis=1)
        base_start = env._backend.get_base_pos()[:, :2].copy()
        eps = []
        for i in range(num_envs):
            keep = kept_series < series_cap
            eps.append(
                _new_episode(env, i, d0[i], base_start[i], float(ee0[i, 2]), capture_outer, keep)
            )
            kept_series += int(keep)

        for _step in range(MAX_EP_STEPS):
            act = policy(obs)
            act_np = act.detach().cpu().numpy()
            obs, _rew, dones, infos = wrapped.step(act)
            dones_np = dones.detach().cpu().numpy().astype(bool)
            if "time_outs" in infos:
                to_np = infos["time_outs"].detach().cpu().numpy().astype(bool)
            else:
                to_np = np.zeros(num_envs, dtype=bool)

            d = np.linalg.norm(_ee_world(env) - env.world_ee_goal, axis=1)
            base_pos_now = env._backend.get_base_pos()
            v_world = env._backend.get_base_lin_vel()[:, :2]
            yaw = _yaw_from_quat(env._backend.get_base_quat())
            cy, sy = np.cos(yaw), np.sin(yaw)
            v_fwd = cy * v_world[:, 0] + sy * v_world[:, 1]  # body-frame forward (+x_b)
            v_lat = -sy * v_world[:, 0] + cy * v_world[:, 1]  # body-frame lateral (+y_b)
            speed = np.linalg.norm(v_world, axis=1)
            moving = speed > MOVE_SPEED_EPS
            vel_dir = np.arctan2(v_world[:, 1], v_world[:, 0])

            for i in range(num_envs):
                ep = eps[i]
                if ep is None:
                    continue
                ep["steps"] += 1
                di = float(d[i])
                ep["d"].append(di)
                ep["base_hist"].append(base_pos_now[i, :2].copy())

                head_err = float(_wrap_angle(float(vel_dir[i]) - ep["goal_dir_xy"]))
                ep["sum_abs_vfwd"] += abs(float(v_fwd[i]))
                ep["sum_abs_vlat"] += abs(float(v_lat[i]))
                ep["sum_abs_act_wz"] += abs(float(act_np[i, 2]))  # wz action channel
                if moving[i]:
                    ep["n_moving"] += 1
                    ep["sum_lat_ratio_moving"] += abs(float(v_lat[i])) / max(float(speed[i]), 1e-6)
                ep["sum_heading_cos"] += math.cos(head_err)
                ep["n_head"] += 1

                if ep["series"] is not None:
                    ep["series"].append(
                        {
                            "step": ep["steps"],
                            "d": di,
                            "v_fwd": round(float(v_fwd[i]), 4),
                            "v_lat": round(float(v_lat[i]), 4),
                            "speed": round(float(speed[i]), 4),
                            "heading_err_deg": round(math.degrees(head_err), 1)
                            if moving[i]
                            else None,
                            "act_vx": round(float(act_np[i, 0]), 4),
                            "act_vy": round(float(act_np[i, 1]), 4),
                            "act_wz": round(float(act_np[i, 2]), 4),
                        }
                    )

                within = di < SUCCESS_10
                if within and not ep["once"]:
                    ep["once"] = True
                    ep["tts"] = ep["steps"]
                ep["hold_timer"] = ep["hold_timer"] + 1 if within else 0
                ep["hold"] = ep["hold"] or (ep["hold_timer"] >= HOLD_STEPS)

                if dones_np[i]:
                    ep["done_reason"] = (
                        "success_hold" if ep["hold"] else ("timeout" if to_np[i] else "fail_term")
                    )
                    finished.append(ep)
                    eps[i] = None
            if all(e is None for e in eps):
                break

        for ep in eps:
            if ep is not None:
                finished.append(ep)

    out_rows = []
    for ep in finished:
        d = np.asarray(ep.pop("d"), dtype=np.float64)
        base = np.stack(ep.pop("base_hist"), axis=0)  # (T+1, 2)
        steps = max(ep["steps"], 1)
        in_cap = d <= capture_outer
        entered = bool(in_cap.any())
        cap_entry = int(np.argmax(in_cap)) + 1 if entered else -1
        exits = int(np.sum(in_cap[:-1] & ~in_cap[1:])) if len(d) > 1 else 0
        tts = ep["tts"]
        ep.update(
            {
                "entered": entered,
                "capture_entry": cap_entry,
                "exits_after_entry": exits,
                # eval_ranger_box_reach-compatible success keys (aggregate reads these)
                "once10": ep["once"],
                "hold10": ep["hold"],
                "cap_to_success": (tts - cap_entry) if (entered and tts >= cap_entry) else -1,
                "min_d": float(d.min()),
                "final_d": float(d[-1]),
                "base_disp": float(np.linalg.norm(base[-1] - base[0])),
                "base_path": float(np.linalg.norm(np.diff(base, axis=0), axis=1).sum()),
                "mean_abs_vfwd": ep["sum_abs_vfwd"] / steps,
                "mean_abs_vlat": ep["sum_abs_vlat"] / steps,
                "mean_abs_act_wz": ep["sum_abs_act_wz"] / steps,
                "lat_ratio_moving": (
                    ep["sum_lat_ratio_moving"] / ep["n_moving"] if ep["n_moving"] else float("nan")
                ),
                "heading_agreement": ep["sum_heading_cos"] / max(ep["n_head"], 1),
                "frac_time_moving": ep["n_moving"] / steps,
            }
        )
        for k in ("goal_world", "base_start"):
            ep[k] = np.round(ep[k], 4).tolist()
        ep["_bin"] = _bin_label(ep["vertical_floor"])
        out_rows.append(ep)
    return out_rows


def _bin_label(floor: float) -> str:
    for k in range(len(DZ_BIN_EDGES) - 1):
        lo, hi = DZ_BIN_EDGES[k], DZ_BIN_EDGES[k + 1]
        if lo <= floor < hi:
            return DZ_BIN_LABELS[k]
    return DZ_BIN_LABELS[-1]


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def m(key):
        vals = [r[key] for r in rows if isinstance(r[key], float) and not math.isnan(r[key])]
        vals = vals if vals else [r[key] for r in rows if not math.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def rate(key):
        return float(np.mean([1.0 if r[key] else 0.0 for r in rows]))

    def pct(key, q):
        return float(np.percentile([r[key] for r in rows], q))

    entered = [r for r in rows if r["entered"]]
    sac = (
        float(np.mean([1.0 if r["hold10"] else 0.0 for r in entered])) if entered else float("nan")
    )
    return {
        "n": n,
        "impossible_frac": rate("impossible"),
        "capture_entry_rate": rate("entered"),
        "success_once_10cm": rate("once10"),
        "success_hold_10cm": rate("hold10"),
        "success_after_capture": sac,
        "escape_rate": float(np.mean([1.0 if r["exits_after_entry"] > 0 else 0.0 for r in rows])),
        "min_d_mean": m("min_d"),
        "final_d_p50": pct("final_d", 50),
        "final_d_p90": pct("final_d", 90),
        "base_disp_mean": m("base_disp"),
        "base_path_mean": m("base_path"),
        "mean_abs_vfwd": m("mean_abs_vfwd"),
        "mean_abs_vlat": m("mean_abs_vlat"),
        "mean_abs_act_wz": m("mean_abs_act_wz"),
        "lat_ratio_moving": m("lat_ratio_moving"),
        "heading_agreement": m("heading_agreement"),
        "frac_time_moving": m("frac_time_moving"),
    }


def _print_bins(bins: dict[str, dict]) -> None:
    hdr = (
        f"{'|dz| bin':>10} {'n':>5} {'impFrac':>8} {'cap':>6} {'once10':>7} {'hold10':>7} "
        f"{'minD':>6} {'finP50':>7} {'finP90':>7} {'disp':>6} {'path':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label in DZ_BIN_LABELS:
        b = bins.get(label)
        if b is None or b["n"] == 0:
            print(f"{label:>10} {'0':>5}")
            continue
        print(
            f"{label:>10} {b['n']:>5} {b['impossible_frac']:>8.2f} {b['capture_entry_rate']:>6.3f} "
            f"{b['success_once_10cm']:>7.3f} {b['success_hold_10cm']:>7.3f} "
            f"{b['min_d_mean']:>6.3f} {b['final_d_p50']:>7.3f} {b['final_d_p90']:>7.3f} "
            f"{b['base_disp_mean']:>6.3f} {b['base_path_mean']:>6.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-run", required=True, help="run directory holding model_<N>.pt")
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=63, help="episodes = num_envs * rounds")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[150, 199])
    ap.add_argument("--series-cap", type=int, default=SERIES_CAP_DEFAULT)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, cfg = build_ext_env(args.num_envs)
    env.set_autoreset(False)
    capture_inner = float(getattr(env._cfg.goal_ee, "capture_inner", 0.15))
    capture_outer = float(getattr(env._cfg.goal_ee, "capture_outer", 0.20))
    assert float(env._cfg.goal_ee.local_fraction) == 0.0, "benchmark must run all-EXTENDED"
    print(
        f"geometry setup: {args.num_envs} envs x {args.rounds} rounds = "
        f"{args.num_envs * args.rounds} EXTENDED episodes, "
        f"capture {capture_inner}/{capture_outer} m"
    )

    wrapped = RslRlVecEnvWrapper(env, device=device)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    train_cfg.setdefault("runner", {})["logger"] = "none"
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)

    for it in args.checkpoints:
        ckpt = Path(args.load_run) / f"model_{it}.pt"
        if not ckpt.exists():
            print(f"!! missing {ckpt}, skipped")
            continue
        with policy_load_dim_guard(
            env_obs_dim=getattr(wrapped, "num_obs", None),
            env_action_dim=getattr(wrapped, "num_actions", None),
            algo_name="ppo",
        ):
            runner.load(str(ckpt), map_location=device)
        policy = runner.get_inference_policy(device=device)

        rows = run_geometry_checkpoint(
            env,
            wrapped,
            policy,
            num_envs=args.num_envs,
            rounds=args.rounds,
            capture_inner=capture_inner,
            capture_outer=capture_outer,
            series_cap=args.series_cap,
        )

        bins = {
            label: _agg([r for r in rows if r["_bin"] == label])
            for label in DZ_BIN_LABELS
            if any(r["_bin"] == label for r in rows)
        }
        possible = _agg([r for r in rows if not r["impossible"]])
        impossible = _agg([r for r in rows if r["impossible"]])
        by_outcome = {
            "success_hold10": _agg([r for r in rows if r["hold10"]]),
            "failure": _agg([r for r in rows if not r["hold10"]]),
        }

        out = {
            "checkpoint": it,
            "num_episodes": len(rows),
            "all_ext_forced": True,
            "capture_inner": capture_inner,
            "capture_outer": capture_outer,
            "impossible_count": sum(1 for r in rows if r["impossible"]),
            "impossible_fraction": (sum(1 for r in rows if r["impossible"]) / max(len(rows), 1)),
            "bins": bins,
            "possible_subset": possible,
            "impossible_subset": impossible,
            "by_outcome": by_outcome,
            "series_samples": [
                {
                    "episode": idx,
                    "vertical_floor": r["vertical_floor"],
                    "impossible": r["impossible"],
                    "hold10": r["hold10"],
                    "goal_dir_xy": r["goal_dir_xy"],
                    "series": r["series"],
                }
                for idx, r in enumerate((r for r in rows if r.get("series")))
            ],
        }
        out_path = Path(args.load_run) / f"extended_geometry_iter{it}.json"
        out_path.write_text(json.dumps(out, indent=2))

        print("=" * 96)
        print(
            f"Checkpoint model_{it}.pt — EXTENDED-only, {len(rows)} episodes, "
            f"impossible |dz|>=capture_outer({capture_outer}): "
            f"{out['impossible_count']} ({out['impossible_fraction']:.1%})"
        )
        print("=" * 96)
        _print_bins(bins)
        print(
            f"\nPOSSIBLE subset   n={possible['n']:>5}  cap {possible['capture_entry_rate']:.3f}  "
            f"once10 {possible['success_once_10cm']:.3f}  hold10 {possible['success_hold_10cm']:.3f}"
        )
        if impossible["n"]:
            print(
                f"IMPOSSIBLE subset n={impossible['n']:>5}  cap {impossible['capture_entry_rate']:.3f}  "
                f"once10 {impossible['success_once_10cm']:.3f}  "
                f"hold10 {impossible['success_hold_10cm']:.3f}"
            )
        for name, grp in by_outcome.items():
            if grp.get("n"):
                print(
                    f"{name:>15}: n={grp['n']:>4}  |vfwd| {grp['mean_abs_vfwd']:.3f}  "
                    f"|vlat| {grp['mean_abs_vlat']:.3f}  |act_wz| {grp['mean_abs_act_wz']:.3f}  "
                    f"latRatio {grp['lat_ratio_moving']:.3f}  "
                    f"headAgr {grp['heading_agreement']:.3f}  moveFrac {grp['frac_time_moving']:.3f}"
                )
        print(f"\nWrote {out_path}")

    env.close()


if __name__ == "__main__":
    main()
