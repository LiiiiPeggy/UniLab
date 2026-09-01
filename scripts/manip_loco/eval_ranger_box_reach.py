"""Deterministic per-checkpoint eval for RangerBoxReach — hard metrics, not reward.

Usage:
  uv run scripts/manip_loco/eval_ranger_box_reach.py \
      --load-run <run_dir> --num-envs 16 --rounds 13 \
      --checkpoints 50 100 200 299

For each checkpoint: loads the actor, runs it deterministically (mean action,
no exploration noise) with autoreset disabled so every terminated episode keeps
its FULL final-step data (autoreset would hide the dying step behind the reset).
One episode per env per round, 500 steps = 10 s, noise/latency/DR off.

Reports LOCAL / EXTENDED / ALL splits:
  success_once_10cm / success_hold_10cm / success_5cm, final EE p50/p90,
  time_to_success, capture_entry_rate, time_to_capture, capture→success,
  base displacement / path length / final base-goal distance, collision &
  joint-limit rates, arm_weight transition bands, arm deviation while the base
  is far (premature extension), arm residual / base action magnitudes.

Writes <run_dir>/eval_metrics_iter{N}.json per checkpoint and prints a
cross-checkpoint summary table (mean reward joined from TensorBoard events
when available).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Sequence

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

from unilab.envs.common.rotation import (  # noqa: E402
    np_quat_apply_batched,
    np_quat_conjugate_batched,
)
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402
from unilab.training.sim2sim import policy_load_dim_guard  # noqa: E402

CONF_DIR = str(ROOT_DIR / "conf" / "ppo")
TORQUE_SENSORS = [f"cr10_j{j}_torque" for j in range(1, 7)]
TORQUE_LIMITS = np.array([15, 50, 50, 25, 25, 25], dtype=np.float64)

SUCCESS_10 = 0.10
SUCCESS_05 = 0.05
HOLD_STEPS = 25  # 0.5 s at ctrl_dt 0.02
MAX_EP_STEPS = 500  # 10 s at ctrl_dt 0.02

_BASE_BOX_HALF = np.array([0.55, 0.38, 0.20], dtype=np.float64)
_BOX_CENTRE_OFFSET = np.array([-0.1262, 0.0, -0.0965], dtype=np.float64)

CLEAN_OVERRIDES = [
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


def _build_env(
    num_envs: int,
    arm_action_scale: float | None = None,
    config_overrides: Sequence[str] | None = None,
):
    ensure_registries()
    overrides = list(CLEAN_OVERRIDES)
    if arm_action_scale is not None:
        # Match the training-time controller scale (deterministic eval must not
        # reintroduce an arm residual the policy was trained without).
        overrides.append(f"env.control_config.arm_action_scale={arm_action_scale}")
    if config_overrides:
        overrides.extend(config_overrides)
    with initialize_config_dir(version_base="1.3", config_dir=CONF_DIR):
        cfg = compose(
            config_name="config",
            overrides=["task=ranger_box_reach/mujoco", f"algo.num_envs={num_envs}"] + overrides,
        )
    env_cfg_override = BackendAdapter(cfg, root_dir=ROOT_DIR).build_task_env_cfg_override()
    return create_env(cfg, num_envs=num_envs, env_cfg_override=env_cfg_override), cfg


def _ee_world(env) -> np.ndarray:
    ee_local, _ = env.get_ee_local_pose()
    return env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)


def _min_arm_signed_dist(env, ee_world: np.ndarray) -> np.ndarray:
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


def _new_episode(env, i: int, d0: float, base_xy: np.ndarray) -> dict:
    return {
        "is_local": bool(env._goal_is_local[i]),
        "g0": float(d0),
        "goal_world": env.world_ee_goal[i].copy(),
        "base_start": base_xy[i].copy(),
        "base_hist": [base_xy[i].copy()],
        "d": [],
        "aw": [],
        "far_dev": [],
        "once": False,
        "once5": False,
        "hold_timer": 0,
        "hold": False,
        "tts": -1,
        "jl": False,
        "col": False,
        "sat": 0.0,
        "arm_res": 0.0,
        "base_act": 0.0,
        "act_rate": 0.0,
        "prev_act": None,
        "steps": 0,
        "mode_switch": 0,
        "prev_mode": None,
        "done_reason": "timeout",
    }


def run_checkpoint(
    env,
    wrapped,
    runner,
    policy,
    *,
    num_envs: int,
    rounds: int,
    capture_inner: float,
    capture_outer: float,
    device: str,
) -> list[dict]:
    aw_span = max(capture_outer - capture_inner, 1e-6)
    finished: list[dict] = []

    for _ in range(rounds):
        obs = wrapped.reset()[0]
        # The public reset does NOT zero the truncation counter (only the
        # autoreset path does, np_env._reset_done_envs), and init_state() starts
        # it at RANDOM values — without this, later rounds truncate at step 1.
        env.state.info["steps"][:] = 0
        d0 = np.linalg.norm(_ee_world(env) - env.world_ee_goal, axis=1)
        base_xy = env._backend.get_base_pos()[:, :2].copy()
        eps = [_new_episode(env, i, d0[i], base_xy) for i in range(num_envs)]

        for _step in range(MAX_EP_STEPS):
            act = policy(obs)
            act_np = act.detach().cpu().numpy()
            obs, rew, dones, infos = wrapped.step(act)
            dones_np = dones.detach().cpu().numpy().astype(bool)
            if "time_outs" in infos:
                to_np = infos["time_outs"].detach().cpu().numpy().astype(bool)
            else:
                to_np = np.zeros(num_envs, dtype=bool)

            # Batched post-step measurements (autoreset OFF → these are the
            # dying episode's true final values for done envs).
            ee_world = _ee_world(env)
            d = np.linalg.norm(ee_world - env.world_ee_goal, axis=1)
            aw = np.clip((capture_outer - d) / aw_span, 0.0, 1.0)
            base_xy_now = env._backend.get_base_pos()[:, :2]
            arm_q = env.get_arm_dof_pos()
            jl_now = ((arm_q > env._arm_joint_upper) | (arm_q < env._arm_joint_lower)).any(axis=1)
            col_now = _min_arm_signed_dist(env, ee_world) < 0.0
            forces = np.stack(
                [env._backend.get_sensor_data(n)[:, 0] for n in TORQUE_SENSORS], axis=1
            )
            sat_now = (np.abs(forces) / TORQUE_LIMITS > 0.95).sum(axis=1)
            arm_dev = np.abs(arm_q - env._default_arm_angles).mean(axis=1)
            far = d > capture_outer

            for i in range(num_envs):
                ep = eps[i]
                if ep is None:
                    continue
                ep["steps"] += 1
                di = float(d[i])
                ep["d"].append(di)
                ep["aw"].append(float(aw[i]))
                ep["base_hist"].append(base_xy_now[i].copy())
                if far[i]:
                    ep["far_dev"].append(float(arm_dev[i]))
                within = di < SUCCESS_10
                if within and not ep["once"]:
                    ep["once"] = True
                    ep["tts"] = ep["steps"]
                ep["once5"] = ep["once5"] or (di < SUCCESS_05)
                ep["hold_timer"] = ep["hold_timer"] + 1 if within else 0
                ep["hold"] = ep["hold"] or (ep["hold_timer"] >= HOLD_STEPS)
                ep["jl"] = ep["jl"] or bool(jl_now[i])
                ep["col"] = ep["col"] or bool(col_now[i])
                ep["sat"] += float(sat_now[i])
                ep["arm_res"] += float(np.abs(act_np[i, 3:9]).mean())
                ep["base_act"] += float(np.abs(act_np[i, 0:3]).mean())
                if ep["prev_act"] is not None:
                    ep["act_rate"] += float(np.abs(act_np[i] - ep["prev_act"]).mean())
                ep["prev_act"] = act_np[i].copy()

                if dones_np[i]:
                    ep["done_reason"] = (
                        "success_hold" if ep["hold"] else ("timeout" if to_np[i] else "fail_term")
                    )
                    finished.append(ep)
                    eps[i] = None
            # Base motion-mode switch count (adapter publishes per-env mode).
            # The rsl_rl wrapper only forwards time_outs in infos, so read the
            # env state.info directly.
            bm = getattr(env.state, "info", {}).get("base_command_mode", None)
            if bm is not None:
                bm_np = np.asarray(bm)
                for i in range(num_envs):
                    ep = eps[i]
                    if ep is None:
                        continue
                    m_i = int(bm_np[i])
                    if ep["prev_mode"] is not None and m_i != ep["prev_mode"]:
                        ep["mode_switch"] += 1
                    ep["prev_mode"] = m_i
            if all(e is None for e in eps):
                break

        for ep in eps:  # round ended: remaining envs ran the full 500 steps
            if ep is not None:
                ep["done_reason"] = "timeout"
                finished.append(ep)

    return [_finalize_ep(ep, capture_inner, capture_outer) for ep in finished]


def _finalize_ep(ep: dict, capture_inner: float, capture_outer: float) -> dict:
    d = np.asarray(ep["d"], dtype=np.float64)
    aw = np.asarray(ep["aw"], dtype=np.float64)
    base = np.stack(ep["base_hist"], axis=0)  # (T+1, 2)
    base_disp = float(np.linalg.norm(base[-1] - base[0]))
    base_path = float(np.linalg.norm(np.diff(base, axis=0), axis=1).sum())
    in_cap = d <= capture_outer
    entered = bool(in_cap.any())
    cap_entry = int(np.argmax(in_cap)) + 1 if entered else -1
    exits = int(np.sum(in_cap[:-1] & ~in_cap[1:])) if len(d) > 1 else 0
    tts = int(ep["tts"])
    far = d > capture_outer
    ramp = (d > capture_inner) & (d <= capture_outer)
    near = d <= capture_inner
    steps = max(ep["steps"], 1)
    return {
        "is_local": ep["is_local"],
        "g0": ep["g0"],
        "n_steps": ep["steps"],
        "once10": ep["once"],
        "once5": ep["once5"],
        "hold10": ep["hold"],
        "tts": tts,
        "final_d": float(d[-1]),
        "min_d": float(d.min()),
        "base_disp": base_disp,
        "base_path": base_path,
        "final_base_goal_horiz": float(np.linalg.norm(base[-1] - ep["goal_world"][:2])),
        "entered": entered,
        "capture_entry": cap_entry,
        "exits_after_entry": exits,
        "cap_to_success": (tts - cap_entry) if (entered and tts >= cap_entry) else -1,
        "aw_far": float(aw[far].mean()) if far.any() else float("nan"),
        "aw_ramp": float(aw[ramp].mean()) if ramp.any() else float("nan"),
        "aw_near": float(aw[near].mean()) if near.any() else float("nan"),
        "far_dev_mean": float(np.mean(ep["far_dev"])) if ep["far_dev"] else 0.0,
        "jl": ep["jl"],
        "col": ep["col"],
        "sat": ep["sat"],
        "arm_res_mean": ep["arm_res"] / steps,
        "base_act_mean": ep["base_act"] / steps,
        "act_rate_mean": ep["act_rate"] / max(ep["steps"] - 1, 1),
        "mode_switching_freq": ep["mode_switch"] / max(ep["steps"] - 1, 1),
        "done_reason": ep["done_reason"],
    }


def _agg(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def m(key):
        return float(np.mean([r[key] for r in rows]))

    def rate(key):
        return float(np.mean([1.0 if r[key] else 0.0 for r in rows]))

    def nm(key):
        # arm_weight bands are NaN for episodes that never visited the band
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def pct(key, q):
        return float(np.percentile([r[key] for r in rows], q))

    entered = [r for r in rows if r["entered"]]
    tts_ok = [r["tts"] for r in rows if r["tts"] > 0]
    ttc_ok = [r["capture_entry"] for r in entered]
    c2s = [r["cap_to_success"] for r in entered if r["cap_to_success"] >= 0]
    reasons: dict[str, int] = {}
    for r in rows:
        reasons[r["done_reason"]] = reasons.get(r["done_reason"], 0) + 1
    # success after capture: of the episodes that entered the capture region,
    # how many eventually held success
    sac = (
        float(np.mean([1.0 if r["hold10"] else 0.0 for r in entered])) if entered else float("nan")
    )
    return {
        "n": n,
        "success_once_10cm": rate("once10"),
        "success_hold_10cm": rate("hold10"),
        "success_5cm": rate("once5"),
        "final_d_p50": pct("final_d", 50),
        "final_d_p90": pct("final_d", 90),
        "min_d_p50": pct("min_d", 50),
        "g0_mean": m("g0"),
        "time_to_success_mean": float(np.mean(tts_ok)) if tts_ok else float("nan"),
        "capture_entry_rate": rate("entered"),
        "time_to_capture_mean": float(np.mean(ttc_ok)) if ttc_ok else float("nan"),
        "capture_to_success_mean": float(np.mean(c2s)) if c2s else float("nan"),
        "success_after_capture": sac,
        "capture_escape_rate": float(
            np.mean([1.0 if r["exits_after_entry"] > 0 else 0.0 for r in rows])
        ),
        "base_disp_mean": m("base_disp"),
        "base_path_mean": m("base_path"),
        "final_base_goal_horiz_mean": m("final_base_goal_horiz"),
        "arm_weight_far_mean": nm("aw_far"),
        "arm_weight_ramp_mean": nm("aw_ramp"),
        "arm_weight_near_mean": nm("aw_near"),
        "far_dev_mean_rad": m("far_dev_mean"),
        "arm_residual_mean": m("arm_res_mean"),
        "base_action_mean": m("base_act_mean"),
        "action_rate_mean": m("act_rate_mean"),
        "collision_rate": rate("col"),
        "joint_limit_rate": rate("jl"),
        "actuator_sat_per_step": m("sat") / max(m("n_steps"), 1.0),
        "episode_len_mean": m("n_steps"),
        "mode_switching_freq": m("mode_switching_freq"),
        "done_reasons": reasons,
    }


def _print_group(name: str, f: dict) -> None:
    if f.get("n", 0) == 0:
        print(f"\n[{name}]  n=0")
        return
    print(f"\n[{name}]  n={f['n']}")
    print(
        f"  once10 {f['success_once_10cm']:.3f}  hold10 {f['success_hold_10cm']:.3f}  "
        f"once5 {f['success_5cm']:.3f}  tts {f['time_to_success_mean']:.0f} steps"
    )
    print(
        f"  final p50 {f['final_d_p50']:.3f}  p90 {f['final_d_p90']:.3f}  "
        f"min p50 {f['min_d_p50']:.3f}  g0 {f['g0_mean']:.3f}"
    )
    print(
        f"  capture_entry {f['capture_entry_rate']:.3f}  ttc {f['time_to_capture_mean']:.1f}  "
        f"cap→succ {f['capture_to_success_mean']:.1f}  succ_after_cap {f['success_after_capture']:.3f}  "
        f"escape {f['capture_escape_rate']:.3f}"
    )
    print(
        f"  base_disp {f['base_disp_mean']:.3f}  base_path {f['base_path_mean']:.3f}  "
        f"final_base_goal {f['final_base_goal_horiz_mean']:.3f}"
    )
    print(
        f"  aw far/ramp/near {f['arm_weight_far_mean']:.3f}/{f['arm_weight_ramp_mean']:.3f}/"
        f"{f['arm_weight_near_mean']:.3f}  far_dev {f['far_dev_mean_rad']:.3f} rad"
    )
    print(
        f"  arm_res {f['arm_residual_mean']:.4f}  base_act {f['base_action_mean']:.4f}  "
        f"act_rate {f['action_rate_mean']:.4f}"
    )
    print(
        f"  col {f['collision_rate']:.3f}  jl {f['joint_limit_rate']:.3f}  "
        f"sat/step {f['actuator_sat_per_step']:.3f}  ep_len {f['episode_len_mean']:.0f}"
    )
    print(f"  done: {f['done_reasons']}")


def _read_tb_reward(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Read Train/mean_reward per iteration from the run's TensorBoard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return np.array([]), np.array([])
    files = sorted(glob.glob(str(Path(run_dir) / "events.out.tfevents.*")))
    if not files:
        return np.array([]), np.array([])
    acc = EventAccumulator(str(Path(run_dir)), size_guidance={"scalars": 0})
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    for tag in ("Train/mean_reward", "train/mean_reward"):
        if tag in tags:
            events = acc.Scalars(tag)
            return (
                np.array([e.step for e in events], dtype=np.float64),
                np.array([e.value for e in events], dtype=np.float64),
            )
    return np.array([]), np.array([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-run", required=True)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=13, help="episodes = num_envs * rounds")
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[50, 100, 200, 299])
    ap.add_argument(
        "--arm-action-scale",
        type=float,
        default=None,
        help="Override env.control_config.arm_action_scale to match the "
        "training contract (e.g. 0.0 when the run trained without residual).",
    )
    ap.add_argument(
        "--config-override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Extra Hydra override appended to the env compose "
        "(repeatable), e.g. --config-override env.command_adapter.enable=false.",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, cfg = _build_env(
        args.num_envs,
        arm_action_scale=args.arm_action_scale,
        config_overrides=args.config_override,
    )
    env.set_autoreset(False)
    capture_inner = float(getattr(env._cfg.goal_ee, "capture_inner", 0.15))
    capture_outer = float(getattr(env._cfg.goal_ee, "capture_outer", 0.20))
    print(
        f"eval setup: {args.num_envs} envs x {args.rounds} rounds "
        f"= {args.num_envs * args.rounds} episodes, capture {capture_inner}/{capture_outer} m"
    )

    wrapped = RslRlVecEnvWrapper(env, device=device)
    rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
    train_cfg = normalize_ppo_train_cfg(rl_cfg)
    train_cfg.setdefault("runner", {})["logger"] = "none"
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)

    tb_it, tb_rew = _read_tb_reward(args.load_run)
    summary_rows = []

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

        rows = run_checkpoint(
            env,
            wrapped,
            runner,
            policy,
            num_envs=args.num_envs,
            rounds=args.rounds,
            capture_inner=capture_inner,
            capture_outer=capture_outer,
            device=device,
        )
        local = _agg([r for r in rows if r["is_local"]])
        extended = _agg([r for r in rows if not r["is_local"]])
        allr = _agg(rows)
        out = {
            "checkpoint": it,
            "num_episodes": len(rows),
            "local": local,
            "extended": extended,
            "all": allr,
        }

        rew = float("nan")
        if len(tb_it):
            idx = int(np.argmin(np.abs(tb_it - it)))
            rew = float(tb_rew[idx])
            out["train_reward_at_iter"] = rew

        out_path = Path(args.load_run) / f"eval_metrics_iter{it}.json"
        out_path.write_text(json.dumps(out, indent=2))

        print("=" * 78)
        print(f"Checkpoint model_{it}.pt   (train reward @iter: {rew:.2f})")
        print("=" * 78)
        _print_group("LOCAL", local)
        _print_group("EXTENDED", extended)
        _print_group("ALL", allr)
        print(f"\nWrote {out_path}")

        summary_rows.append(
            {
                "iter": it,
                "rew": rew,
                "loc_hold": local.get("success_hold_10cm", float("nan")),
                "loc_once": local.get("success_once_10cm", float("nan")),
                "ext_cap": extended.get("capture_entry_rate", float("nan")),
                "ext_hold": extended.get("success_hold_10cm", float("nan")),
                "ext_p50": extended.get("final_d_p50", float("nan")),
                "ext_disp": extended.get("base_disp_mean", float("nan")),
                "col": allr.get("collision_rate", float("nan")),
                "jl": allr.get("joint_limit_rate", float("nan")),
            }
        )

    if summary_rows:
        print("\n" + "=" * 100)
        print("Run-3 stage summary  (LOCAL/EXTENDED split, deterministic mean-action eval)")
        print("=" * 100)
        hdr = (
            f"{'iter':>5} {'rew':>7} {'LOConce':>8} {'LOChold':>8} {'EXTcap':>7} "
            f"{'EXThold':>8} {'EXTp50':>7} {'EXTdisp':>8} {'col':>5} {'jl':>5}"
        )
        print(hdr)
        for r in summary_rows:
            print(
                f"{r['iter']:>5} {r['rew']:>7.2f} {r['loc_once']:>8.3f} {r['loc_hold']:>8.3f} "
                f"{r['ext_cap']:>7.3f} {r['ext_hold']:>8.3f} {r['ext_p50']:>7.3f} "
                f"{r['ext_disp']:>8.3f} {r['col']:>5.3f} {r['jl']:>5.3f}"
            )

    env.close()


if __name__ == "__main__":
    main()
