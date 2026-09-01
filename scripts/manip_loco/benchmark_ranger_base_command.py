"""Benchmark the Ranger base command chain and mode-switching behaviour.

Drives the RangerBoxReach env with scripted / random base commands and reports:

- mode_switch_count / mode_switching_freq for four input cases, run under the
  mode-hysteresis BEFORE (single 0.05 vy threshold, no dwell) and AFTER
  (enter 0.08 / exit 0.03 + min_mode_duration 0.2 s) configs.
- For the random case, the full command chain statistics (zero-command
  residual, velocity jitter, sign flips) under the raw (Run-4B, adapter off)
  and adapter-on configs.

Cases:
  A  vy_threshold : vy ~ U(0.02, 0.1) around the boundary, vx=0.5
  B  vy_osc       : vy = 0.06 sin(2π t/T), vx=0.5  (crosses the boundary)
  C  stop_go      : vx alternates 0.5 / 0.0 every 25 steps
  D  random       : full random policy-output commands

Writes a JSON with per-case / per-config mode-switch stats.

Usage:
  uv run scripts/manip_loco/benchmark_ranger_base_command.py \
      --steps 200 --num-envs 64 --out logs/.../base_command_benchmark.json
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

sys.path.insert(0, str(ROOT_DIR / "scripts" / "manip_loco"))
from eval_ranger_box_reach import _build_env  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from unilab.envs.locomotion.ranger_box.ranger_command_adapter import MODE_NAMES  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402

# BEFORE = legacy single-threshold behaviour (previous batch).
CONFIG_BEFORE = {
    "parallel_enter_vy": 0.05,
    "parallel_exit_vy": 0.05,
    "min_mode_duration": 0.0,
}
# AFTER = mode hysteresis + min dwell (this batch).
CONFIG_AFTER = {
    "parallel_enter_vy": 0.08,
    "parallel_exit_vy": 0.03,
    "min_mode_duration": 0.2,
}

CASES = ("vy_threshold", "vy_osc", "stop_go", "random")


def _case_action(case: str, step: int, num_envs: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((num_envs, 9), dtype=np.float32)
    if case == "vy_threshold":
        a[:, 0] = 0.5
        a[:, 1] = rng.uniform(0.02, 0.10, size=num_envs)
    elif case == "vy_osc":
        a[:, 0] = 0.5
        a[:, 1] = 0.06 * np.sin(2.0 * np.pi * step / 40.0)
    elif case == "stop_go":
        a[:, 0] = 0.5 if (step // 25) % 2 == 0 else 0.0
    else:  # random
        a[:, :] = rng.uniform(-1.0, 1.0, size=(num_envs, 9))
    return a


def _run_chain(
    num_envs: int,
    steps: int,
    seed: int,
    case: str,
    overrides: list[str] | None = None,
    adapter_on: bool = True,
) -> dict:
    env, cfg = _build_env(num_envs, arm_action_scale=0.0, config_overrides=overrides)
    wrapped = RslRlVecEnvWrapper(env, device="cpu")
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
    train_cfg.setdefault("runner", {})["logger"] = "none"

    rng = np.random.default_rng(seed)
    wrapped.reset()

    lin_scale = float(cfg.env.base_velocity_controller.action_scale_lin)
    ang_scale = float(cfg.env.base_velocity_controller.action_scale_ang)

    modes_all: list[np.ndarray] = []
    a_in_all: list[np.ndarray] = []
    a_out_all: list[np.ndarray] = []
    root_all: list[np.ndarray] = []

    for step in range(steps):
        action = _case_action(case, step, num_envs, rng)
        raw = action[:, 0:3].copy()
        a_in = raw.astype(np.float64).copy()
        a_in[:, 0:2] *= lin_scale
        a_in[:, 2] *= ang_scale

        wrapped.step(torch.from_numpy(action))
        if adapter_on:
            a_out = env._base_command_adapter.last_output.copy()
            mode = env._base_command_adapter.last_mode.copy()
        else:
            a_out = a_in.copy()
            mode = np.zeros(num_envs, dtype=np.int64)
        linvel = env._backend.get_sensor_data(env._cfg.sensor.local_linvel)[:, :2]
        gyro = env._backend.get_sensor_data(env._cfg.sensor.gyro)[:, 2:3]

        modes_all.append(mode)
        a_in_all.append(a_in)
        a_out_all.append(a_out)
        root_all.append(np.concatenate([linvel, gyro], axis=1))

    env.close()

    mode = np.stack(modes_all)  # (steps, N)
    a_in = np.stack(a_in_all)
    a_out = np.stack(a_out_all)
    root = np.stack(root_all)

    # Mode switching: adjacent-step changes, count + per-env frequency.
    switch = (mode[1:] != mode[:-1]).sum()
    freq = float((mode[1:] != mode[:-1]).mean())
    mode_counts = {MODE_NAMES[k]: int((mode == k).sum()) for k in MODE_NAMES}

    result: dict = {
        "case": case,
        "mode_switch_count": int(switch),
        "mode_switching_freq": freq,
        "mode_counts": mode_counts,
    }
    # Full chain stats only for the random case (as before).
    if case == "random":
        nonzero_in = np.abs(a_in) > 0.01
        zeroed = nonzero_in & (np.abs(a_out) < 1e-9)
        result["zero_command_residual"] = float(zeroed.sum() / max(nonzero_in.sum(), 1))
        result["adapter_output_std"] = float(np.mean(np.std(a_out, axis=0)))
        result["root_velocity_std"] = float(np.mean(np.std(root, axis=0)))

        def _flips(x: np.ndarray) -> float:
            active = np.abs(x) > 1e-6
            pairs = active[1:] & active[:-1]
            fl = np.sign(x[1:]) != np.sign(x[:-1])
            return float((fl & pairs).sum() / max(int(pairs.sum()), 1))

        result["sign_flip_freq"] = {f"c{i}": _flips(a_out[:, :, i]) for i in range(3)}
    return result


def _adapter_overrides(cfg: dict) -> list[str]:
    return [
        f"env.command_adapter.parallel_enter_vy={cfg['parallel_enter_vy']}",
        f"env.command_adapter.parallel_exit_vy={cfg['parallel_exit_vy']}",
        f"env.command_adapter.min_mode_duration={cfg['min_mode_duration']}",
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="base_command_benchmark.json")
    args = ap.parse_args()

    results: dict = {
        "configs": {"before": CONFIG_BEFORE, "after": CONFIG_AFTER},
        "cases": {},
    }
    for case in CASES:
        results["cases"][case] = {
            "before": _run_chain(
                args.num_envs,
                args.steps,
                args.seed,
                case,
                overrides=_adapter_overrides(CONFIG_BEFORE),
            ),
            "after": _run_chain(
                args.num_envs,
                args.steps,
                args.seed,
                case,
                overrides=_adapter_overrides(CONFIG_AFTER),
            ),
        }

    # Raw (Run-4B, adapter off) random-path stats for reference.
    results["raw_trained_contract"] = _run_chain(
        args.num_envs,
        args.steps,
        args.seed,
        "random",
        overrides=["env.command_adapter.enable=false"],
        adapter_on=False,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
