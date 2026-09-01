"""Benchmark the Ranger base command chain: raw action -> adapter -> controller -> root velocity.

Drives the RangerBoxReach env with random base actions and records the full
command chain per step:

    raw_action[0:3] -> (scale) -> adapter_input -> adapter_output + mode
                   -> BaseVelocityController.v_real -> root sensor velocity

Statistics (per config, mean over envs/steps):

- zero_command_residual : fraction of steps a non-trivial scaled command
    (|v| > 0.01) was zeroed by the adapter (deadband / mode gating).
- velocity_jitter        : std of the adapter output and of the executed
    root velocity across steps (lower = smoother).
- sign_flip_freq         : fraction of adjacent-step pairs where a channel
    flipped sign while it was active.
- mode_switching_freq    : fraction of adjacent-step pairs where the motion
    mode changed.

Runs with the adapter ON (new YAML default) and OFF (Run-4B raw path) and
writes both to a JSON.

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


def _run_chain(num_envs: int, steps: int, seed: int, adapter_on: bool) -> dict:
    overrides = [f"env.command_adapter.enable={str(adapter_on).lower()}"]
    env, cfg = _build_env(num_envs, arm_action_scale=0.0, config_overrides=overrides)
    wrapped = RslRlVecEnvWrapper(env, device="cpu")
    train_cfg = normalize_ppo_train_cfg(OmegaConf.to_container(cfg.algo, resolve=True))
    train_cfg.setdefault("runner", {})["logger"] = "none"

    rng = np.random.default_rng(seed)
    wrapped.reset()

    lin_scale = float(cfg.env.base_velocity_controller.action_scale_lin)
    ang_scale = float(cfg.env.base_velocity_controller.action_scale_ang)

    raw_all: list[np.ndarray] = []
    a_in_all: list[np.ndarray] = []
    a_out_all: list[np.ndarray] = []
    mode_all: list[np.ndarray] = []
    vreal_all: list[np.ndarray] = []
    root_all: list[np.ndarray] = []

    for _ in range(steps):
        action = rng.uniform(-1.0, 1.0, size=(num_envs, 9)).astype(np.float32)
        raw = action[:, 0:3].copy()
        # Scale to velocity units (adapter input).
        a_in = raw.astype(np.float64).copy()
        a_in[:, 0:2] *= lin_scale
        a_in[:, 2] *= ang_scale

        wrapped.step(torch.from_numpy(action))
        # Adapter output / mode come from the state the env's step just wrote
        # (process() already ran inside apply_action; reading last_output avoids
        # re-running it and double-advancing the hysteresis).
        if adapter_on:
            a_out = env._base_command_adapter.last_output.copy()
            mode = env._base_command_adapter.last_mode.copy()
        else:
            mode = np.zeros(num_envs, dtype=np.int64)
            a_out = a_in.copy()
        vreal = env._base_controller.v_real.copy()
        linvel = env._backend.get_sensor_data(env._cfg.sensor.local_linvel)[:, :2]
        gyro = env._backend.get_sensor_data(env._cfg.sensor.gyro)[:, 2:3]
        root = np.concatenate([linvel, gyro], axis=1)

        raw_all.append(raw)
        a_in_all.append(a_in)
        a_out_all.append(a_out)
        mode_all.append(mode)
        vreal_all.append(vreal)
        root_all.append(root)

    env.close()

    raw = np.stack(raw_all)  # (steps, N, 3)
    a_in = np.stack(a_in_all)
    a_out = np.stack(a_out_all)
    mode = np.stack(mode_all)
    vreal = np.stack(vreal_all)
    root = np.stack(root_all)

    # 1. zero command residual: |scaled input| > 0.01 but adapter output == 0.
    nonzero_in = np.abs(a_in) > 0.01
    zeroed = nonzero_in & (np.abs(a_out) < 1e-9)
    zero_frac = zeroed.sum() / max(nonzero_in.sum(), 1)

    # 2. velocity jitter: std across time for each channel (mean over envs).
    a_out_std = float(np.mean(np.std(a_out, axis=0)))
    root_std = float(np.mean(np.std(root, axis=0)))

    # 3. sign flip frequency (adjacent steps, only where the channel was active).
    def _flips(x: np.ndarray) -> float:
        active = np.abs(x) > 1e-6
        pairs = active[1:] & active[:-1]
        fl = np.sign(x[1:]) != np.sign(x[:-1])
        denom = max(int(pairs.sum()), 1)
        return float((fl & pairs).sum() / denom)

    sign_flip = {f"c{i}": _flips(a_out[:, :, i]) for i in range(3)}

    # 4. mode switching frequency.
    mode_switch = float((mode[1:] != mode[:-1]).mean())

    mode_counts = {MODE_NAMES[k]: int((mode == k).sum()) for k in MODE_NAMES}

    return {
        "adapter_enabled": bool(adapter_on),
        "steps": int(steps),
        "num_envs": int(num_envs),
        "zero_command_residual": zero_frac,
        "adapter_output_std": a_out_std,
        "root_velocity_std": root_std,
        "sign_flip_freq": sign_flip,
        "mode_switching_freq": mode_switch,
        "mode_counts": mode_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="base_command_benchmark.json")
    args = ap.parse_args()

    results = {
        "raw_trained_contract": _run_chain(args.num_envs, args.steps, args.seed, adapter_on=False),
        "adapter_on": _run_chain(args.num_envs, args.steps, args.seed, adapter_on=True),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
