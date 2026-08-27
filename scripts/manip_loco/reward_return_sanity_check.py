"""Reward-return sanity check for RangerBoxReach Run 4A reward semantics.

Run:  uv run scripts/manip_loco/reward_return_sanity_check.py

Compares the cumulative return of three HAND-CODED EE-distance trajectories
using the SAME reward scales as the env (read from the owner YAML):

  A  fast approach → hold 10 cm for 0.5 s (the intended terminal success)
  B  repeated crossing of the 10 cm boundary (the Run-3 exploit: in/out/in/out)
  C  lingering at 10-15 cm forever, never completing

Requires the event-based reward semantics of Run 4A:
  - success_once_10cm / success_once_05cm fire ONCE per episode (first entry)
  - success_hold_10cm fires once, on the step held-success is achieved, and the
    episode terminates on that same step
  - ee_distance is 0 (no per-step hovering bonus near the goal)

We assert  Return(A) > Return(B)  and  Return(A) > Return(C).
Distance/success terms dominate; other reward terms (action rate, arm velocity,
etc.) only penalise the noisy crossing trajectory further, so this is a
conservative lower bound on A's advantage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OWNER_YAML = ROOT_DIR / "conf/ppo/task/ranger_box_reach/mujoco.yaml"

CTRL_DT = 0.02
HOLD_STEPS = int(0.5 / CTRL_DT)  # 25
EPISODE_STEPS = 500  # 10 s


def _load_scales() -> dict[str, float]:
    with open(OWNER_YAML) as f:
        data = yaml.safe_load(f)
    return {k: float(v) for k, v in data["reward"]["scales"].items()}


def _series_return(d_series: list[float], scales: dict[str, float]) -> tuple[float, int]:
    """Cumulative reward for an EE-distance trajectory.

    Returns (total, steps_used).  Episodes terminate on the hold step.
    """
    once10 = once05 = False
    hold_timer = 0
    total = 0.0
    prev_d: float | None = None
    for t, d in enumerate(d_series):
        r = 0.0
        # continuous (integrated over ctrl_dt)
        if prev_d is not None:
            r += scales["ee_progress"] * (prev_d - d) * CTRL_DT
        r += scales["ee_distance_l2"] * (d * d) * CTRL_DT
        r += scales["ee_distance"] * 0.0  # informational (0.0 in Run 4A)
        # events (unscaled)
        if d < 0.10 and not once10:
            r += scales["success_once_10cm"]
            once10 = True
        if d < 0.05 and not once05:
            r += scales["success_once_05cm"]
            once05 = True
        if d < 0.10:
            hold_timer += 1
        else:
            hold_timer = 0
        if hold_timer >= HOLD_STEPS:
            r += scales["success_hold_10cm"]
            return total + r, t + 1
        total += r
        prev_d = d
    return total, EPISODE_STEPS


def _traj_hold_fast() -> list[float]:
    """A: approach to 8 cm, hold 0.5 s, terminate."""
    d = [0.50 - 0.084 * i for i in range(5)]  # 0.50 → 0.08 over 5 steps
    return d + [0.08] * HOLD_STEPS  # hold 25 steps → bonus + terminate


def _traj_crossing() -> list[float]:
    """B: oscillate 0.090 ↔ 0.115, crossing 10 cm every few steps, never hold."""
    out = []
    for _ in range(EPISODE_STEPS):
        out.extend([0.090, 0.115, 0.092, 0.118, 0.088, 0.112])  # 6-step period
    return out[:EPISODE_STEPS]


def _traj_linger() -> list[float]:
    """C: approach to 0.12 m and stay there forever (> 10 cm, no success)."""
    return [0.50 - 0.076 * i for i in range(5)] + [0.12] * (EPISODE_STEPS - 5)


def main() -> None:
    scales = _load_scales()
    print("Reward scales (from owner YAML):")
    for k in (
        "ee_distance",
        "ee_progress",
        "ee_distance_l2",
        "success_once_10cm",
        "success_once_05cm",
        "success_hold_10cm",
    ):
        print(f"  {k:<20} {scales[k]:>8.3f}")
    print()

    cases = {
        "A hold-success": (_traj_hold_fast(), "hold 10 cm for 0.5 s then terminate"),
        "B 10cm crossing": (_traj_crossing(), "repeatedly cross the 10 cm boundary"),
        "C 10-15cm linger": (_traj_linger(), "hover at 12 cm, never complete"),
    }
    results: dict[str, tuple[float, int]] = {}
    for name, (traj, desc) in cases.items():
        ret, steps = _series_return(traj, scales)
        results[name] = (ret, steps)
        print(f"  {name:<18} return {ret:>9.3f}  ({steps} steps)  {desc}")

    ret_a, _ = results["A hold-success"]
    ret_b, _ = results["B 10cm crossing"]
    ret_c, _ = results["C 10-15cm linger"]

    print()
    ok = ret_a > ret_b and ret_a > ret_c
    print(f"  Return(A) = {ret_a:.3f}  vs  Return(B) = {ret_b:.3f}  vs  Return(C) = {ret_c:.3f}")
    print(f"  A > B: {ret_a > ret_b}    A > C: {ret_a > ret_c}")
    print()
    print(
        "PASS — hold-success dominates the exploit trajectories"
        if ok
        else "FAIL — reward semantics allow the exploit"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
