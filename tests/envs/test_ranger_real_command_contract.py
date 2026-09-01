"""Ranger real-command contract tests.

Verifies that Twist-like ``(vx, vy, wz)`` commands map to the motion modes a
real AgileX Ranger base node would select, and that a vy hovering around the
parallel threshold does not churn modes.  These are the sim2real command
semantics both the MuJoCo adapter and the deployed ``ranger_base_node`` share.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.envs.locomotion.ranger_box.ranger_command_adapter import (
    MODE_ACKERMAN,
    MODE_NAMES,
    MODE_PARALLEL,
    MODE_SPIN,
    MODE_STOP,
    RangerCommandAdapter,
    RangerCommandAdapterConfig,
    ranger_wheel_visualization,
)

_DT = 0.02
_WHEEL_POS = ((0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28))
_WHEEL_R = 0.152


def _adapter(cfg: RangerCommandAdapterConfig | None = None) -> RangerCommandAdapter:
    ad = RangerCommandAdapter(cfg or RangerCommandAdapterConfig(), _DT, 1)
    ad.reset(np.array([0]))
    return ad


def _mode_of(ad: RangerCommandAdapter, cmd: list[float]) -> int:
    _, mode = ad.process(np.asarray([cmd], dtype=np.float64))
    return int(mode[0])


def test_twist_straight_forward_is_ackerman():
    """linear.x=0.5, linear.y=0, angular.z=0.3 -> ACKERMAN (vy absent)."""
    ad = _adapter()
    assert _mode_of(ad, [0.5, 0.0, 0.3]) == MODE_ACKERMAN


def test_twist_lateral_is_parallel():
    """linear.x=0.5, linear.y=0.2 -> PARALLEL (vy >= enter threshold)."""
    ad = _adapter()
    assert _mode_of(ad, [0.5, 0.2, 0.0]) == MODE_PARALLEL


def test_twist_all_zero_is_stop():
    """linear.x=0, linear.y=0, angular.z=0 -> STOP."""
    ad = _adapter()
    assert _mode_of(ad, [0.0, 0.0, 0.0]) == MODE_STOP


def test_twist_spin_is_spin():
    """angular.z=0.5 with no translation -> SPIN."""
    ad = _adapter()
    assert _mode_of(ad, [0.0, 0.0, 0.5]) == MODE_SPIN


def test_vy_threshold_oscillation_no_churn():
    """vy 0.049/0.051/0.049/... does NOT toggle ACKERMAN<->PARALLEL.

    With the hysteresis (enter 0.08 / exit 0.03) a vy hovering just under the
    old single 0.05 threshold stays in ACKERMAN; it never enters PARALLEL and
    therefore never churns.
    """
    ad = _adapter()
    modes = []
    for vy in (0.049, 0.051, 0.049, 0.051, 0.049, 0.051):
        modes.append(_mode_of(ad, [0.5, vy, 0.0]))
    assert set(modes) <= {MODE_ACKERMAN}, f"mode churned: {[MODE_NAMES[m] for m in modes]}"
    assert all(m == MODE_ACKERMAN for m in modes)


def test_vy_crossing_enter_then_exit_with_dwell():
    """|vy| must exceed enter to enter PARALLEL; dwell delays rapid exit.

    Sequence: 0.02 (ACKERMAN) -> 0.09 (wants PARALLEL but dwell blocks for
    min_mode_duration) -> hold long enough -> 0.09 stays PARALLEL.
    """
    cfg = RangerCommandAdapterConfig(min_mode_duration=0.2)  # 10 steps @ 0.02
    ad = _adapter(cfg)
    assert _mode_of(ad, [0.5, 0.02, 0.0]) == MODE_ACKERMAN
    # Push |vy| above enter; the very next step is still dwell-blocked.
    assert _mode_of(ad, [0.5, 0.09, 0.0]) == MODE_ACKERMAN
    # After min_mode_duration elapses the switch to PARALLEL is allowed.
    for _ in range(11):
        _mode_of(ad, [0.5, 0.09, 0.0])
    assert _mode_of(ad, [0.5, 0.09, 0.0]) == MODE_PARALLEL
    # Once PARALLEL, a vy above exit holds PARALLEL.
    assert _mode_of(ad, [0.5, 0.05, 0.0]) == MODE_PARALLEL


def test_stop_wheel_viz_no_residual():
    """STOP produces zero wheel joint speed (no visual residual)."""
    s, w = ranger_wheel_visualization(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([MODE_STOP], dtype=np.int64),
        _WHEEL_POS,
        _WHEEL_R,
        np.full((1, 4), 0.2),
    )
    assert np.allclose(w, 0.0)
    # Steer is held, wheels fully stopped.
    assert np.allclose(s, 0.2)
