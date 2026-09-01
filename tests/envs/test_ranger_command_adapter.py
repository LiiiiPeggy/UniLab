"""Unit tests for the Ranger hardware-compatible command adapter.

Covers the deadband + hysteresis (Schmitt trigger), motion-mode decision,
vy force-to-zero in Ackermann, the mode-based wheel visualization, and the
optional jerk limiter / velocity-command path on the base controller.
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

_WHEEL_POS = ((0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28))
_WHEEL_R = 0.152


def _make(cfg: RangerCommandAdapterConfig | None = None, n: int = 4) -> RangerCommandAdapter:
    ad = RangerCommandAdapter(cfg or RangerCommandAdapterConfig(), n)
    ad.reset(np.arange(n))
    return ad


def test_deadband_zeroes_below_exit():
    """Commands below the deadband exit are zeroed."""
    ad = _make(n=2)
    v, mode = ad.process(np.array([[0.01, -0.02, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64))
    assert np.allclose(v, 0.0)
    assert set(mode.tolist()) == {MODE_STOP}


def test_hysteresis_holds_between_enter_and_exit():
    """Between exit and enter the channel holds its previous state."""
    ad = _make(n=1)
    # 0.04 is between exit(0.03) and enter(0.05); starts inactive.
    v, mode = ad.process(np.array([[0.04, 0.0, 0.0]], dtype=np.float64))
    assert np.allclose(v[0], 0.0) and mode[0] == MODE_STOP
    # 0.06 exceeds enter -> active.
    v, mode = ad.process(np.array([[0.06, 0.0, 0.0]], dtype=np.float64))
    assert v[0, 0] == pytest.approx(0.06) and mode[0] == MODE_ACKERMAN
    # Back to 0.04 (between) -> still active (hysteresis).
    v, mode = ad.process(np.array([[0.04, 0.0, 0.0]], dtype=np.float64))
    assert v[0, 0] == pytest.approx(0.04) and mode[0] == MODE_ACKERMAN
    # 0.01 below exit -> off.
    v, mode = ad.process(np.array([[0.01, 0.0, 0.0]], dtype=np.float64))
    assert np.allclose(v[0], 0.0) and mode[0] == MODE_STOP


def test_mode_decision_all_modes():
    """STOP / ACKERMAN / PARALLEL / SPIN classification."""
    ad = _make(n=5)
    cmds = np.array(
        [
            [0.0, 0.0, 0.0],  # STOP
            [0.5, 0.0, 0.0],  # ACKERMAN straight
            [0.5, 0.0, 0.3],  # ACKERMAN turn (vy absent)
            [0.4, 0.2, 0.0],  # PARALLEL (|vy| >= threshold)
            [0.0, 0.0, 0.5],  # SPIN
        ],
        dtype=np.float64,
    )
    v, mode = ad.process(cmds)
    assert mode.tolist() == [MODE_STOP, MODE_ACKERMAN, MODE_ACKERMAN, MODE_PARALLEL, MODE_SPIN]
    assert set(MODE_NAMES[m] for m in mode) == {"STOP", "ACKERMAN", "PARALLEL", "SPIN"}


def test_vy_forced_zero_in_ackerman():
    """In ACKERMAN the lateral command is forced to 0 (real base rejects vy)."""
    ad = _make(n=2)
    cmds = np.array([[0.5, 0.04, 0.3], [0.5, 0.2, 0.0]], dtype=np.float64)  # vy<thr and vy>=thr
    v, mode = ad.process(cmds)
    # env0 |vy| 0.04 < 0.05 -> ACKERMAN, vy zeroed.
    assert mode[0] == MODE_ACKERMAN and v[0, 1] == 0.0
    assert v[0, 0] == pytest.approx(0.5) and v[0, 2] == pytest.approx(0.3)
    # env1 |vy| 0.2 >= 0.05 -> PARALLEL, vy kept.
    assert mode[1] == MODE_PARALLEL and v[1, 1] == pytest.approx(0.2)


def test_velocity_clip_applies_limits():
    """Adapter clips to max_lin / max_ang and never exceeds them."""
    cfg = RangerCommandAdapterConfig(max_lin_vel=1.0, max_ang_vel=0.5)
    ad = _make(cfg)
    v, _ = ad.process(np.array([[2.0, -2.0, 2.0]], dtype=np.float64))
    assert v[0, 0] == pytest.approx(1.0) and v[0, 1] == pytest.approx(-1.0)
    assert v[0, 2] == pytest.approx(0.5)


def test_wheel_viz_stop_holds_steer():
    """STOP: wheel speed zero, steering held at the previous value."""
    prev = np.full((1, 4), 0.3)
    s, w = ranger_wheel_visualization(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        np.array([MODE_STOP], dtype=np.int64),
        _WHEEL_POS,
        _WHEEL_R,
        prev,
    )
    assert np.allclose(w, 0.0)
    assert np.allclose(s, prev)


def test_wheel_viz_parallel_same_angle():
    """PARALLEL: all four wheels crab at the same angle."""
    s, w = ranger_wheel_visualization(
        np.array([[0.4, 0.2, 0.0]], dtype=np.float64),
        np.array([MODE_PARALLEL], dtype=np.int64),
        _WHEEL_POS,
        _WHEEL_R,
        np.zeros((1, 4)),
    )
    assert np.allclose(s, s[0, 0])  # identical steer
    assert np.allclose(w, w[0, 0])  # identical speed
    expected = float(np.arctan2(0.2, 0.4))
    assert s[0, 0] == pytest.approx(expected, abs=1e-6)


def test_wheel_viz_spin_opposite_sides():
    """SPIN: left/right wheels roll in opposite directions."""
    s, w = ranger_wheel_visualization(
        np.array([[0.0, 0.0, 0.5]], dtype=np.float64),
        np.array([MODE_SPIN], dtype=np.int64),
        _WHEEL_POS,
        _WHEEL_R,
        np.zeros((1, 4)),
    )
    # Front pair (x>0) and rear pair roll opposite the other side.
    assert np.sign(w[0, 0]) != np.sign(w[0, 1])  # fr vs fl (y -0.28 vs +0.28)
    assert np.sign(w[0, 2]) != np.sign(w[0, 3])
    assert not np.allclose(w[0, 0], 0.0)


def test_wheel_viz_ackerman_rear_straight():
    """ACKERMAN: rear wheels stay straight, front wheels steer by ICR."""
    s, w = ranger_wheel_visualization(
        np.array([[0.5, 0.0, 0.3]], dtype=np.float64),
        np.array([MODE_ACKERMAN], dtype=np.int64),
        _WHEEL_POS,
        _WHEEL_R,
        np.zeros((1, 4)),
    )
    rear = [2, 3]  # x < 0
    assert np.allclose(s[0, rear], 0.0)
    assert np.all(s[0, 0:2] > 0.0)  # front wheels steer into the turn
    assert np.all(w[0] > 0.0)  # all roll forward


def test_controller_step_from_velocity_and_jerk_limit():
    """BaseVelocityController advances from velocity units; jerk limits da/dt."""
    from dataclasses import dataclass

    from unilab.envs.locomotion.ranger_box.base_velocity_controller import BaseVelocityController

    @dataclass
    class FakeBackend:
        def get_keyframe_qpos(self, name):
            return np.zeros(7)

    @dataclass
    class _Cfg:
        max_lin_vel = 1.5
        max_ang_vel = 0.78
        action_scale_lin = 1.5
        action_scale_ang = 1.0
        tau = 0.05
        max_lin_acc = 1.5
        max_ang_acc = 3.0
        max_latency_steps = 4
        action_noise_scale = 0.05
        enable_latency = False
        enable_noise = False
        enable_wheel_visualization = False
        enable_jerk_limit = True
        max_lin_jerk = 0.5
        max_ang_jerk = 0.5
        mode_wheel_visualization = True

    class _Asset:
        wheel_positions = _WHEEL_POS
        wheel_radius = _WHEEL_R

    backend = FakeBackend()
    ctrl = BaseVelocityController(_Cfg(), 0.02, backend, _Asset(), 2)
    ctrl.reset(np.array([0, 1]), np.random.default_rng(0))
    # Raw action path still scales; velocity path does not double-scale.
    ctrl.step(np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64))
    assert ctrl.v_real.shape == (2, 3)
    assert np.all(ctrl.v_real >= 0) and np.all(ctrl.v_real <= 1.5)
    # Jerk-limited: v_real cannot change faster than the accel+jerk bound allows.
    prev = ctrl.v_real.copy()
    ctrl.step_from_velocity(np.array([[1.5, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64))
    max_dv = np.abs(ctrl.v_real - prev).max()
    # max_lin_acc*dt = 1.5*0.02 = 0.03; jerk limits the accel change, so the
    # one-step velocity change stays bounded by the accel clip.
    assert max_dv <= 0.03 + 1e-9
