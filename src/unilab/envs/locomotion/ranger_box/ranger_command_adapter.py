"""Ranger command adapter — hardware-compatible base command shaping for RangerBox.

The real AgileX Ranger consumes ``geometry_msgs/Twist`` (``linear.x/y``,
``angular.z``) and interprets it with motion-mode semantics in
``ranger_base_node``: ``linear.y`` near zero runs Ackermann / spinning,
``linear.y != 0`` runs Parallel mode, and the node applies a command
deadband/hysteresis so near-zero commands stop cleanly.

The trained PPO policy outputs a raw normalized base action; this adapter
shapes that command exactly like the real base node would, so the kinematic
base controller in MuJoCo sees the same command the real ``/cmd_vel`` would
deliver.  Only the *command* is shaped — the kinematic base controller,
SE(2) lock, and arm layers are untouched.

Pipeline::

    PPO action[0:3] · base_weight
        ─ (scale to velocity units) →
    RangerCommandAdapter.process(v_cmd)
        deadband + hysteresis (per channel, Schmitt trigger)
        motion-mode decision  STOP / ACKERMAN / PARALLEL / SPIN
        vy force-to-zero      in ACKERMAN (real base enters Parallel otherwise)
        velocity clip         max_lin / max_ang
        ─ v_cmd_filtered + mode →
    BaseVelocityController.step_from_velocity(v_cmd)
        accel limit → (optional jerk limit) → first-order filter → clip
        ─ v_real →
    set_root_planar_velocity(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ── Motion modes (mirror ranger_base_node semantics) ─────────────────────────

MODE_STOP = 0
MODE_ACKERMAN = 1
MODE_PARALLEL = 2
MODE_SPIN = 3
MODE_NAMES = {
    MODE_STOP: "STOP",
    MODE_ACKERMAN: "ACKERMAN",
    MODE_PARALLEL: "PARALLEL",
    MODE_SPIN: "SPIN",
}


@dataclass
class RangerCommandAdapterConfig:
    """YAML-configurable command shaping (deadband / mode / limits).

    All values are in **velocity units** (m/s for linear/lateral, rad/s for
    angular) — the same domain the base controller's velocity command lives
    in, so they map 1:1 onto real ``/cmd_vel`` semantics.

    Deadband uses a Schmitt trigger: the channel turns ON once ``|v|`` exceeds
    ``*_enter`` and turns OFF once it drops below ``*_exit`` (enter > exit).
    Between the two it holds its previous state, so a hovering command just
    above ``exit`` does not chatter on/off every frame (hysteresis).
    """

    # Master gate: disable to keep the raw trained-policy command path.
    enable: bool = True
    # Deadband / hysteresis thresholds.
    linear_deadband_enter: float = 0.05  # vx ON threshold  (m/s)
    linear_deadband_exit: float = 0.03  # vx OFF threshold (m/s)
    lateral_deadband_enter: float = 0.05  # vy ON threshold  (m/s)
    lateral_deadband_exit: float = 0.03  # vy OFF threshold (m/s)
    angular_deadband_enter: float = 0.05  # wz ON threshold  (rad/s)
    angular_deadband_exit: float = 0.03  # wz OFF threshold (rad/s)
    # vy is the parallel-mode trigger on the real base: |vy| >= this enters
    # PARALLEL, |vy| < this runs ACKERMAN with vy forced to 0.
    parallel_vy_threshold: float = 0.05
    # SPIN is recognised when yaw dominates and the translation is negligible.
    spin_angular_threshold: float = 0.10  # |wz| >= this
    spin_linear_threshold: float = 0.05  # and hypot(vx, vy) < this
    # Velocity limits applied here (must match base_velocity_controller).
    max_lin_vel: float = 1.5
    max_ang_vel: float = 0.75

    # Mode-based wheel visualization (Task 6).  Only affects the visual
    # steering / wheel joints — no contact dynamics.
    mode_wheel_visualization: bool = True


class RangerCommandAdapter:
    """Shapes a base velocity command into hardware-compatible Ranger semantics.

    Vectorized over ``(N, 3)``; keeps per-env hysteresis / mode state.
    """

    def __init__(self, cfg: RangerCommandAdapterConfig, num_envs: int):
        self._cfg = cfg
        self._num_envs = num_envs
        # Schmitt-trigger state per (env, channel) — [vx, vy, wz].
        self._active = np.zeros((num_envs, 3), dtype=bool)
        self._mode = np.full(num_envs, MODE_STOP, dtype=np.int64)
        # Last output (for diagnostics / benchmark reads; no behavioural role).
        self.last_output = np.zeros((num_envs, 3), dtype=np.float64)
        self.last_mode = np.full(num_envs, MODE_STOP, dtype=np.int64)

    @property
    def enabled(self) -> bool:
        return bool(self._cfg.enable)

    @property
    def cfg(self) -> RangerCommandAdapterConfig:
        return self._cfg

    def reset(self, env_ids: np.ndarray) -> None:
        self._active[env_ids] = False
        self._mode[env_ids] = MODE_STOP

    def process(self, v_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply deadband/hysteresis + mode + vy-zero + clip.

        Args:
            v_cmd: ``(N, 3)`` base velocity command in velocity units
                (``[vx, vy, wz]`` m/s, m/s, rad/s).

        Returns:
            ``(v_filtered (N, 3), mode (N,) int)``.
        """
        cfg = self._cfg
        v = np.asarray(v_cmd, dtype=np.float64).copy()

        # 1. Deadband + hysteresis per channel.
        enter = np.array(
            [
                cfg.linear_deadband_enter,
                cfg.lateral_deadband_enter,
                cfg.angular_deadband_enter,
            ],
            dtype=np.float64,
        )
        exit_ = np.array(
            [
                cfg.linear_deadband_exit,
                cfg.lateral_deadband_exit,
                cfg.angular_deadband_exit,
            ],
            dtype=np.float64,
        )
        a = np.abs(v)
        self._active |= a > enter[None, :]
        self._active &= ~(a < exit_[None, :])
        v = np.where(self._active, v, 0.0)

        # 2. Motion-mode decision on the filtered command.  STOP means every
        # channel was deadbanded to zero (not the spin translation bound — a
        # 0.04 m/s active command must stay a real Ackermann move, not STOP).
        speed_xy = np.hypot(v[:, 0], v[:, 1])
        abs_wz = np.abs(v[:, 2])
        stopped = (np.abs(v[:, 0]) < 1e-9) & (np.abs(v[:, 1]) < 1e-9) & (abs_wz < 1e-9)
        spin = (
            (~stopped)
            & (abs_wz >= cfg.spin_angular_threshold)
            & (speed_xy < cfg.spin_linear_threshold)
        )
        parallel = (~stopped) & (~spin) & (np.abs(v[:, 1]) >= cfg.parallel_vy_threshold)
        mode = np.where(
            stopped,
            MODE_STOP,
            np.where(spin, MODE_SPIN, np.where(parallel, MODE_PARALLEL, MODE_ACKERMAN)),
        )
        self._mode[:] = mode

        # 3. ACKERMAN: the real base only delivers this when linear.y ~ 0, so
        # force vy to 0 here — otherwise sim would send vy the base rejects.
        ackerman = mode == MODE_ACKERMAN
        v[ackerman, 1] = 0.0

        # 4. Clip to the hardware velocity limits.
        v[:, 0:2] = np.clip(v[:, 0:2], -cfg.max_lin_vel, cfg.max_lin_vel)
        v[:, 2] = np.clip(v[:, 2], -cfg.max_ang_vel, cfg.max_ang_vel)

        self.last_output[:] = v
        self.last_mode[:] = mode
        return v, mode


def ranger_wheel_visualization(
    v_cmd: np.ndarray,
    mode: np.ndarray,
    wheel_positions: tuple[tuple[float, float], ...],
    wheel_radius: float,
    prev_steer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mode-based wheel steering / spin for the visual joints only.

    Mirrors the real Ranger's Ackermann / Parallel / Spin behaviour instead of
    the generic per-wheel swerve IK.  ``x > 0`` is the front axle.

    Args:
        v_cmd: ``(N, 3)`` final base velocity (``[vx, vy, wz]``).
        mode: ``(N,)`` ``MODE_*``.
        wheel_positions: 4 wheel ``(x, y)`` in base frame.
        wheel_radius: wheel radius (m).
        prev_steer: ``(N, 4)`` previous steer angles (held while stopped).

    Returns:
        ``(steer, omega)`` both ``(N, 4)``, ``steer`` in ``[-pi/2, pi/2]``.
    """
    pos = np.asarray(wheel_positions, dtype=np.float64)  # (4, 2)
    x, y = pos[:, 0], pos[:, 1]
    r = float(wheel_radius)
    N = v_cmd.shape[0]
    vx = v_cmd[:, 0]
    vy = v_cmd[:, 1]
    wz = v_cmd[:, 2]

    steer = prev_steer.copy()
    omega = np.zeros((N, 4), dtype=np.float64)

    front = x > 0.0  # (4,)

    for i in range(N):
        m = int(mode[i])
        if m == MODE_STOP:
            # Hold steering, wheels stopped.
            omega[i] = 0.0
            continue
        if m == MODE_SPIN:
            # Pure rotation about the base centre.
            vi_x = -wz[i] * y
            vi_y = wz[i] * x
        elif m == MODE_PARALLEL:
            # All four wheels crab at the same angle.
            vi_x = np.full(4, vx[i])
            vi_y = np.full(4, vy[i])
        else:  # MODE_ACKERMAN
            # vy forced to 0 by the adapter; front wheels steer on the ICR
            # geometry, rear wheels stay straight (car-like).
            vi_x = vx[i] - wz[i] * y
            vi_y = wz[i] * x
            rear = ~front
            vi_x[rear] = vx[i]
            vi_y[rear] = 0.0

        speed = np.hypot(vi_x, vi_y)
        s = np.arctan2(vi_y, vi_x)
        om = np.where(speed > 1e-6, speed / r, 0.0)

        # Map steer into [-pi/2, pi/2], flipping omega sign on the wrap so the
        # wheel keeps rolling in the equivalent direction.
        flip = s > np.pi / 2
        s = np.where(flip, s - np.pi, s)
        flip2 = s < -np.pi / 2
        s = np.where(flip2, s + np.pi, s)
        om = np.where(flip | flip2, -om, om)

        steer[i] = s
        omega[i] = om

    return steer, omega
