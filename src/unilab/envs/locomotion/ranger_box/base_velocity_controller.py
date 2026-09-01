"""Vectorized base velocity controller (A+ scheme) for RangerBox.

Latency ring buffer, acceleration clip, first-order response, noise,
and wheel IK — all on (N, 3) tensors.
"""

from __future__ import annotations

import numpy as np


def _compute_wheel_ik(
    v_real: np.ndarray,
    wheel_positions: tuple[tuple[float, float], ...],
    wheel_radius: float,
    prev_steer: np.ndarray | None = None,
    deadband: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute steer angles and wheel angular velocities from base velocity.

    Args:
        v_real: ``(N, 3)`` — vx, vy, vyaw in base frame.
        wheel_positions: ``((x, y), ...)`` — 4 wheel positions in base frame.
        wheel_radius: wheel radius in meters.
        prev_steer: ``(N, 4)`` — previous steer angles, held when speed is
            below the deadband.  Pass ``None`` to disable hold behaviour.
        deadband: linear speed threshold (m/s).  Below this, omega is zeroed
            and steer holds its previous value.

    Returns:
        ``(steer, omega)`` — both ``(N, 4)`` float64, with ``steer`` in
        ``[-pi/2, pi/2]``.
    """
    pos = np.asarray(wheel_positions, dtype=np.float64)  # (4, 2)
    x = pos[:, 0]  # (4,)
    y = pos[:, 1]  # (4,)

    vx_i = v_real[:, 0:1] - v_real[:, 2:3] * y[None, :]  # (N, 4)
    vy_i = v_real[:, 1:2] + v_real[:, 2:3] * x[None, :]  # (N, 4)

    speed = np.sqrt(vx_i**2 + vy_i**2)  # (N, 4)
    steer = np.arctan2(vy_i, vx_i)  # [-pi, pi]
    omega = speed / wheel_radius

    # Map steer into [-pi/2, pi/2]; when we subtract/add pi, flip omega sign
    # so the wheel still rolls in the equivalent direction.
    flip = steer > np.pi / 2
    steer = np.where(flip, steer - np.pi, steer)
    flip2 = steer < -np.pi / 2
    steer = np.where(flip2, steer + np.pi, steer)
    omega = np.where(flip | flip2, -omega, omega)

    # Deadband: hold previous steer and zero omega when nearly stationary so
    # atan2(noise, noise) doesn't jitter all four wheels every frame.
    if prev_steer is not None:
        moving = speed >= deadband
        steer = np.where(moving, steer, prev_steer)
        omega = np.where(moving, omega, 0.0)

    return steer, omega


class BaseVelocityController:
    """Vectorized base velocity controller with latency, acceleration,
    first-order response, noise, and wheel visualization.

    Operates on ``(N, 3)`` tensors for N parallel envs.
    """

    def __init__(
        self,
        cfg,  # BaseVelocityControllerConfig
        dt: float,
        backend,  # SimBackend
        asset_cfg,  # RangerBoxAsset
        num_envs: int,
    ):
        self._cfg = cfg
        self._dt = dt
        self._backend = backend
        self._asset = asset_cfg
        self._num_envs = num_envs

        self.v_real = np.zeros((num_envs, 3), dtype=np.float64)
        self.latency_ring = np.zeros((cfg.max_latency_steps + 1, num_envs, 3), dtype=np.float64)
        self.latency_steps = np.zeros(num_envs, dtype=np.int32)
        self.latency_write_ptr = np.zeros(num_envs, dtype=np.int32)
        self._prev_steer = np.zeros((num_envs, 4), dtype=np.float64)
        # Optional jerk limiter state (acceleration derivative bound).
        self._prev_accel = np.zeros((num_envs, 3), dtype=np.float64)

        # Cache initial base z height for SE(2) planar lock
        init_qpos = backend.get_keyframe_qpos("home")
        self._init_base_z = float(init_qpos[2])  # z is index 2 in qpos

        # Pre-compute clip arrays
        self._max_vel_arr = np.array(
            [cfg.max_lin_vel, cfg.max_lin_vel, cfg.max_ang_vel], dtype=np.float64
        )
        self._max_acc_arr = np.array(
            [cfg.max_lin_acc, cfg.max_lin_acc, cfg.max_ang_acc], dtype=np.float64
        )

    def reset(self, env_ids: np.ndarray, rng: np.random.Generator) -> None:
        self.latency_steps[env_ids] = rng.integers(
            0, self._cfg.max_latency_steps + 1, size=len(env_ids)
        )
        self.latency_write_ptr[env_ids] = 0
        self.latency_ring[:, env_ids, :] = 0.0
        self.v_real[env_ids] = 0.0
        self._prev_steer[env_ids] = 0.0
        self._prev_accel[env_ids] = 0.0

    def step(self, action_base_vel: np.ndarray) -> None:
        """Advance from a raw policy action in [-1, 1] (steps 1-7).

        Args:
            action_base_vel: ``(N, 3)`` — policy output in [-1, 1].
        """
        cfg = self._cfg
        v_cmd = action_base_vel.astype(np.float64).copy()
        v_cmd[:, 0:2] *= cfg.action_scale_lin
        v_cmd[:, 2] *= cfg.action_scale_ang
        self._advance(v_cmd)

    def step_from_velocity(self, v_cmd: np.ndarray) -> None:
        """Advance from an already velocity-unit command (RangerCommandAdapter).

        The command-shaping adapter (deadband/mode) produces a velocity in
        m/s·rad/s, so the internal action-scale step is skipped.

        Args:
            v_cmd: ``(N, 3)`` — base velocity command ``[vx, vy, wz]``.
        """
        self._advance(np.asarray(v_cmd, dtype=np.float64).copy())

    def _advance(self, v_cmd: np.ndarray) -> None:
        """Shared velocity-shaping pipeline (steps 2-7)."""
        N = self._num_envs
        cfg = self._cfg
        dt = self._dt

        # --- 2. Clip ---
        v_cmd[:, 0:2] = np.clip(v_cmd[:, 0:2], -cfg.max_lin_vel, cfg.max_lin_vel)
        v_cmd[:, 2] = np.clip(v_cmd[:, 2], -cfg.max_ang_vel, cfg.max_ang_vel)

        # --- 3. Latency ring ---
        if cfg.enable_latency:
            L = self.latency_ring.shape[0]
            wp = self.latency_write_ptr
            self.latency_ring[wp, np.arange(N), :] = v_cmd
            rp = (wp - self.latency_steps) % L
            v_cmd = self.latency_ring[rp, np.arange(N), :].copy()
            self.latency_write_ptr = (wp + 1) % L

        # --- 4. Acceleration limit ---
        dv = v_cmd - self.v_real
        dv[:, 0:2] = np.clip(dv[:, 0:2], -cfg.max_lin_acc * dt, cfg.max_lin_acc * dt)
        dv[:, 2] = np.clip(dv[:, 2], -cfg.max_ang_acc * dt, cfg.max_ang_acc * dt)

        # --- 4b. Optional jerk limit (da/dt bound) — after the accel clip ---
        if getattr(cfg, "enable_jerk_limit", False):
            dv = self._apply_jerk_limit(dv, dt)

        v_target = self.v_real + dv

        # --- 5. First-order response (filtered state, NO noise) ---
        alpha = dt / (cfg.tau + dt)
        self.v_real = self.v_real + alpha * (v_target - self.v_real)

        # --- 6. Noise applied to execution velocity only (NOT accumulated) ---
        v_exec = self.v_real.copy()
        if cfg.enable_noise:
            noise = np.random.standard_normal((N, 3)).astype(np.float64)
            noise[:, 0:2] *= cfg.action_noise_scale * cfg.max_lin_vel
            noise[:, 2] *= cfg.action_noise_scale * cfg.max_ang_vel
            v_exec = v_exec + noise

        # --- 7. Final clip ---
        v_exec = np.clip(v_exec, -self._max_vel_arr, self._max_vel_arr)

        # Store execution velocity for world-frame conversion
        self._v_exec = v_exec

    def _apply_jerk_limit(self, dv: np.ndarray, dt: float) -> np.ndarray:
        """Bound the change in acceleration (da/dt) between control steps.

        ``jerk = dv/dt`` is the current acceleration; the limiter keeps
        ``|a_new - a_prev| <= max_jerk * dt`` so the command cannot jerk
        between steps.  Zero ``max_*_jerk`` disables that channel.
        """
        cfg = self._cfg
        jerk = np.array([cfg.max_lin_jerk, cfg.max_lin_jerk, cfg.max_ang_jerk], dtype=np.float64)
        if np.all(jerk <= 0.0):
            return dv
        a_new = dv / dt
        a_prev = self._prev_accel
        a_lim = a_prev + np.clip(a_new - a_prev, -jerk[None, :] * dt, jerk[None, :] * dt)
        self._prev_accel[:] = a_lim
        return a_lim * dt

    def apply_velocity(self, mode: np.ndarray | None = None) -> None:
        """Write v_exec to backend via set_root_planar_velocity + wheel viz.
        Called BEFORE physics step (in apply_action).

        Args:
            mode: ``(N,)`` motion mode from ``RangerCommandAdapter`` (``MODE_*``),
                or ``None`` for the legacy per-wheel swerve wheel IK.
        """
        N = self._num_envs
        cfg = self._cfg
        v_apply = getattr(self, "_v_exec", self.v_real)

        # --- 8. Wheel visualization (before world-frame conversion) ---
        if cfg.enable_wheel_visualization:
            if mode is not None and getattr(cfg, "mode_wheel_visualization", False):
                from .ranger_command_adapter import ranger_wheel_visualization

                steer, omega = ranger_wheel_visualization(
                    v_apply,
                    mode,
                    self._asset.wheel_positions,
                    self._asset.wheel_radius,
                    self._prev_steer,
                )
            else:
                steer, omega = _compute_wheel_ik(
                    v_apply,
                    self._asset.wheel_positions,
                    self._asset.wheel_radius,
                    prev_steer=self._prev_steer,
                )
            self._prev_steer[:] = steer
            self._backend.set_joint_qpos(list(self._asset.steering_joint_names), steer)
            self._backend.set_joint_qvel(list(self._asset.wheel_joint_names), omega)

        # --- 9. World-frame conversion ---
        base_quat = self._backend.get_sensor_data("imu-framequat")

        v_body = np.concatenate([v_apply[:, 0:2], np.zeros((N, 1))], axis=1, dtype=np.float64)
        w_body = np.concatenate([np.zeros((N, 2)), v_apply[:, 2:3]], axis=1, dtype=np.float64)

        from unilab.envs.common.rotation import np_quat_apply_batched

        v_world = np_quat_apply_batched(base_quat, v_body)
        w_world = np_quat_apply_batched(base_quat, w_body)

        self._backend.set_root_planar_velocity(
            v_world[:, :2], w_world[:, 2], preserve_uncontrolled=True
        )
