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
) -> tuple[np.ndarray, np.ndarray]:
    """Compute steer angles and wheel angular velocities from base velocity.

    Args:
        v_real: ``(N, 3)`` — vx, vy, vyaw in base frame.
        wheel_positions: ``((x, y), ...)`` — 4 wheel positions in base frame.
        wheel_radius: wheel radius in meters.

    Returns:
        ``(steer, omega)`` — both ``(N, 4)`` float64.
    """
    pos = np.asarray(wheel_positions, dtype=np.float64)  # (4, 2)
    x = pos[:, 0]  # (4,)
    y = pos[:, 1]  # (4,)

    vx_i = v_real[:, 0:1] - v_real[:, 2:3] * y[None, :]   # (N, 4)
    vy_i = v_real[:, 1:2] + v_real[:, 2:3] * x[None, :]   # (N, 4)

    steer = np.arctan2(vy_i, vx_i)          # (N, 4)
    omega = np.sqrt(vx_i**2 + vy_i**2) / wheel_radius  # (N, 4)
    return steer, omega


class BaseVelocityController:
    """Vectorized base velocity controller with latency, acceleration,
    first-order response, noise, and wheel visualization.

    Operates on ``(N, 3)`` tensors for N parallel envs.
    """

    def __init__(
        self,
        cfg,                # BaseVelocityControllerConfig
        dt: float,
        backend,            # SimBackend
        asset_cfg,          # RangerBoxAsset
        num_envs: int,
    ):
        self._cfg = cfg
        self._dt = dt
        self._backend = backend
        self._asset = asset_cfg
        self._num_envs = num_envs

        self.v_real = np.zeros((num_envs, 3), dtype=np.float64)
        self.latency_ring = np.zeros(
            (cfg.max_latency_steps + 1, num_envs, 3), dtype=np.float64
        )
        self.latency_steps = np.zeros(num_envs, dtype=np.int32)
        self.latency_write_ptr = np.zeros(num_envs, dtype=np.int32)

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

    def step(self, action_base_vel: np.ndarray) -> None:
        """Compute v_real from policy action (steps 1-7).
        Does NOT write to backend — apply_velocity() does that.

        Args:
            action_base_vel: ``(N, 3)`` — policy output in [-1, 1].
        """
        N = self._num_envs
        cfg = self._cfg
        dt = self._dt

        # --- 1. Scale ---
        v_cmd = action_base_vel.astype(np.float64).copy()
        v_cmd[:, 0:2] *= cfg.action_scale_lin
        v_cmd[:, 2] *= cfg.action_scale_ang

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
        v_target = self.v_real + dv

        # --- 5. First-order response ---
        alpha = dt / (cfg.tau + dt)
        self.v_real = self.v_real + alpha * (v_target - self.v_real)

        # --- 6. Noise (after first-order) ---
        if cfg.enable_noise:
            noise = np.random.standard_normal((N, 3)).astype(np.float64)
            noise[:, 0:2] *= cfg.action_noise_scale * cfg.max_lin_vel
            noise[:, 2] *= cfg.action_noise_scale * cfg.max_ang_vel
            self.v_real = self.v_real + noise

        # --- 7. Final clip ---
        self.v_real = np.clip(self.v_real, -self._max_vel_arr, self._max_vel_arr)

    def apply_velocity(self) -> None:
        """Write v_real to backend via set_root_planar_velocity + wheel IK.
        Called AFTER physics integration (in update_state).
        """
        N = self._num_envs
        cfg = self._cfg

        # --- 8. Wheel visualization (before world-frame conversion) ---
        if cfg.enable_wheel_visualization:
            steer, omega = _compute_wheel_ik(
                self.v_real, self._asset.wheel_positions, self._asset.wheel_radius
            )
            self._backend.set_joint_qpos(list(self._asset.steering_joint_names), steer)
            self._backend.set_joint_qvel(list(self._asset.wheel_joint_names), omega)

        # --- 9. World-frame conversion ---
        base_quat = self._backend.get_sensor_data("imu-framequat")

        v_body = np.concatenate(
            [self.v_real[:, 0:2], np.zeros((N, 1))], axis=1, dtype=np.float64
        )
        w_body = np.concatenate(
            [np.zeros((N, 2)), self.v_real[:, 2:3]], axis=1, dtype=np.float64
        )

        from unilab.envs.common.rotation import np_quat_apply_batched
        v_world = np_quat_apply_batched(base_quat, v_body)
        w_world = np_quat_apply_batched(base_quat, w_body)

        self._backend.set_root_planar_velocity(
            v_world[:, :2], w_world[:, 2], preserve_uncontrolled=True
        )
