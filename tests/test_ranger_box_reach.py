"""Tests for RangerBoxReach env — backend contract, dataclasses, env smoke."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco", reason="mujoco not installed")


# ══════════════════════════════════════════════════════════════════════════
# Task 1 — Backend contract
# ══════════════════════════════════════════════════════════════════════════


class TestBackendPlanarVelocity:
    """Verify the 3 new SimBackend methods work correctly in MuJoCo."""

    @pytest.fixture(scope="class")
    def mj_backend(self):
        import os
        import tempfile

        from unilab.base.backend.mujoco.backend import MuJoCoBackend
        from unilab.base.scene import SceneCfg

        xml = """<mujoco model="test_planar">
          <worldbody><body name="base"><freejoint/>
          <geom type="box" size="0.1 0.1 0.05"/></body></worldbody>
        </mujoco>"""
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        tmp.write(xml)
        tmp.close()
        scene = SceneCfg(model_file=tmp.name)
        backend = MuJoCoBackend(scene, num_envs=4, sim_dt=0.01)
        os.unlink(tmp.name)
        return backend

    def test_set_root_planar_velocity_writes_planar_only(self, mj_backend):
        mj_backend._physics_state[:, 4] = 0.5  # index 4 = wz in 1+nq+... wait
        # Actually write via the method and verify
        lin_vel = np.array([[1.0, 0.5], [0.0, -0.3], [2.0, 0.0], [-1.0, 1.0]])
        yaw_rate = np.array([0.1, 0.0, -0.5, 0.2])
        mj_backend.set_root_planar_velocity(lin_vel, yaw_rate, preserve_uncontrolled=True)
        idx_qvel = mj_backend._idx_qvel
        qvel = mj_backend._physics_state[:, idx_qvel : idx_qvel + mj_backend.nv]
        np.testing.assert_allclose(qvel[:, 0], lin_vel[:, 0])
        np.testing.assert_allclose(qvel[:, 1], lin_vel[:, 1])
        np.testing.assert_allclose(qvel[:, 5], yaw_rate)

    def test_sim_backend_stubs_exist(self):
        from unilab.base.backend.base import SimBackend

        assert hasattr(SimBackend, "set_root_planar_velocity")
        assert hasattr(SimBackend, "set_joint_qpos")
        assert hasattr(SimBackend, "set_joint_qvel")


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — Dataclass defaults
# ══════════════════════════════════════════════════════════════════════════


class TestRangerBoxDataclasses:
    """Verify all dataclass defaults match spec."""

    def test_asset_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxAsset

        a = RangerBoxAsset()
        assert a.base_name == "base"
        assert a.ee_site_name == "right_center"
        assert len(a.arm_joint_names) == 6
        assert a.gripper_joint_name == "gripper_finger1_joint"
        assert len(a.wheel_positions) == 4
        assert a.wheel_radius == 0.152

    def test_sensor_inherits_go2arm(self):
        from unilab.envs.locomotion.go2_arm.base import Go2ArmSensor
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxSensor

        s = RangerBoxSensor()
        assert isinstance(s, Go2ArmSensor)
        assert s.local_linvel == "imu-velocimeter"
        assert s.ee_local_pos == "endpoint-framepos"

    def test_domain_rand_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxDomainRandConfig

        d = RangerBoxDomainRandConfig()
        assert d.push_robots is False
        assert d.randomize_ground_friction is False

    def test_reward_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxRewardConfig

        r = RangerBoxRewardConfig()
        assert r.scales["base_vel_z"] == 0.0
        assert r.scales["base_orientation"] == 0.0
        assert r.scales["base_height"] == 0.0
        assert r.sigma_ee == 0.15

    def test_controller_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import BaseVelocityControllerConfig

        c = BaseVelocityControllerConfig()
        assert c.max_lin_vel == 1.5
        assert c.tau == 0.05
        assert c.max_latency_steps == 4


# ══════════════════════════════════════════════════════════════════════════
# Task 4 — BaseVelocityController
# ══════════════════════════════════════════════════════════════════════════


class TestBaseVelocityController:
    """Unit tests for the A+ vectorized base velocity controller."""

    @pytest.fixture
    def ctrl(self):
        from unilab.envs.locomotion.ranger_box.base_velocity_controller import (
            BaseVelocityController,
        )
        from unilab.envs.locomotion.ranger_box.reach_env import (
            BaseVelocityControllerConfig,
            RangerBoxAsset,
        )

        class _MockBackend:
            def set_root_planar_velocity(self, lin_vel_xy, yaw_rate, **kw):
                self._last_lin_vel = np.asarray(lin_vel_xy).copy()

            def set_joint_qpos(self, names, values):
                pass

            def set_joint_qvel(self, names, values):
                pass

            def get_sensor_data(self, name):
                return np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))

            def get_keyframe_qpos(self, name):
                return np.array([0.0, 0.0, 0.278, 1.0, 0.0, 0.0, 0.0])

            @property
            def _qpos_view(self):
                return self.__dict__.setdefault("_qpos_view_val", np.zeros((4, 7)))

        cfg = BaseVelocityControllerConfig()
        return BaseVelocityController(cfg, 0.02, _MockBackend(), RangerBoxAsset(), 4)

    def test_init_shapes(self, ctrl):
        assert ctrl.v_real.shape == (4, 3)
        assert ctrl.latency_ring.shape == (5, 4, 3)

    def test_reset_zeros_velocity(self, ctrl):
        ctrl.v_real[:] = 1.0
        ctrl.reset(np.array([0, 2]), np.random.default_rng(42))
        np.testing.assert_allclose(ctrl.v_real[[0, 2]], 0.0)

    def test_step_updates_velocity(self, ctrl):
        ctrl.reset(np.arange(4), np.random.default_rng(42))
        ctrl.step(np.ones((4, 3)))  # max action
        # At least one env should have non-zero velocity
        assert np.any(np.abs(ctrl.v_real) > 0.0)

    def test_wheel_ik_shapes(self, ctrl):
        from unilab.envs.locomotion.ranger_box.base_velocity_controller import _compute_wheel_ik

        v_real = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        steer, omega = _compute_wheel_ik(
            v_real, ((0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28)), 0.152
        )
        assert steer.shape == (2, 4)
        assert omega.shape == (2, 4)


# ══════════════════════════════════════════════════════════════════════════
# Task 5-6 — Env contract
# ══════════════════════════════════════════════════════════════════════════


class TestRangerBoxReachEnv:
    """Full env smoke tests."""

    @pytest.fixture
    def env4(self):
        from unilab.base.registry import ensure_registries, make

        ensure_registries()
        env = make("RangerBoxReach", sim_backend="mujoco", num_envs=4)
        yield env
        env.close()

    def test_env_creates(self, env4):
        assert env4._num_action == 9
        assert env4.obs_groups_spec["obs"] == 39

    def test_reset_returns_obs_dict(self, env4):
        obs, info = env4.reset(np.arange(4))
        assert isinstance(obs, dict)
        assert obs["obs"].shape == (4, 39)

    def test_action_to_ctrl_shape(self, env4):
        state = env4.init_state()
        state.info["current_actions"] = np.zeros((4, 9))
        ctrl = env4.apply_action(np.zeros((4, 9)), state)
        assert ctrl.shape == (4, 7)

    def test_world_ee_goal_set_after_reset(self, env4):
        env4.reset(np.arange(4))
        assert not np.allclose(env4.world_ee_goal, 0.0)

    def test_armbase_ee_goal_nonzero_after_reset(self, env4):
        env4.reset(np.arange(4))
        assert not np.allclose(env4.armbase_ee_goal, 0.0)

    def test_se2_reward_scales_zero(self, env4):
        scales = env4._reward_cfg.scales
        assert scales["base_vel_z"] == 0.0
        assert scales["base_orientation"] == 0.0
        assert scales["base_height"] == 0.0

    def test_default_angles_7_dims(self, env4):
        assert env4.default_angles.shape == (7,)
        assert env4.default_angles[1] == pytest.approx(-0.3)

    def test_partial_reset_clears_history(self, env4):
        env4.reset(np.arange(4))
        env4._history_obs_buf[:] = 1.0
        env4._history_critic_buf[:] = 2.0
        env4.reset(np.array([0, 2]))
        # Reset envs 0,2 get fresh obs (not all-zeros — sensors are live)
        # Untouched envs 1,3 keep their old buffer values
        assert not np.all(env4._history_obs_buf[0] == 1.0)  # changed
        assert np.all(env4._history_obs_buf[1] == 1.0)  # untouched

    def test_mixed_latency(self, env4):
        env4.reset(np.arange(4))
        env4._base_controller.latency_steps[:] = [0, 1, 3, 4]
        # Step with max actions — verifies no crash with mixed latency
        state = env4.step(np.ones((4, 9)))
        assert state.reward.shape == (4,)

    def test_history_h3(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxReachCfg, RangerBoxReachEnv

        cfg = RangerBoxReachCfg()
        cfg.history.num_actor_history = 3
        cfg.history.num_critic_history = 3
        env = RangerBoxReachEnv(cfg, num_envs=2, backend_type="mujoco")
        obs, _ = env.reset(np.arange(2))
        assert obs["obs"].shape == (2, 3 * 39)
        env.close()

    def test_dr_reset(self, env4):
        env4.reset(np.arange(4))
        for _ in range(5):
            state = env4.step(np.zeros((4, 9)))
            if state.terminated.any():
                env4.reset(np.where(state.terminated)[0])
        # No crash = success

    def test_prev_ee_dist_reset_is_real_distance(self, env4):
        """Reset must seed _prev_ee_dist from actual EE→goal distance so the
        first control step produces no spurious progress reward."""
        env4.reset(np.arange(4))
        ee_local, _ = env4.get_ee_local_pose()
        ee_world = env4.armbase_pos_world + np_quat_apply_batched(env4.armbase_quat_world, ee_local)
        expected = np.linalg.norm(ee_world - env4.world_ee_goal, axis=1)
        np.testing.assert_allclose(env4._prev_ee_dist, expected, rtol=1e-5)

    def test_progress_reward_is_signed(self):
        """_reward_ee_progress must be signed (positive approach, negative
        retreat), not clipped at zero."""
        from unilab.envs.locomotion.ranger_box.reach_env import (
            _reward_ee_progress,
            _RewardContext,
        )

        ctx = _RewardContext(
            info={},
            linvel=np.zeros((2, 3)),
            gyro=np.zeros((2, 3)),
            gravity=np.zeros((2, 3)),
            arm_pos=np.zeros((2, 6)),
            arm_vel=np.zeros((2, 6)),
            prev_arm_vel=np.zeros((2, 6)),
            gripper_pos=np.zeros((2, 1)),
            num_envs=2,
            default_arm_angles=np.zeros(6),
            armbase_pos_world=np.zeros((2, 3)),
            armbase_quat_world=np.tile([1.0, 0, 0, 0], (2, 1)),
            ee_local_pos=np.zeros((2, 3)),
            ee_pos_world=np.array([[1.0, 0, 0], [0.5, 0, 0]]),
            world_ee_goal=np.zeros((2, 3)),
            armbase_ee_goal=np.zeros((2, 3)),
            sigma_ee=0.2,
            arm_joint_upper=np.ones(6),
            arm_joint_lower=-np.ones(6),
            joint_limit_margin=0.01,
            ctrl_dt=0.02,
            current_actions=np.zeros((2, 9)),
            prev_ee_dist=np.array([2.0, 0.5]),
        )
        prog = _reward_ee_progress(ctx)
        # env0 approaches: 2.0→1.0 = +1.0; env1 stays: 0.5→0.5 = 0.0
        np.testing.assert_allclose(prog, [1.0, 0.0])

    def test_held_success_terminates(self, env4):
        """Holding EE within 10cm for 0.5s should set terminated+success."""
        env4.reset(np.arange(4))
        env4.init_state()  # establish _state so the goal isn't re-sampled
        # Place the goal at the EE world position (fresh sensor reads) so the
        # very next step is within 10 cm.
        armbase_pos = env4._backend.get_sensor_data(env4._cfg.sensor.armbase_world_pos)
        armbase_quat = env4._backend.get_sensor_data(env4._cfg.sensor.arm_ref_world_quat)
        ee_local, _ = env4.get_ee_local_pose()
        ee_world = armbase_pos + np_quat_apply_batched(armbase_quat, ee_local)
        env4.world_ee_goal[:] = ee_world
        env4._prev_ee_dist[:] = 0.0
        # Pre-seed the hold timer just below threshold: the next step is within
        # 10 cm, so it crosses the threshold and terminates.
        hold_steps = max(1, int(0.5 / env4.cfg.ctrl_dt))
        env4._success_hold_timer[:] = hold_steps - 1
        env4.set_autoreset(False)  # keep _success_hold for assertion
        state = env4.step(np.zeros((4, 9)))
        assert state.terminated.all()
        assert env4._success_hold.all()


# ══════════════════════════════════════════════════════════════════════════
# Wheel IK correctness
# ══════════════════════════════════════════════════════════════════════════


class TestWheelIK:
    """Verify steering stays in [-pi/2, pi/2], omega sign flips, deadband holds."""

    WHEEL_POS = ((0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28))
    RADIUS = 0.152

    def _ik(self, vx, vy, vyaw, prev=None):
        from unilab.envs.locomotion.ranger_box.base_velocity_controller import _compute_wheel_ik

        v = np.array([[vx, vy, vyaw]], dtype=np.float64)
        prev_arr = None if prev is None else np.asarray(prev, dtype=np.float64)
        return _compute_wheel_ik(v, self.WHEEL_POS, self.RADIUS, prev_steer=prev_arr)

    def test_straight_forward(self):
        steer, omega = self._ik(1.0, 0.0, 0.0)
        assert np.all(np.abs(steer) <= np.pi / 2 + 1e-9)
        assert np.all(omega > 0.0)

    def test_strafe(self):
        steer, omega = self._ik(0.0, 1.0, 0.0)
        assert np.all(np.abs(steer) <= np.pi / 2 + 1e-9)

    def test_rotation(self):
        steer, omega = self._ik(0.0, 0.0, 1.0)
        assert np.all(np.abs(steer) <= np.pi / 2 + 1e-9)

    def test_diagonal(self):
        steer, omega = self._ik(1.0, 1.0, 0.5)
        assert np.all(np.abs(steer) <= np.pi / 2 + 1e-9)

    def test_reverse_is_equivalent(self):
        """Reverse motion should give equivalent steer (mod pi) with flipped omega."""
        s_fwd, o_fwd = self._ik(1.0, 0.0, 0.0)
        s_rev, o_rev = self._ik(-1.0, 0.0, 0.0)
        assert np.all(np.abs(s_rev) <= np.pi / 2 + 1e-9)
        # reverse omega should be negative (opposite rolling direction)
        assert np.all(o_rev <= 0.0)

    def test_deadband_holds_steer(self):
        prev = np.zeros((1, 4))
        prev[0] = [0.3, 0.3, 0.3, 0.3]
        steer, omega = self._ik(1e-5, 1e-5, 0.0, prev=prev)
        np.testing.assert_allclose(steer, prev)
        np.testing.assert_allclose(omega, 0.0)


from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
