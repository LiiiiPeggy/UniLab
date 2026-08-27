"""Tests for RangerBoxReach env — backend contract, dataclasses, env smoke."""

from __future__ import annotations

from pathlib import Path

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

    def test_traj_recording_is_lazy(self, env4):
        env4.reset(np.arange(4))
        # Recording must be OFF during training; only the eval marker getter
        # (play-only) enables it.
        assert env4._record_traj is False
        env4.eval_visualization_markers()
        assert env4._record_traj is True

    def test_eval_visualization_markers_shape(self, env4):
        env4.reset(np.arange(4))
        for _ in range(3):
            env4.step(np.zeros((4, 9)))
        m = env4.eval_visualization_markers()
        assert m.shape == (4, 6 + 6 * env4._traj_len)
        # Goal + EE columns are finite; unfilled trail slots are NaN.
        assert np.isfinite(m[:, :6]).all()
        assert np.isnan(m[:, 6:]).any()

    def test_eval_visualization_text(self, env4):
        env4.reset(np.arange(4))
        for _ in range(3):
            env4.step(np.zeros((4, 9)))
        t = env4.eval_visualization_text()
        assert isinstance(t, list) and len(t) == 4
        assert all(isinstance(s, str) and "env" in s and "d=" in s for s in t)

    def test_traj_last_slot_tracks_ee(self, env4):
        from unilab.envs.common.rotation import np_quat_apply_batched

        env4.reset(np.arange(4))
        env4.eval_visualization_markers()  # enable recording
        for _ in range(5):
            env4.step(np.zeros((4, 9)))
        ee_local, _ = env4.get_ee_local_pose()
        ee_world = env4.armbase_pos_world + np_quat_apply_batched(env4.armbase_quat_world, ee_local)
        assert np.allclose(env4._traj_ee[:, -1], ee_world, atol=1e-6)
        base_world = env4._backend.get_base_pos()
        assert np.allclose(env4._traj_base[:, -1], base_world, atol=1e-6)

    def test_goal_config_local_extended_fields(self, env4):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxEEGoalConfig

        cfg = RangerBoxEEGoalConfig()
        assert cfg.local_fraction == 0.30
        assert cfg.local_radius_range == (0.10, 0.15)
        assert cfg.extended_xy_radius_range == (0.30, 0.70)
        assert cfg.extended_z_range == (-0.10, 0.10)
        # |dz| bound keeps EXTENDED goals vertically inside base+arm reach.
        assert abs(cfg.extended_z_range[1] - cfg.extended_z_range[0]) / 2.0 < cfg.capture_outer
        assert cfg.capture_inner < cfg.capture_outer
        # Old spherical reachable fields are gone from the ranger config.
        assert not hasattr(cfg, "reachable_fraction")
        assert not hasattr(cfg, "extended_radius_range")

    def test_reset_goals_local_extended_radial(self, env4):
        from unilab.envs.common.rotation import np_quat_apply_batched

        # With only 4 envs, a 30 % local-fraction reset can (by chance) draw all
        # extended (~24 %); resample until mixed so the invariants below
        # are exercised for BOTH goal types.
        for _ in range(40):
            env4.reset(np.arange(4))
            loc = env4._goal_is_local
            if 0.0 < loc.mean() < 1.0:
                break
        ee_local, _ = env4.get_ee_local_pose()
        ee_world = env4.armbase_pos_world + np_quat_apply_batched(env4.armbase_quat_world, ee_local)
        d = np.linalg.norm(ee_world - env4.world_ee_goal, axis=1)
        assert 0.0 < loc.mean() < 1.0  # mixed local/extended
        cfg = env4._cfg.goal_ee
        # LOCAL: true 3D radial — EE-to-goal distance == sampled radius.
        assert np.all(d[loc] >= cfg.local_radius_range[0] - 1e-3)
        assert np.all(d[loc] <= cfg.local_radius_range[1] + 1e-3)
        gl = env4._world_goal_to_armbase(
            env4.world_ee_goal, env4.armbase_pos_world, env4.armbase_quat_world
        )
        assert np.allclose(np.linalg.norm(gl - ee_local, axis=1)[loc], d[loc], atol=1e-3)
        # EXTENDED: planar — r_xy in range and |dz| bounded (base-only closure).
        r_xy = np.linalg.norm(gl[~loc, :2] - ee_local[~loc, :2], axis=1)
        dz = gl[~loc, 2] - ee_local[~loc, 2]
        assert np.all(r_xy >= cfg.extended_xy_radius_range[0] - 1e-3)
        assert np.all(r_xy <= cfg.extended_xy_radius_range[1] + 1e-3)
        assert np.all(dz >= cfg.extended_z_range[0] - 1e-3)
        assert np.all(dz <= cfg.extended_z_range[1] + 1e-3)

    def test_capture_gate_arm_weight_by_distance(self, env4):
        # The arm-engagement gate ramps arm_weight on the EE-to-goal distance:
        # LOCAL goals (within capture_inner) → fully engaged; EXTENDED goals
        # (beyond capture_outer) → disengaged (held at the ready pose).
        for _ in range(40):  # resample until BOTH goal types are present (4 envs)
            env4.reset(np.arange(4))
            if env4._goal_is_local.any() and (~env4._goal_is_local).any():
                break
        cfg = env4._cfg.goal_ee
        ee_local, _ = env4.get_ee_local_pose()
        gl = env4._world_goal_to_armbase(
            env4.world_ee_goal, env4.armbase_pos_world, env4.armbase_quat_world
        )
        ee_error = np.linalg.norm(gl - ee_local, axis=1)
        aw = np.clip(
            (cfg.capture_outer - ee_error) / max(cfg.capture_outer - cfg.capture_inner, 1e-6),
            0.0,
            1.0,
        )
        loc = env4._goal_is_local
        assert loc.any() and (~loc).any()  # mixed local/extended at reset
        assert np.all(aw[loc] > 0.99)  # local goals fully engaged
        assert np.all(aw[~loc] < 0.01)  # extended goals disengaged

    def test_base_arm_capture_complementary_weights(self, env4):
        """base_weight = 1 - arm_weight: smooth handoff, no hard switch at capture."""
        env4.reset(np.arange(4))
        cfg = env4._cfg.goal_ee
        ci = cfg.capture_inner
        co = cfg.capture_outer
        ee_local, _ = env4.get_ee_local_pose()
        gl = env4._world_goal_to_armbase(
            env4.world_ee_goal, env4.armbase_pos_world, env4.armbase_quat_world
        )
        ee_error = np.linalg.norm(gl - ee_local, axis=1)
        aw = np.clip((co - ee_error) / max(co - ci, 1e-6), 0.0, 1.0)
        bw = 1.0 - aw
        far = ee_error >= co
        near = ee_error <= ci
        if far.any():
            assert np.all(aw[far] == 0.0) and np.all(bw[far] == 1.0)
        if near.any():
            assert np.all(aw[near] == 1.0) and np.all(bw[near] == 0.0)
        # complement everywhere (smooth blend in the transition band)
        assert np.allclose(aw + bw, 1.0)

    def test_capture_handoff_suppresses_base_inside_capture(self, env4):
        """LOCAL goal (inside capture_inner): random base action is suppressed."""
        for _ in range(40):
            # init_state() first: env.reset() rebuilds physics but leaves
            # self._state None, so the FIRST step would otherwise call
            # init_state() → extra reset → goal resampled → LOCAL lost.
            env4.init_state()
            env4.reset(np.arange(4))
            loc = env4._goal_is_local
            if loc.any():
                break
        assert loc.any()
        env4.set_autoreset(False)
        rng = np.random.default_rng(0)
        act = np.zeros((4, 9))
        act[:, 0:3] = rng.uniform(-1.0, 1.0, size=(4, 3))  # random base, zero arm
        base_start = env4._backend.get_base_pos()[:, :2].copy()
        for _ in range(30):
            env4.step(act)
        base_disp = np.linalg.norm(env4._backend.get_base_pos()[:, :2] - base_start, axis=1)
        # LOCAL goals sit inside capture_inner → arm_weight≈1 → base_weight≈0
        assert np.max(base_disp[loc]) < 0.05, f"local base moved {base_disp[loc].max():.3f}m"

    def test_extended_goal_blend_weights(self, env4):
        """EXT goal handoff (Task 19-B): far → bw≈1 base free; near → aw≈1."""
        cfg = env4._cfg.goal_ee
        ci, co = cfg.capture_inner, cfg.capture_outer
        for ee_error in (0.30, 0.22, co, (ci + co) / 2.0, ci, 0.12):
            aw = float(np.clip((co - ee_error) / max(co - ci, 1e-6), 0.0, 1.0))
            bw = 1.0 - aw
            assert np.allclose(aw + bw, 1.0)
            if ee_error >= co:
                assert aw == 0.0 and bw == 1.0
            elif ee_error <= ci:
                assert aw == 1.0 and bw == 0.0
            else:
                assert 0.0 < aw < 1.0 and 0.0 < bw < 1.0

    def test_held_success_terminates_with_terminal_bonus(self):
        """Task 19-C: hold 0.5 s fires the terminal event AND terminates that env.

        Uses a Hydra-composed env (mujoco.yaml IK settings) because the registry
        ``make`` fixture builds with dataclass IK defaults (gain 1.0) that are
        measurably slower than the training config (gain 1.5) — the pure-IK
        convergence this test depends on needs the real task config.
        """
        from hydra import compose, initialize_config_dir

        from unilab.training import BackendAdapter, create_env, ensure_registries

        ensure_registries()
        with initialize_config_dir(
            version_base="1.3", config_dir=str(Path(__file__).parent.parent / "conf" / "ppo")
        ):
            cfg = compose(
                config_name="config",
                overrides=["task=ranger_box_reach/mujoco", "algo.num_envs=4"],
            )
        root = Path(__file__).parent.parent
        env_cfg_override = BackendAdapter(cfg, root_dir=root).build_task_env_cfg_override()
        env = create_env(cfg, num_envs=4, env_cfg_override=env_cfg_override)

        try:
            for _ in range(60):
                env.init_state()
                env.reset(np.arange(4))
                if env._goal_is_local.any():
                    break
            assert env._goal_is_local.any()  # LOCAL goal → pure IK reaches & holds
            env.set_autoreset(False)
            for _ in range(120):
                st = env.step(np.zeros((4, 9)))
                ev = st.info.get("event_success_hold", np.zeros(4, dtype=bool))
                if np.any(ev):
                    idx = np.flatnonzero(ev)
                    assert np.all(st.terminated[idx]), "hold event envs must terminate"
                    return
            assert False, "a LOCAL env never reached held success (pure IK, zero action)"
        finally:
            env.close()

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
        # default_angles = manipulation-ready folded pose (find_ranger_box_ready_pose),
        # NOT the old nominal 0,-0.3,0.75,...  The reset keyframe arm qpos must
        # match it so reset starts at the folded ready pose.
        assert env4.default_angles[1] == pytest.approx(0.0031, abs=1e-3)
        assert np.allclose(env4._init_qpos[15:21], env4.default_angles[:6], atol=1e-4)
        assert np.allclose(env4._init_qpos[15:21], env4._default_arm_angles, atol=1e-4)

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


# ══════════════════════════════════════════════════════════════════════════
# IK Jacobian + anti-windup
# ══════════════════════════════════════════════════════════════════════════


class TestIKAntiWindup:
    """Verify Jacobian frame is correct and _ik_target does not wind up."""

    @pytest.fixture
    def env1(self):
        from unilab.base.registry import ensure_registries, make

        ensure_registries()
        env = make("RangerBoxReach", sim_backend="mujoco", num_envs=1)
        yield env
        env.close()

    def test_ik_jacobian_matches_finite_difference(self, env1):
        from unilab.envs.common.rotation import np_matrix_from_quat

        env1.reset(np.array([0]))
        env1.init_state()
        q0 = env1.get_arm_dof_pos()[0].copy()
        names = list(env1._cfg.asset.arm_joint_names)

        # Analytic armbase-frame position Jacobian (same as compute_arm_ik_delta).
        jacp_w, _ = env1._backend.get_site_jacobian_w(
            env1._ee_site_id, env1._arm_jacobian_dof_indices
        )
        ref_rot_w = np_matrix_from_quat(
            env1._backend.get_sensor_data(env1._cfg.sensor.arm_ref_world_quat)
        )
        rot_w_to_b = np.swapaxes(ref_rot_w, 1, 2)
        jacp_b = np.matmul(rot_w_to_b, jacp_w)[0]  # (3, 6) for env 0

        # Numerical finite-difference Jacobian in armbase frame (endpoint-framepos
        # is already expressed relative to armbasepoint).
        eps = 1e-5
        jac_num = np.zeros((3, 6))
        for j in range(6):
            q_plus = q0.copy()
            q_minus = q0.copy()
            q_plus[j] += eps
            q_minus[j] -= eps
            env1._backend.set_joint_qpos(names, q_plus[None])
            env1._backend.forward_sensors()
            p_plus = env1.get_ee_local_pose()[0][0].copy()
            env1._backend.set_joint_qpos(names, q_minus[None])
            env1._backend.forward_sensors()
            p_minus = env1.get_ee_local_pose()[0][0].copy()
            jac_num[:, j] = (p_plus - p_minus) / (2 * eps)

        err = np.max(np.abs(jac_num - jacp_b))
        assert err < 1e-2, f"Jacobian frame mismatch: max_abs_error={err}"

    def test_ik_target_does_not_windup_on_unreachable_goal(self, env1):
        # Place a clearly unreachable goal (2 m away in armbase frame) and run
        # many IK-only steps: _ik_target must stay near q_actual, NOT pin at
        # the soft joint limits.
        env1.reset(np.array([0]))
        env1.init_state()
        far_world = env1.armbase_pos_world[0] + np.array([2.0, 0.0, 0.0])
        env1.world_ee_goal[:] = far_world
        for _ in range(500):
            env1.step(np.zeros((1, 9)))
        q_actual = env1.get_arm_dof_pos()[0]
        err = np.abs(env1._ik_target[0] - q_actual)
        # Allow one step of arm motion past the clip (the clip runs in
        # apply_action, then physics moves the arm before this read); a true
        # windup would pin _ik_target ~2.5 rad from q_actual at the soft limits.
        assert err.max() < env1._cfg.control_config.max_target_error + 0.05

    def test_ik_target_never_saturates_soft_limits(self, env1):
        # The target must never pin at the soft joint limits (the windup
        # symptom).  With a random goal (some reachable, some not), the
        # anti-windup should keep _ik_target inside the soft limits and near
        # the actual pose, never railing to ±2.8 rad.
        env1.reset(np.array([0]))
        env1.init_state()
        for _ in range(500):
            env1.step(np.zeros((1, 9)))
            t = env1._ik_target[0]
            assert np.all(t >= env1._arm_soft_lower - 1e-6)
            assert np.all(t <= env1._arm_soft_upper + 1e-6)
            # The target must stay reasonably close to actual (anti-windup),
            # not drift several rad away.
            q_actual = env1.get_arm_dof_pos()[0]
            assert np.abs(t - q_actual).max() < 0.5


from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
