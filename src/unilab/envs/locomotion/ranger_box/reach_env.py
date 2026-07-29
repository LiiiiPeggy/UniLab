"""RangerBoxReach — mobile manipulator EE reaching environment.

Dataclass definitions, reward functions, DR provider, and env class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.types import ResetPlan
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import (
    np_matrix_from_quat,
    np_quat_apply_batched,
    np_quat_conjugate_batched,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.go2_arm.base import (
    Asset,
    ControlConfig,
    Go2ArmBaseCfg,
    Go2ArmBaseEnv,
    Go2ArmSensor,
    IKConfig,
)
from unilab.envs.locomotion.go2_arm.base import (
    NoiseConfig as _Go2ArmNoiseConfig,
)
from unilab.envs.locomotion.go2_arm.manip_loco import (
    ArmStageConfig,
    EEGoalConfig,
    HistoryConfig,
    InitState,
)
from unilab.envs.locomotion.ranger_box.base_velocity_controller import BaseVelocityController

_RAW_OBS_DIM = 41


# ══════════════════════════════════════════════════════════════════════════
# Asset
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxAsset(Asset):
    base_name: str = "base"
    ground: str = "floor"
    ee_site_name: str = "right_center"
    ee_body_name: str = "cr10_Link6"
    arm_joint_names: tuple[str, ...] = (
        "cr10_joint1", "cr10_joint2", "cr10_joint3",
        "cr10_joint4", "cr10_joint5", "cr10_joint6",
    )
    gripper_joint_name: str = "gripper_finger1_joint"
    steering_joint_names: tuple[str, ...] = (
        "fr_steering_joint", "fl_steering_wheel_joint",
        "rl_steering_wheel_joint", "rr_steering_wheel_joint",
    )
    wheel_joint_names: tuple[str, ...] = (
        "fr_wheel_joint", "fl_wheel_joint",
        "rl_wheel_joint", "rr_wheel_joint",
    )
    wheel_positions: tuple[tuple[float, float], ...] = (
        (0.445, -0.28), (0.445, 0.28),
        (-0.445, 0.28), (-0.445, -0.28),
    )
    wheel_radius: float = 0.152


# ══════════════════════════════════════════════════════════════════════════
# Sensor
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxSensor(Go2ArmSensor):
    local_linvel: str = "imu-velocimeter"
    gyro: str = "imu-gyro"
    framequat: str = "imu-framequat"
    framezaxis: str = "imu-framezaxis"
    upvector: str = "imu-framezaxis"
    ee_local_pos: str = "endpoint-framepos"
    ee_local_quat: str = "endpoint-framequat"
    arm_ref_world_quat: str = "armbasepoint-framequat"
    armbase_world_pos: str = "armbasepoint-framepos"


# ══════════════════════════════════════════════════════════════════════════
# Noise
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxNoiseConfig(_Go2ArmNoiseConfig):
    scale_ee_goal: float = 0.01


# ══════════════════════════════════════════════════════════════════════════
# Control
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxControlConfig(ControlConfig):
    arm_action_scale: float = 0.01
    arm_kp: tuple[float, ...] = (0.1, 0.11, 0.095, 0.05, 0.05, 0.05)
    arm_kd: tuple[float, ...] = (0.5, 0.55, 0.48, 0.25, 0.25, 0.25)
    gripper_kp: float = 500.0
    gripper_kd: float = 10.0
    arm_max_delta_per_step: float = 0.05


# ══════════════════════════════════════════════════════════════════════════
# Base Velocity Controller Config
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class BaseVelocityControllerConfig:
    max_lin_vel: float = 1.5
    max_ang_vel: float = 1.0
    action_scale_lin: float = 1.5
    action_scale_ang: float = 1.0
    tau: float = 0.05
    max_lin_acc: float = 1.5
    max_ang_acc: float = 3.0
    max_latency_steps: int = 4
    action_noise_scale: float = 0.05
    enable_latency: bool = True
    enable_noise: bool = True
    enable_wheel_visualization: bool = True


# ══════════════════════════════════════════════════════════════════════════
# Domain Randomization
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxDomainRandConfig(DomainRandConfig):
    randomize_ground_friction: bool = False
    randomize_kp: bool = True
    kp_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = True
    kd_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_body_mass: bool = True
    body_mass_multiplier_range: tuple[float, float] = (0.9, 1.1)
    random_com: bool = True
    com_offset_x: tuple[float, float] = (-0.03, 0.03)
    randomize_dof_armature: bool = True
    dof_armature_multiplier_range: tuple[float, float] = (0.8, 1.2)
    push_robots: bool = False


# ══════════════════════════════════════════════════════════════════════════
# Reward
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RangerBoxRewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        "ee_distance": 4.0,
        "ee_distance_l2": -1.0,
        "base_vel_xy": -0.05,
        "base_vel_z": 0.0,
        "base_vel_yaw": -0.01,
        "arm_dof_vel": -0.001,
        "arm_dof_acc": -1.0e-6,
        "torques": 0.0,
        "base_orientation": 0.0,
        "base_height": 0.0,
        "arm_joint_limits": -1.0,
        "arm_collision": 0.0,
        "action_rate": -0.01,
        "similar_to_default": -0.005,
        "alive": 0.3,
    })
    sigma_ee: float = 0.15


# ══════════════════════════════════════════════════════════════════════════
# Scene resolution
# ══════════════════════════════════════════════════════════════════════════

def _default_ranger_box_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "ranger_box" / "scene_flat.xml")


def _default_ranger_box_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_ranger_box_model_file())


def _resolve_ranger_box_scene(cfg: "RangerBoxReachCfg") -> SceneCfg:
    scene = cfg.scene
    default_model_file = _default_ranger_box_model_file()
    if scene is None:
        scene = SceneCfg(model_file=cfg.model_file)
    elif cfg.model_file != default_model_file and scene.model_file == default_model_file:
        scene = SceneCfg(
            model_file=cfg.model_file,
            fragment_files=list(scene.fragment_files) if scene.fragment_files else [],
            terrain=scene.terrain,
        )
    cfg.scene = scene
    return scene


def build_ranger_box_position_gains(cc: RangerBoxControlConfig) -> dict[str, np.ndarray]:
    return {
        "kp": np.concatenate([
            np.asarray(cc.arm_kp, dtype=np.float64),
            np.asarray([cc.gripper_kp], dtype=np.float64),
        ]),
        "kd": np.concatenate([
            np.asarray(cc.arm_kd, dtype=np.float64),
            np.asarray([cc.gripper_kd], dtype=np.float64),
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════
# Top-level EnvCfg
# ══════════════════════════════════════════════════════════════════════════

@registry.envcfg("RangerBoxReach")
@dataclass
class RangerBoxReachCfg(Go2ArmBaseCfg):
    scene: SceneCfg | None = field(default_factory=_default_ranger_box_scene)
    model_file: str = field(default_factory=_default_ranger_box_model_file)
    max_episode_seconds: float = 30.0
    init_state: InitState = field(default_factory=InitState)
    control_config: RangerBoxControlConfig = field(default_factory=RangerBoxControlConfig)
    sensor: RangerBoxSensor = field(default_factory=RangerBoxSensor)
    noise_config: RangerBoxNoiseConfig = field(default_factory=RangerBoxNoiseConfig)
    goal_ee: EEGoalConfig = field(default_factory=EEGoalConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    arm_stage: ArmStageConfig = field(default_factory=ArmStageConfig)
    reward_config: RangerBoxRewardConfig = field(default_factory=RangerBoxRewardConfig)
    domain_rand: RangerBoxDomainRandConfig = field(default_factory=RangerBoxDomainRandConfig)
    base_velocity_controller: BaseVelocityControllerConfig = field(
        default_factory=BaseVelocityControllerConfig
    )
    asset: RangerBoxAsset = field(default_factory=RangerBoxAsset)


# ══════════════════════════════════════════════════════════════════════════
# Reward context
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _RewardContext:
    info: dict
    linvel: np.ndarray
    gyro: np.ndarray
    gravity: np.ndarray
    arm_pos: np.ndarray
    arm_vel: np.ndarray
    prev_arm_vel: np.ndarray
    gripper_pos: np.ndarray
    num_envs: int
    default_arm_angles: np.ndarray
    armbase_pos_world: np.ndarray
    armbase_quat_world: np.ndarray
    ee_local_pos: np.ndarray
    ee_pos_world: np.ndarray
    world_ee_goal: np.ndarray
    armbase_ee_goal: np.ndarray
    sigma_ee: float
    arm_joint_upper: np.ndarray
    arm_joint_lower: np.ndarray
    joint_limit_margin: float
    ctrl_dt: float
    current_actions: np.ndarray


# ══════════════════════════════════════════════════════════════════════════
# Reward functions — all return positive values (negative in YAML scale)
# ══════════════════════════════════════════════════════════════════════════

def _reward_ee_distance(ctx: _RewardContext) -> np.ndarray:
    diff = ctx.ee_pos_world - ctx.world_ee_goal
    d2 = np.sum(diff * diff, axis=1)
    sigma2 = ctx.sigma_ee * ctx.sigma_ee
    return np.exp(-d2 / sigma2)


def _reward_ee_distance_l2(ctx: _RewardContext) -> np.ndarray:
    diff = ctx.ee_pos_world - ctx.world_ee_goal
    return np.sum(diff * diff, axis=1)


def _reward_base_vel_xy(ctx: _RewardContext) -> np.ndarray:
    return ctx.linvel[:, 0] ** 2 + ctx.linvel[:, 1] ** 2


def _reward_base_vel_yaw(ctx: _RewardContext) -> np.ndarray:
    return ctx.gyro[:, 2] ** 2


def _reward_arm_dof_vel(ctx: _RewardContext) -> np.ndarray:
    return np.sum(ctx.arm_vel * ctx.arm_vel, axis=1)


def _reward_arm_dof_acc(ctx: _RewardContext) -> np.ndarray:
    acc = (ctx.arm_vel - ctx.prev_arm_vel) / ctx.ctrl_dt
    return np.sum(acc * acc, axis=1)


def _reward_arm_joint_limits(ctx: _RewardContext) -> np.ndarray:
    margin = ctx.joint_limit_margin
    upper_violation = np.maximum(0.0, ctx.arm_pos - (ctx.arm_joint_upper - margin))
    lower_violation = np.maximum(0.0, (ctx.arm_joint_lower + margin) - ctx.arm_pos)
    return np.sum(upper_violation + lower_violation, axis=1)


# ══════════════════════════════════════════════════════════════════════════
# DR Provider
# ══════════════════════════════════════════════════════════════════════════

class RangerBoxReachDRProvider(LocomotionDRProvider):
    """DR provider for RangerBoxReach — caches kp/kd/mass/armature at init."""

    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_body_mass: np.ndarray | None = None,
        base_dof_armature: np.ndarray | None = None,
    ):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_dof_armature = base_dof_armature

    def _sample_commands(self, env, num_reset: int) -> np.ndarray:
        return np.zeros((num_reset, 3), dtype=get_global_dtype())

    def _get_base_actuator_gains(self, env) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(self, env):
        return self._base_body_mass, None, None, self._base_dof_armature

    def build_reset_plan(self, env, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        env._arm_goal_timer[env_ids] = 0
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._prev_arm_vel[env_ids] = 0.0
        env._base_controller.reset(env_ids, np.random.default_rng())
        return plan

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        # Read armbase pose before reset_ee_goals (needed for world-frame goal sampling)
        env.armbase_pos_world[env_ids] = env._backend.get_sensor_data(
            env._cfg.sensor.armbase_world_pos
        )[env_ids]
        env.armbase_quat_world[env_ids] = env._backend.get_sensor_data(
            env._cfg.sensor.arm_ref_world_quat
        )[env_ids]
        env.reset_ee_goals(env_ids)
        # Compute armbase_ee_goal for initial obs
        env.armbase_ee_goal[env_ids] = env._world_goal_to_armbase(
            env.world_ee_goal[env_ids],
            env.armbase_pos_world[env_ids],
            env.armbase_quat_world[env_ids],
        )
        sliced_info: dict = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        ee_local_pos, _ = env.get_ee_local_pose()
        raw = env._compute_raw_obs(
            sliced_info, linvel, gyro, gravity, dof_pos, dof_vel,
            ee_local_pos[env_ids], env.armbase_ee_goal[env_ids],
            add_noise=True,
        )
        return env._update_history(raw, env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════
# RangerBoxReachEnv
# ══════════════════════════════════════════════════════════════════════════

@registry.env("RangerBoxReach", sim_backend="mujoco")
class RangerBoxReachEnv(Go2ArmBaseEnv):
    """Mobile manipulator EE reaching with A+ freejoint base controller."""

    _cfg: RangerBoxReachCfg

    def __init__(self, cfg: RangerBoxReachCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        scene = _resolve_ranger_box_scene(cfg)
        position_actuator_gains = build_ranger_box_position_gains(cfg.control_config)

        backend_kwargs: dict[str, Any] = {
            "base_name": cfg.asset.base_name,
            "push_body_name": cfg.domain_rand.push_body_name,
            "position_actuator_gains": position_actuator_gains,
            "iterations": getattr(cfg, "iterations", None),
            "post_step_forward_sensor": getattr(cfg, "post_step_forward_sensor", False),
        }
        backend = create_backend(backend_type, scene, num_envs, cfg.sim_dt, **backend_kwargs)
        super().__init__(cfg, backend, num_envs)

        self._num_action = 10

        ctrl_range = self._backend.get_actuator_ctrl_range()
        self._ctrl_low = np.asarray(ctrl_range[:, 0], dtype=np.float64)
        self._ctrl_high = np.asarray(ctrl_range[:, 1], dtype=np.float64)

        self._gripper_dof_pos_idx = self._backend.get_joint_dof_pos_indices(
            [cfg.asset.gripper_joint_name]
        )
        self._gripper_dof_vel_idx = self._backend.get_joint_dof_vel_indices(
            [cfg.asset.gripper_joint_name]
        )

        self._base_controller = BaseVelocityController(
            cfg.base_velocity_controller, cfg.ctrl_dt, backend, cfg.asset, num_envs
        )

        self.world_ee_goal = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_ee_goal = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_pos_world = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_quat_world = np.zeros((num_envs, 4), dtype=np.float64)

        self._prev_arm_vel = np.zeros((num_envs, 6), dtype=np.float64)
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)

        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._history_obs_buf = np.zeros((num_envs, H_a * _RAW_OBS_DIM), dtype=get_global_dtype())
        self._history_critic_buf = np.zeros((num_envs, H_c * _RAW_OBS_DIM), dtype=get_global_dtype())

        self._default_arm_angles = self.default_angles[:6].copy()

        self._arm_joint_upper = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14], dtype=np.float64)
        self._arm_joint_lower = np.array(
            [-3.92, -1.57, -2.86, -3.14, -3.14, -3.14], dtype=np.float64
        )

        # DR provider
        base_kp, base_kd = (None, None)
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
        base_body_mass = None
        if cfg.domain_rand.randomize_body_mass:
            base_body_mass = backend.get_body_mass()
        base_dof_armature = None
        if cfg.domain_rand.randomize_dof_armature:
            base_dof_armature = backend.get_dof_armature()
        dr_provider = RangerBoxReachDRProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=base_body_mass,
            base_dof_armature=base_dof_armature,
        )
        self._init_domain_randomization(dr_provider)

        self._init_ee_goals()
        self._init_reward_functions()

    # ── Action space ────────────────────────────────────────────────

    def _init_action_space(self) -> None:
        import gymnasium as gym
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

    def _init_buffers(self) -> None:
        dtype = get_global_dtype()
        raw_qpos = self._backend.get_keyframe_qpos("home")
        self._init_qpos = np.asarray(raw_qpos, dtype=dtype)
        # default_angles = arm control defaults from keyframe ctrl
        # (6 arm joints + 1 gripper), NOT from qpos tail.
        self.default_angles = np.array(
            [0.0, -0.3, 0.75, 0.0, 0.45, 0.0, 0.0], dtype=dtype
        )
        raw_qvel = self._backend.get_init_qvel()
        self._init_qvel = np.asarray(raw_qvel, dtype=dtype)

    # ── Action application ──────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions.copy()

        self._base_controller.step(actions[:, 0:3])

        arm_gripper_action = (
            state.info["last_actions"][:, 3:10]
            if self._cfg.control_config.simulate_action_latency
            else actions[:, 3:10]
        )

        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world
        )

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        dq_ik = self.compute_arm_ik_delta(
            self.armbase_ee_goal, ee_local_pos,
            self.ee_goal_orn_quat, ee_local_quat,
        )

        arm_pos = self.get_arm_dof_pos()
        arm_delta = (
            arm_gripper_action[:, 0:6] * self._cfg.control_config.arm_action_scale
            + self._cfg.ik.gain * dq_ik
        )
        # Clamp per-step joint delta to prevent QACC explosion
        max_delta = getattr(self._cfg.control_config, "arm_max_delta_per_step", 0.1)
        if max_delta > 0:
            arm_delta = np.clip(arm_delta, -max_delta, max_delta)
        arm_ctrl = arm_pos + arm_delta

        # Store arm target for kinematic write in update_state (after physics)
        self._pending_arm_target = np.clip(
            arm_ctrl, self._ctrl_low[:6], self._ctrl_high[:6]
        ).astype(np.float64)

        # Send CURRENT arm positions as ctrl so position actuators
        # produce minimal force during physics step.
        grip_ctrl = np.zeros((actions.shape[0], 1), dtype=np.float64)
        ctrl = np.concatenate([arm_pos, grip_ctrl], axis=1)

        # Apply base velocity BEFORE physics step so MuJoCo integrates from it
        self._base_controller.apply_velocity()

        return ctrl.astype(get_global_dtype())

    # ── Observation ─────────────────────────────────────────────────

    def _compute_raw_obs(
        self, info, linvel, gyro, gravity, dof_pos, dof_vel,
        ee_local_pos, armbase_ee_goal, *, add_noise=True,
    ) -> np.ndarray:
        n = len(dof_pos)
        # Slice arm DOF from full dof_pos/dof_vel
        arm_pos = dof_pos[:, self._arm_dof_pos_indices] if dof_pos.shape[1] > 6 else dof_pos
        arm_vel = dof_vel[:, self._arm_dof_vel_indices] if dof_vel.shape[1] > 6 else dof_vel
        arm_diff = arm_pos - self._default_arm_angles
        if add_noise:
            noise_cfg = self._cfg.noise_config
            linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
            gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
            gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
            arm_diff = self._obs_noise(arm_diff, noise_cfg.scale_joint_angle)
            arm_vel = self._obs_noise(arm_vel, noise_cfg.scale_joint_vel)
            ee_local_pos = self._obs_noise(ee_local_pos, noise_cfg.scale_ee_pos)
            armbase_ee_goal = self._obs_noise(armbase_ee_goal, noise_cfg.scale_ee_goal)

        last_actions = info.get("current_actions", np.zeros((n, 10), dtype=get_global_dtype()))
        ee_error = armbase_ee_goal - ee_local_pos
        # gripper_pos: ensure shape matches batch size
        raw_gripper = self._backend.get_dof_pos()[:, self._gripper_dof_pos_idx]
        if raw_gripper.shape[0] != n:
            raw_gripper = raw_gripper[:n]
        gripper_pos = raw_gripper

        return np.concatenate([
            linvel.astype(get_global_dtype()),              # 3
            gyro.astype(get_global_dtype()),                # 3
            (-gravity).astype(get_global_dtype()),          # 3
            arm_diff.astype(get_global_dtype()),            # 6
            arm_vel.astype(get_global_dtype()),             # 6
            ee_local_pos.astype(get_global_dtype()),        # 3
            armbase_ee_goal.astype(get_global_dtype()),     # 3
            ee_error.astype(get_global_dtype()),            # 3
            gripper_pos.astype(get_global_dtype()),         # 1
            last_actions.astype(get_global_dtype()),        # 10
        ], axis=1)

    def _update_history(self, raw_obs, env_ids=None, *, critic_raw_obs=None):
        D = _RAW_OBS_DIM
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        critic_step = raw_obs if critic_raw_obs is None else critic_raw_obs
        if env_ids is None:
            if H_a > 1:
                self._history_obs_buf = np.roll(self._history_obs_buf, -D, axis=1)
            self._history_obs_buf[:, -D:] = raw_obs
            if H_c > 1:
                self._history_critic_buf = np.roll(self._history_critic_buf, -D, axis=1)
            self._history_critic_buf[:, -D:] = critic_step
            return {"obs": self._history_obs_buf.copy(),
                    "critic": self._history_critic_buf.copy()}
        else:
            if H_a > 1:
                self._history_obs_buf[env_ids] = np.roll(
                    self._history_obs_buf[env_ids], -D, axis=1)
            self._history_obs_buf[env_ids, -D:] = raw_obs
            if H_c > 1:
                self._history_critic_buf[env_ids] = np.roll(
                    self._history_critic_buf[env_ids], -D, axis=1)
            self._history_critic_buf[env_ids, -D:] = critic_step
            return {"obs": self._history_obs_buf[env_ids].copy(),
                    "critic": self._history_critic_buf[env_ids].copy()}

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {"obs": H_a * _RAW_OBS_DIM, "critic": H_c * _RAW_OBS_DIM}

    # ── State update ────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # Base velocity is already applied in apply_action (before physics step).
        # Use controller's v_real directly as base linvel.
        linvel = self._base_controller.v_real.astype(get_global_dtype())

        # Apply arm kinematic positions AFTER physics step to prevent
        # position-actuator / freejoint instability.  The arm delta was
        # already computed in apply_action; we just persist the target here.
        self._backend.set_joint_qpos(
            list(self._cfg.asset.arm_joint_names), self._pending_arm_target
        )
        self._backend.set_joint_qvel(
            list(self._cfg.asset.arm_joint_names),
            np.zeros((self._num_envs, 6), dtype=np.float64),
        )
        gyro = self.get_gyro()
        gravity = self._get_projected_gravity()

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        arm_pos = self.get_arm_dof_pos()
        arm_vel = self.get_arm_dof_vel()

        self.armbase_pos_world = self._backend.get_sensor_data(
            self._cfg.sensor.armbase_world_pos)
        self.armbase_quat_world = self._backend.get_sensor_data(
            self._cfg.sensor.arm_ref_world_quat)

        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world)

        ee_pos_world = self.armbase_pos_world + np_quat_apply_batched(
            self.armbase_quat_world, ee_local_pos)

        tilt_sq = gravity[:, 0]**2 + gravity[:, 1]**2
        limit_violated = (
            (arm_pos > self._arm_joint_upper) | (arm_pos < self._arm_joint_lower)
        ).any(axis=1)
        terminated = (tilt_sq > np.sin(1.0)**2) | limit_violated

        prev_arm_vel_saved = self._prev_arm_vel.copy()
        self._prev_arm_vel = arm_vel.copy()

        ctx = _RewardContext(
            info=state.info,
            linvel=linvel, gyro=gyro, gravity=gravity,
            arm_pos=arm_pos, arm_vel=arm_vel,
            prev_arm_vel=prev_arm_vel_saved,
            gripper_pos=self.get_gripper_dof_pos(),
            num_envs=self._num_envs,
            default_arm_angles=self._default_arm_angles,
            armbase_pos_world=self.armbase_pos_world,
            armbase_quat_world=self.armbase_quat_world,
            ee_local_pos=ee_local_pos,
            ee_pos_world=ee_pos_world,
            world_ee_goal=self.world_ee_goal,
            armbase_ee_goal=self.armbase_ee_goal,
            sigma_ee=self._reward_cfg.sigma_ee,
            arm_joint_upper=self._arm_joint_upper,
            arm_joint_lower=self._arm_joint_lower,
            joint_limit_margin=0.01,
            ctrl_dt=self._cfg.ctrl_dt,
            current_actions=state.info.get(
                "current_actions", np.zeros((self._num_envs, 10))),
        )

        reward = self._compute_reward(ctx)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, arm_pos, arm_vel,
                                ee_local_pos, self.armbase_ee_goal, add_noise=True)
        # NaN guard: physics instability can produce NaN obs; zero them out
        for k in obs:
            obs[k] = np.nan_to_num(obs[k], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        # Also terminate envs with NaN in arm pos (physics explosion)
        nan_terminated = np.any(np.isnan(arm_pos), axis=1)
        return state.replace(
            obs=obs, reward=reward,
            terminated=np.logical_or(terminated, nan_terminated),
        )

    # ── Reward ──────────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_cfg = self._cfg.reward_config
        # Wrap common rewards to map context field names
        def _action_rate_wrapper(ctx):
            current = ctx.info.get("current_actions",
                                   np.zeros((ctx.num_envs, 10)))
            prev = ctx.info.get("last_actions",
                                np.zeros((ctx.num_envs, 10)))
            return np.sum((current - prev) ** 2, axis=1)

        def _similar_to_default_wrapper(ctx):
            return np.sum(np.abs(ctx.arm_pos - ctx.default_arm_angles), axis=1)

        def _alive_wrapper(ctx):
            return np.ones(ctx.num_envs)

        self._reward_fns: dict[str, Any] = {
            "ee_distance": _reward_ee_distance,
            "ee_distance_l2": _reward_ee_distance_l2,
            "base_vel_xy": _reward_base_vel_xy,
            "base_vel_yaw": _reward_base_vel_yaw,
            "arm_dof_vel": _reward_arm_dof_vel,
            "arm_dof_acc": _reward_arm_dof_acc,
            "arm_joint_limits": _reward_arm_joint_limits,
            "action_rate": _action_rate_wrapper,
            "similar_to_default": _similar_to_default_wrapper,
            "alive": _alive_wrapper,
        }

    def _compute_reward(self, ctx: _RewardContext) -> np.ndarray:
        scales = self._reward_cfg.scales
        reward = np.zeros(ctx.num_envs, dtype=np.float64)
        for name, fn in self._reward_fns.items():
            scale = scales.get(name, 0.0)
            if abs(scale) < 1e-15:
                continue
            reward = reward + scale * fn(ctx)
        return np.asarray(reward * self._cfg.ctrl_dt, dtype=get_global_dtype())

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_projected_gravity(self) -> np.ndarray:
        quat = self._backend.get_sensor_data(self._cfg.sensor.framequat)
        R_wb = np_matrix_from_quat(quat)
        return np.einsum("nij,j->ni", np.swapaxes(R_wb, 1, 2),
                          np.array([0.0, 0.0, -1.0], dtype=R_wb.dtype))

    def get_arm_dof_pos(self) -> np.ndarray:
        idx = self._backend.get_joint_dof_pos_indices(self._cfg.asset.arm_joint_names)
        return self._backend.get_dof_pos()[:, idx]

    def get_arm_dof_vel(self) -> np.ndarray:
        idx = self._backend.get_joint_dof_vel_indices(self._cfg.asset.arm_joint_names)
        return self._backend.get_dof_vel()[:, idx]

    def get_gripper_dof_pos(self) -> np.ndarray:
        return self._backend.get_dof_pos()[:, self._gripper_dof_pos_idx]

    def get_ee_local_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self._backend.get_sensor_data(self._cfg.sensor.ee_local_pos)
        quat = self._backend.get_sensor_data(self._cfg.sensor.ee_local_quat)
        return pos, quat

    # compute_arm_ik_delta inherited from Go2ArmBaseEnv — uses Jacobian

    # ── Goal management ─────────────────────────────────────────────

    def _init_ee_goals(self) -> None:
        self.world_ee_goal = np.zeros((self._num_envs, 3), dtype=np.float64)
        self.ee_goal_orn_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (self._num_envs, 1)
        )
        self._arm_goal_timer = np.zeros((self._num_envs,), dtype=np.int32)

    def reset_ee_goals(self, env_ids: np.ndarray) -> None:
        n = len(env_ids)
        rng = np.random
        reachable = rng.random(n) < 0.30
        goals = np.zeros((n, 3), dtype=np.float64)

        n_reach = int(reachable.sum())
        if n_reach > 0:
            r = rng.uniform(0.2, 0.5, size=n_reach)
            phi = rng.uniform(-1.2, 1.0, size=n_reach)
            theta = rng.uniform(-2.0, 2.0, size=n_reach)
            goals[reachable, 0] = r * np.cos(phi) * np.cos(theta)
            goals[reachable, 1] = r * np.cos(phi) * np.sin(theta)
            goals[reachable, 2] = r * np.sin(phi)

        n_ext = n - n_reach
        if n_ext > 0:
            r_e = rng.uniform(0.5, 1.2, size=n_ext)
            phi_e = rng.uniform(-1.2, 1.0, size=n_ext)
            theta_e = rng.uniform(-2.0, 2.0, size=n_ext)
            goals[~reachable, 0] = r_e * np.cos(phi_e) * np.cos(theta_e)
            goals[~reachable, 1] = r_e * np.cos(phi_e) * np.sin(theta_e)
            goals[~reachable, 2] = r_e * np.sin(phi_e)

        armbase_pos = self.armbase_pos_world[env_ids]
        armbase_quat = self.armbase_quat_world[env_ids]
        goals_world = armbase_pos + np_quat_apply_batched(armbase_quat, goals)
        self.world_ee_goal[env_ids] = goals_world
        self._arm_goal_timer[env_ids] = 0

    def _world_goal_to_armbase(self, world_goal, armbase_pos, armbase_quat):
        rel = world_goal - armbase_pos
        q_conj = np_quat_conjugate_batched(armbase_quat)
        return np_quat_apply_batched(q_conj, rel)

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel,
                     ee_local_pos, armbase_ee_goal, *, add_noise=True):
        raw = self._compute_raw_obs(
            info, linvel, gyro, gravity, dof_pos, dof_vel,
            ee_local_pos, armbase_ee_goal, add_noise=add_noise,
        )
        return self._update_history(raw)
