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

_RAW_OBS_DIM = 39


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
        "cr10_joint1",
        "cr10_joint2",
        "cr10_joint3",
        "cr10_joint4",
        "cr10_joint5",
        "cr10_joint6",
    )
    gripper_joint_name: str = "gripper_finger1_joint"
    steering_joint_names: tuple[str, ...] = (
        "fr_steering_joint",
        "fl_steering_wheel_joint",
        "rl_steering_wheel_joint",
        "rr_steering_wheel_joint",
    )
    wheel_joint_names: tuple[str, ...] = (
        "fr_wheel_joint",
        "fl_wheel_joint",
        "rl_wheel_joint",
        "rr_wheel_joint",
    )
    wheel_positions: tuple[tuple[float, float], ...] = (
        (0.445, -0.28),
        (0.445, 0.28),
        (-0.445, 0.28),
        (-0.445, -0.28),
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
    # World-frame body positions of arm links (for arm-base clearance).
    link_pos: tuple[str, ...] = (
        "link2-framepos",
        "link3-framepos",
        "link4-framepos",
        "link5-framepos",
        "link6-framepos",
    )


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
    gripper_kp: float = 50.0
    gripper_kd: float = 5.0
    arm_max_delta_per_step: float = 0.05
    # Terminate episodes when the arm hits a hard joint limit.  During the
    # arm-learning stage this kills the learning signal (exploration always
    # clips a joint), so the staged config can disable it and rely on the
    # ``arm_joint_limits`` reward penalty instead.
    terminate_on_arm_limits: bool = True
    # Arm-engagement gate radii in the armbase frame (meters).
    # When the EE goal is closer than engage_inner, the arm is fully engaged.
    # When it is farther than engage_outer, the arm stays near the home pose
    # and lets the base do the navigation.  Between the two, IK/policy and
    # home-return are linearly blended.
    engage_inner: float = 0.40
    engage_outer: float = 0.70
    # Home-return gain (per step) when the goal is outside the engagement gate.
    home_return_gain: float = 0.02
    # Max |q_target - q_actual| (rad).  The persistent IK target is anchored to
    # the actual joint positions and clamped to this bound, so a persistently
    # unreachable goal cannot drive the target to the soft joint limits (anti-
    # windup).  Default 0.08 rad ≈ 4.6°.
    max_target_error: float = 0.08


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
    scales: dict[str, float] = field(
        default_factory=lambda: {
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
        }
    )
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
        "kp": np.concatenate(
            [
                np.asarray(cc.arm_kp, dtype=np.float64),
                np.asarray([cc.gripper_kp], dtype=np.float64),
            ]
        ),
        "kd": np.concatenate(
            [
                np.asarray(cc.arm_kd, dtype=np.float64),
                np.asarray([cc.gripper_kd], dtype=np.float64),
            ]
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# Top-level EnvCfg
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class RangerBoxEEGoalConfig(EEGoalConfig):
    """RangerBox goal sampling: LOCAL (arm capture region) + EXTENDED (base nav).

    Goals are sampled by TRUE radial distance from the current EE
    (direction = random unit vector, then scale by a radius) — NOT an
    axis-aligned Cartesian box, which would let |delta| exceed the radius.

    - LOCAL goals (``local_fraction``): EE-to-goal distance in
      ``local_radius_range``.  These sit inside the arm's reliable capture
      radius and are IK-feasibility filtered at reset.
    - EXTENDED goals: EE-to-goal distance in ``extended_radius_range``.  These
      require the base to navigate the goal into the arm's capture region
      (they are NOT IK-feasibility filtered — the IK is not expected to reach
      them alone).

    ``capture_inner`` / ``capture_outer`` define the arm-capture region in
    EE-to-goal distance, used by the arm-engagement gate in apply_action:
    fully engaged within ``capture_inner``, blended between, held at the ready
    pose beyond ``capture_outer``.
    """

    local_fraction: float = 0.30
    # Measured reliable capture radius (benchmark_ranger_box_ik_capture_radius):
    # once10/hold10 > 0.95 within 0.15 m, drops to ~0.65 at 0.15-0.20 m.
    # LOCAL goals sit inside capture_inner (fully engaged); capture_outer is
    # the blend boundary toward the base-navigation regime.
    local_radius_range: tuple[float, float] = (0.10, 0.15)
    extended_radius_range: tuple[float, float] = (0.30, 0.70)
    capture_inner: float = 0.15
    capture_outer: float = 0.20


@registry.envcfg("RangerBoxReach")
@dataclass
class RangerBoxReachCfg(Go2ArmBaseCfg):
    scene: SceneCfg | None = field(default_factory=_default_ranger_box_scene)
    model_file: str = field(default_factory=_default_ranger_box_model_file)
    max_episode_seconds: float = 30.0
    init_state: InitState = field(default_factory=InitState)
    # Manipulation-ready arm pose (rad, j1..j6).  The arm RESETS to this folded,
    # well-conditioned pose (NOT the gravity-hanging equilibrium), so EE goals
    # sampled near the current EE are locally reachable.  When None, falls back
    # to the keyframe default.  Found by scripts/manip_loco/find_ranger_box_ready_pose.py.
    init_arm_pose: tuple[float, ...] | None = None
    control_config: RangerBoxControlConfig = field(default_factory=RangerBoxControlConfig)
    sensor: RangerBoxSensor = field(default_factory=RangerBoxSensor)
    noise_config: RangerBoxNoiseConfig = field(default_factory=RangerBoxNoiseConfig)
    goal_ee: RangerBoxEEGoalConfig = field(default_factory=RangerBoxEEGoalConfig)
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
    prev_ee_dist: np.ndarray


# ══════════════════════════════════════════════════════════════════════════
# Reward functions — all return positive values (negative in YAML scale)
# ══════════════════════════════════════════════════════════════════════════


def _ee_world_distance(ctx: _RewardContext) -> np.ndarray:
    """Euclidean distance EE→goal in world frame (used by several terms)."""
    return np.linalg.norm(ctx.ee_pos_world - ctx.world_ee_goal, axis=1)


def _reward_ee_distance(ctx: _RewardContext) -> np.ndarray:
    d = _ee_world_distance(ctx)
    sigma2 = ctx.sigma_ee * ctx.sigma_ee
    return np.exp(-(d * d) / sigma2)


def _reward_ee_distance_l2(ctx: _RewardContext) -> np.ndarray:
    diff = ctx.ee_pos_world - ctx.world_ee_goal
    return np.sum(diff * diff, axis=1)


def _reward_ee_progress(ctx: _RewardContext) -> np.ndarray:
    # Signed progress: positive when EE approaches the goal, negative when it
    # moves away.  Deliberately NOT clipped at zero — clipping allows a policy
    # to oscillate and harvest progress repeatedly (reward hacking).
    d = _ee_world_distance(ctx)
    return ctx.prev_ee_dist - d


def _reward_success_10cm(ctx: _RewardContext) -> np.ndarray:
    return (_ee_world_distance(ctx) < 0.10).astype(np.float64)


def _reward_success_05cm(ctx: _RewardContext) -> np.ndarray:
    return (_ee_world_distance(ctx) < 0.05).astype(np.float64)


def _reward_base_vel_xy(ctx: _RewardContext) -> np.ndarray:
    return ctx.linvel[:, 0] ** 2 + ctx.linvel[:, 1] ** 2


def _reward_base_stop_near(ctx: _RewardContext) -> np.ndarray:
    d = _ee_world_distance(ctx)
    near = np.exp(-((d / 0.20) ** 2))
    return (ctx.linvel[:, 0] ** 2 + ctx.linvel[:, 1] ** 2) * near


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
        env._ik_target[env_ids] = env._default_arm_angles
        env._success_hold_timer[env_ids] = 0
        env._success_once[env_ids] = False
        env._success_hold[env_ids] = False
        env._steps_to_success[env_ids] = 0
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
        # Initialize prev_ee_dist from the real EE→goal distance so the first
        # control step does not fabricate a spurious progress reward.
        ee_local_pos0, _ = env.get_ee_local_pose()
        ee_world0 = env.armbase_pos_world[env_ids] + np_quat_apply_batched(
            env.armbase_quat_world[env_ids], ee_local_pos0[env_ids]
        )
        env._prev_ee_dist[env_ids] = np.linalg.norm(ee_world0 - env.world_ee_goal[env_ids], axis=1)
        sliced_info: dict = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        ee_local_pos, _ = env.get_ee_local_pose()
        raw = env._compute_raw_obs(
            sliced_info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            ee_local_pos[env_ids],
            env.armbase_ee_goal[env_ids],
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

        self._num_action = 9

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
        self._prev_ee_dist = np.zeros(num_envs, dtype=np.float64)
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)

        # Success tracking (for held-success early termination + eval metrics).
        self._success_hold_timer = np.zeros((num_envs,), dtype=np.int32)
        self._success_once = np.zeros((num_envs,), dtype=bool)
        self._success_hold = np.zeros((num_envs,), dtype=bool)
        self._steps_to_success = np.zeros((num_envs,), dtype=np.int32)

        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._history_obs_buf = np.zeros((num_envs, H_a * _RAW_OBS_DIM), dtype=get_global_dtype())
        self._history_critic_buf = np.zeros(
            (num_envs, H_c * _RAW_OBS_DIM), dtype=get_global_dtype()
        )

        # Manipulation-ready arm pose: override the (gravity-hanging) keyframe
        # default with the folded, well-conditioned ready pose from config.
        # This flows into _init_qpos (reset), default_angles (obs baseline),
        # _ik_target init, and the home-return target.
        ready_pose = getattr(self._cfg, "init_arm_pose", None)
        if ready_pose is not None:
            arm_q = np.asarray(ready_pose, dtype=np.float64)
            if arm_q.shape != (6,):
                raise ValueError(
                    f"env.init_arm_pose must have shape (6,), got {arm_q.shape}"
                )
            arm_qpos_idx = (
                self._backend.get_joint_dof_pos_indices(self._cfg.asset.arm_joint_names) + 1
            )
            self._init_qpos[arm_qpos_idx] = arm_q
            self.default_angles[:6] = arm_q

        self._default_arm_angles = self.default_angles[:6].copy()

        # Persistent IK-only arm target reference (rad).  Only the IK delta and
        # the home-return term are integrated here.  The RL policy residual is
        # applied as an instantaneous offset on top of this target and is NOT
        # integrated — integrating it would accumulate DC bias (a small
        # constant action would steadily creep the target, mimicking arm sag).
        self._ik_target = (
            np.broadcast_to(self._default_arm_angles, (num_envs, 6)).astype(np.float64).copy()
        )

        self._arm_joint_upper = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14], dtype=np.float64)
        self._arm_joint_lower = np.array(
            [-3.92, -1.57, -2.86, -3.14, -3.14, -3.14], dtype=np.float64
        )

        # Soft joint limits: 5 % margin inside the hard limits so the
        # integrated target never saturates at the mechanical stop.
        _arm_range = self._arm_joint_upper - self._arm_joint_lower
        self._arm_soft_lower = self._arm_joint_lower + 0.05 * _arm_range
        self._arm_soft_upper = self._arm_joint_upper - 0.05 * _arm_range

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

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _init_buffers(self) -> None:
        dtype = get_global_dtype()
        raw_qpos = self._backend.get_keyframe_qpos("home")
        self._init_qpos = np.asarray(raw_qpos, dtype=dtype)
        # default_angles = gravity-settle equilibrium of the current physics
        # (gravcomp=1, joint damping=1, actuator kp/force limits), measured by
        # scripts/manip_loco/calibrate_ranger_box_settle.py.  Resetting at the
        # settled pose (instead of the old 0,-0.3,0.75,0,0.45,0 nominal) removes
        # the initial sag transient and keeps obs arm_diff ≈ 0 at reset.
        # Fallback arm control defaults (overridden by env.init_arm_pose when
        # set): the manipulation-ready folded pose, NOT the gravity-hanging
        # equilibrium.  See scripts/manip_loco/find_ranger_box_ready_pose.py.
        self.default_angles = np.array(
            [-0.0793, 0.0031, -2.1214, -1.6912, 2.118, 1.1986, 0.0], dtype=dtype
        )
        raw_qvel = self._backend.get_init_qvel()
        self._init_qvel = np.asarray(raw_qvel, dtype=dtype)

    # ── Action application ──────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions.copy()

        self._base_controller.step(actions[:, 0:3])

        arm_action = (
            state.info["last_actions"][:, 3:9]
            if self._cfg.control_config.simulate_action_latency
            else actions[:, 3:9]
        )

        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world
        )

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        dq_ik = self.compute_arm_ik_delta(
            self.armbase_ee_goal,
            ee_local_pos,
            self.ee_goal_orn_quat,
            ee_local_quat,
        )

        # Arm-engagement gate (Task 6): capture-region based on the EE-to-goal
        # distance, NOT goal-to-armbase distance and NOT dq_ik nonzero.  The
        # arm's reliable capture radius was measured empirically (~0.2 m), so:
        #   ee_error <= capture_inner  → arm fully engaged (arm_weight = 1)
        #   capture_inner < ee_error < capture_outer → linear blend
        #   ee_error >= capture_outer → arm holds the ready pose (arm_weight=0)
        # and the base is responsible for bringing the goal into the region.
        goal_cfg = self._cfg.goal_ee
        capture_inner = float(getattr(goal_cfg, "capture_inner", 0.18))
        capture_outer = float(getattr(goal_cfg, "capture_outer", 0.25))
        ee_error = np.linalg.norm(self.armbase_ee_goal - ee_local_pos, axis=1)
        arm_weight = np.clip(
            (capture_outer - ee_error) / max(capture_outer - capture_inner, 1e-6),
            0.0,
            1.0,
        )

        # Anti-windup on dq: once the ACTUAL joint is at the soft limit, stop
        # commanding further motion in the direction that would push it out.
        _limit_eps = 0.02
        q_actual = self.get_arm_dof_pos()
        qdot = self.get_arm_dof_vel()
        _at_upper = q_actual >= self._arm_soft_upper - _limit_eps
        _at_lower = q_actual <= self._arm_soft_lower + _limit_eps
        dq_ik = np.where(_at_upper & (dq_ik > 0), 0.0, dq_ik)
        dq_ik = np.where(_at_lower & (dq_ik < 0), 0.0, dq_ik)

        ik = self._cfg.ik
        home_return_gain = getattr(self._cfg.control_config, "home_return_gain", 0.02)
        home_delta = self._default_arm_angles - q_actual
        arm_residual = arm_action * self._cfg.control_config.arm_action_scale
        mode = getattr(ik, "controller_mode", "resolved_rate_damped")

        if mode == "integrated":
            # (ablation A) OLD chase-current target integration — kept only for
            # controller-ablation comparison.  The target is anchored to the
            # actual pose, but its constant small offset ahead of q_actual makes
            # the position actuator keep pushing, causing overshoot / divergence.
            q_candidate = (
                q_actual
                + arm_weight[:, None] * ik.gain * dq_ik
                + (1.0 - arm_weight[:, None]) * home_return_gain * home_delta
            )
            delta_target = q_candidate - self._ik_target
            max_delta = getattr(self._cfg.control_config, "arm_max_delta_per_step", 0.1)
            if max_delta > 0:
                delta_target = np.clip(delta_target, -max_delta, max_delta)
            self._ik_target = self._ik_target + delta_target
            max_target_error = getattr(self._cfg.control_config, "max_target_error", 0.08)
            self._ik_target = np.clip(
                self._ik_target, q_actual - max_target_error, q_actual + max_target_error
            )
            self._ik_target = np.clip(self._ik_target, self._arm_soft_lower, self._arm_soft_upper)
            arm_ctrl = np.clip(
                self._ik_target + arm_residual, self._arm_soft_lower, self._arm_soft_upper
            )
        else:
            # Resolved-rate position target (modes B/C): NO integration.
            #   dq_cmd = dq_ik - kv * qdot   (velocity damping, mode C)
            #   q_target = q_actual + arm_weight * gain * dq_cmd + home-return
            # The actuator pulls toward q_target, which is always anchored to the
            # actual pose, so it never commands a persistent lead that would
            # overshoot.  Soft-limit clip is the joint-limit anti-windup.
            dq_cmd = dq_ik
            if mode == "resolved_rate_damped":
                kv = float(getattr(ik, "velocity_damping", 0.0))
                dq_cmd = dq_ik - kv * qdot
            dq_cmd = np.clip(dq_cmd, -ik.dq_clip, ik.dq_clip)
            q_target = (
                q_actual
                + arm_weight[:, None] * ik.gain * dq_cmd
                + (1.0 - arm_weight[:, None]) * home_return_gain * home_delta
            )
            q_target = np.clip(q_target, self._arm_soft_lower, self._arm_soft_upper)
            self._ik_target[:] = q_target  # mirror for diagnostics / tests
            arm_ctrl = np.clip(
                q_target + arm_residual, self._arm_soft_lower, self._arm_soft_upper
            )
        grip_ctrl = np.zeros((actions.shape[0], 1), dtype=np.float64)
        ctrl = np.concatenate([arm_ctrl, grip_ctrl], axis=1)

        # Apply base velocity BEFORE physics step so MuJoCo integrates from it
        self._base_controller.apply_velocity()

        return ctrl.astype(get_global_dtype())

    # ── Observation ─────────────────────────────────────────────────

    def _compute_raw_obs(
        self,
        info,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        ee_local_pos,
        armbase_ee_goal,
        *,
        add_noise=True,
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

        last_actions = info.get("current_actions", np.zeros((n, 9), dtype=get_global_dtype()))
        ee_error = armbase_ee_goal - ee_local_pos

        return np.concatenate(
            [
                linvel.astype(get_global_dtype()),  # 3
                gyro.astype(get_global_dtype()),  # 3
                (-gravity).astype(get_global_dtype()),  # 3
                arm_diff.astype(get_global_dtype()),  # 6
                arm_vel.astype(get_global_dtype()),  # 6
                ee_local_pos.astype(get_global_dtype()),  # 3
                armbase_ee_goal.astype(get_global_dtype()),  # 3
                ee_error.astype(get_global_dtype()),  # 3
                last_actions.astype(get_global_dtype()),  # 9
            ],
            axis=1,
        )

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
            return {"obs": self._history_obs_buf.copy(), "critic": self._history_critic_buf.copy()}
        else:
            if H_a > 1:
                self._history_obs_buf[env_ids] = np.roll(self._history_obs_buf[env_ids], -D, axis=1)
            self._history_obs_buf[env_ids, -D:] = raw_obs
            if H_c > 1:
                self._history_critic_buf[env_ids] = np.roll(
                    self._history_critic_buf[env_ids], -D, axis=1
                )
            self._history_critic_buf[env_ids, -D:] = critic_step
            return {
                "obs": self._history_obs_buf[env_ids].copy(),
                "critic": self._history_critic_buf[env_ids].copy(),
            }

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {"obs": H_a * _RAW_OBS_DIM, "critic": H_c * _RAW_OBS_DIM}

    # ── State update ────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # SE(2) planar lock: correct any z/roll/pitch drift that accumulated
        # during the physics step, THEN refresh sensors so the subsequent
        # observation/reward reads see the locked state.
        self._apply_se2_lock()

        linvel = self._backend.get_sensor_data(self._cfg.sensor.local_linvel).astype(
            get_global_dtype()
        )
        gyro = self.get_gyro()
        gravity = self._get_projected_gravity()

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        arm_pos = self.get_arm_dof_pos()
        arm_vel = self.get_arm_dof_vel()

        self.armbase_pos_world = self._backend.get_sensor_data(self._cfg.sensor.armbase_world_pos)
        self.armbase_quat_world = self._backend.get_sensor_data(self._cfg.sensor.arm_ref_world_quat)

        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world
        )

        ee_pos_world = self.armbase_pos_world + np_quat_apply_batched(
            self.armbase_quat_world, ee_local_pos
        )

        # Eval trajectory history (play-only; no-op during training).
        if self._record_traj:
            base_pos = self._backend.get_base_pos()
            self._traj_base = np.roll(self._traj_base, -1, axis=1)
            self._traj_base[:, -1] = base_pos
            self._traj_ee = np.roll(self._traj_ee, -1, axis=1)
            self._traj_ee[:, -1] = ee_pos_world

        tilt_sq = gravity[:, 0] ** 2 + gravity[:, 1] ** 2
        terminated = tilt_sq > np.sin(1.0) ** 2
        if self._cfg.control_config.terminate_on_arm_limits:
            limit_violated = (
                (arm_pos > self._arm_joint_upper) | (arm_pos < self._arm_joint_lower)
            ).any(axis=1)
            terminated = terminated | limit_violated

        # Held-success: EE within 10 cm continuously for >= hold_time (0.5 s)
        # → early termination + success flag, so easy episodes don't run the
        # full 10 s harvesting distance/success reward.
        ee_dist = np.linalg.norm(ee_pos_world - self.world_ee_goal, axis=1)
        within_10cm = ee_dist < 0.10
        hold_steps = max(1, int(0.5 / self._cfg.ctrl_dt))
        self._success_hold_timer = np.where(within_10cm, self._success_hold_timer + 1, 0)
        self._success_once |= within_10cm
        newly_held = self._success_hold_timer >= hold_steps
        self._success_hold |= newly_held
        # Record steps-to-first-success for eval
        first_success = self._success_once & (self._steps_to_success == 0)
        self._steps_to_success = np.where(
            first_success, state.info.get("steps", 0) + 1, self._steps_to_success
        )
        terminated = terminated | newly_held

        prev_arm_vel_saved = self._prev_arm_vel.copy()
        self._prev_arm_vel = arm_vel.copy()

        ctx = _RewardContext(
            info=state.info,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            arm_pos=arm_pos,
            arm_vel=arm_vel,
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
            current_actions=state.info.get("current_actions", np.zeros((self._num_envs, 9))),
            prev_ee_dist=self._prev_ee_dist,
        )

        # Track EE distance for next step's progress reward
        self._prev_ee_dist = _ee_world_distance(ctx)

        reward = self._compute_reward(ctx)
        obs = self._compute_obs(
            state.info,
            linvel,
            gyro,
            gravity,
            arm_pos,
            arm_vel,
            ee_local_pos,
            self.armbase_ee_goal,
            add_noise=True,
        )
        for k in obs:
            obs[k] = np.nan_to_num(obs[k], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        nan_terminated = np.any(np.isnan(arm_pos), axis=1)

        # Publish success/debug metrics for eval + logging.
        state.info["ee_dist"] = ee_dist
        state.info["success_once"] = self._success_once.copy()
        state.info["success_hold"] = self._success_hold.copy()
        state.info["steps_to_success"] = self._steps_to_success.copy()
        state.info["q_target"] = self._ik_target.copy()
        state.info["q_actual"] = arm_pos.copy()

        return state.replace(
            obs=obs,
            reward=reward,
            terminated=np.logical_or(terminated, nan_terminated),
        )

    def _apply_se2_lock(self) -> None:
        """SE(2) planar lock: pin z, zero roll/pitch, preserve yaw.

        Called AFTER physics step.  Writes corrected qpos + qvel, then
        refreshes sensor data so subsequent reads see the locked state.
        """
        _qpos = self._backend._qpos_view
        qvel = self._backend._physics_state[
            :, self._backend._idx_qvel : self._backend._idx_qvel + self._backend.nv
        ]

        # Pin z to initial height
        _qpos[:, 2] = self._base_controller._init_base_z

        # Extract yaw from current quaternion, rebuild yaw-only quaternion
        qw, qx, qy, qz = (_qpos[:, i] for i in range(3, 7))
        yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        half_yaw = yaw * 0.5
        _qpos[:, 3] = np.cos(half_yaw)  # qw
        _qpos[:, 4] = 0.0  # qx
        _qpos[:, 5] = 0.0  # qy
        _qpos[:, 6] = np.sin(half_yaw)  # qz

        # Zero vz, wx, wy (indices 2, 3, 4 in freejoint qvel)
        qvel[:, 2] = 0.0
        qvel[:, 3] = 0.0
        qvel[:, 4] = 0.0

        # Refresh sensor data from corrected state
        self._backend.forward_sensors()

    # ── Reward ──────────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_cfg = self._cfg.reward_config

        # Wrap common rewards to map context field names
        def _action_rate_wrapper(ctx):
            current = ctx.info.get("current_actions", np.zeros((ctx.num_envs, 9)))
            prev = ctx.info.get("last_actions", np.zeros((ctx.num_envs, 9)))
            return np.sum((current - prev) ** 2, axis=1)

        def _similar_to_default_wrapper(ctx):
            return np.sum(np.abs(ctx.arm_pos - ctx.default_arm_angles), axis=1)

        def _alive_wrapper(ctx):
            return np.zeros(ctx.num_envs)  # disabled: no task-independent reward

        # Arm-base clearance / collision reward using proper signed distance
        # to the base collision box (0.55×0.38×0.20 m in base-local).
        # Box centre is offset from the armbase point by the XML geometry:
        #   base_collision pos=(0.12,0,0.18), armbasepoint pos=(0.2462,0,0.2765)
        #   → offset = (0.12-0.2462, 0, 0.18-0.2765) = (-0.1262, 0, -0.0965)
        _BASE_BOX_HALF = np.array([0.55, 0.38, 0.20], dtype=np.float64)
        _BOX_CENTRE_OFFSET = np.array([-0.1262, 0.0, -0.0965], dtype=np.float64)

        def _arm_point_signed_dist(p_w, box_centre_w, base_quat_w):
            """Signed distance from point(s) to axis-aligned base box in base frame."""
            p_local = np_quat_apply_batched(
                np_quat_conjugate_batched(base_quat_w),
                p_w - box_centre_w,
            )
            q = np.abs(p_local) - _BASE_BOX_HALF
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            return outside + inside  # >0 outside, <0 inside

        def _min_arm_signed_dist(ctx):
            """Min signed distance over arm links (Link2-6) + EE to base box."""
            base_quat_w = ctx.armbase_quat_world
            box_centre_w = ctx.armbase_pos_world + np_quat_apply_batched(
                base_quat_w,
                np.broadcast_to(_BOX_CENTRE_OFFSET, (ctx.num_envs, 3)),
            )
            # Collect world positions of arm links + EE
            points = [ctx.ee_pos_world]
            for name in self._cfg.sensor.link_pos:
                points.append(self._backend.get_sensor_data(name))
            sd = np.stack(
                [_arm_point_signed_dist(p, box_centre_w, base_quat_w) for p in points],
                axis=1,
            )
            return np.min(sd, axis=1)

        def _arm_base_clearance_fn(ctx):
            sd = _min_arm_signed_dist(ctx)
            margin = 0.05
            return np.square(np.maximum(margin - sd, 0.0))

        def _arm_base_collision_fn(ctx):
            sd = _min_arm_signed_dist(ctx)
            penetration = np.maximum(-sd, 0.0)
            return penetration * penetration

        self._reward_fns: dict[str, Any] = {
            "ee_distance": _reward_ee_distance,
            "ee_distance_l2": _reward_ee_distance_l2,
            "ee_progress": _reward_ee_progress,
            "success_10cm": _reward_success_10cm,
            "success_05cm": _reward_success_05cm,
            "base_vel_xy": _reward_base_vel_xy,
            "base_stop_near_goal": _reward_base_stop_near,
            "base_vel_yaw": _reward_base_vel_yaw,
            "arm_dof_vel": _reward_arm_dof_vel,
            "arm_dof_acc": _reward_arm_dof_acc,
            "arm_joint_limits": _reward_arm_joint_limits,
            "action_rate": _action_rate_wrapper,
            "similar_to_default": _similar_to_default_wrapper,
            "alive": _alive_wrapper,
            "arm_base_clearance": _arm_base_clearance_fn,
            "arm_base_collision": _arm_base_collision_fn,
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
        return np.einsum(
            "nij,j->ni", np.swapaxes(R_wb, 1, 2), np.array([0.0, 0.0, -1.0], dtype=R_wb.dtype)
        )

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
        # LOCAL (True) vs EXTENDED (False) goal type per env, set at reset.
        self._goal_is_local = np.zeros((self._num_envs,), dtype=bool)
        # Eval trajectory recording (play/video only).  Recording is disabled
        # during training; the first eval_visualization_markers() call enables
        # it.  NaN marks slots not yet filled so the renderer skips them.
        self._record_traj = False
        self._traj_len = 60
        self._traj_base = np.full(
            (self._num_envs, self._traj_len, 3), np.nan, dtype=np.float64
        )
        self._traj_ee = np.full(
            (self._num_envs, self._traj_len, 3), np.nan, dtype=np.float64
        )

    @property
    def curr_ee_goal_world(self) -> np.ndarray:
        """Goal world position — used by playback for the red goal marker."""
        return self.world_ee_goal

    def eval_visualization_markers(self) -> np.ndarray | None:
        """Return playback marker data for goal/EE spheres + trajectory trails.

        Shape (num_envs, 6 + 6*K) where K = self._traj_len:
          [0:3]        goal world (red sphere)
          [3:6]        current EE world (green sphere)
          [6:6+3K]     base trajectory (blue trail, NaN = unfilled slot)
          [6+3K:6+6K]  EE trajectory (green trail, NaN = unfilled slot)

        The first call enables trajectory recording (play-only, no training
        overhead).  ``None`` disables markers.
        """
        self._record_traj = True
        ee_local, _ = self.get_ee_local_pose()
        ee_world = self.armbase_pos_world + np_quat_apply_batched(self.armbase_quat_world, ee_local)
        base_flat = self._traj_base.reshape(self._num_envs, -1)
        ee_flat = self._traj_ee.reshape(self._num_envs, -1)
        return np.concatenate([self.world_ee_goal, ee_world, base_flat, ee_flat], axis=1)

    def eval_visualization_text(self) -> list[str]:
        """Per-env debug strings for the video overlay (Task 8).

        Format: "env{i} {LOC|EXT} d={EE-goal dist} aw={arm_weight} {flag}".
        Shows the base→capture-region→arm-takeover transition in playback.
        """
        ee_local, _ = self.get_ee_local_pose()
        ee_world = self.armbase_pos_world + np_quat_apply_batched(self.armbase_quat_world, ee_local)
        d = np.linalg.norm(ee_world - self.world_ee_goal, axis=1)
        goal_cfg = self._cfg.goal_ee
        ci = float(getattr(goal_cfg, "capture_inner", 0.18))
        co = float(getattr(goal_cfg, "capture_outer", 0.25))
        aw = np.clip((co - d) / max(co - ci, 1e-6), 0.0, 1.0)
        out = []
        for i in range(self._num_envs):
            gtype = "LOC" if self._goal_is_local[i] else "EXT"
            flag = "SUCCESS" if self._success_once[i] else "run"
            out.append(f"env{i} {gtype} d={d[i]:.2f} aw={aw[i]:.2f} {flag}")
        return out

    def reset_ee_goals(self, env_ids: np.ndarray) -> None:
        """Sample LOCAL / EXTENDED EE goals by TRUE radial distance (Task 4).

        goal = current EE + r * u,  u = random unit vector, r ~ radius range.

        - LOCAL (``local_fraction``): r in ``local_radius_range`` — inside the
          arm's reliable capture radius; fully IK-feasibility filtered.
        - EXTENDED: r in ``extended_radius_range`` — requires base navigation;
          only physical sanity is checked (floor / chassis), NOT IK feasibility
          (the base brings the goal into the capture region).

        Rejection-sampled; fallback after all attempts is a small forward-down
        reach (guaranteed locally reachable).
        """
        n = len(env_ids)
        rng = np.random
        goal_cfg = self._cfg.goal_ee

        ee_local, _ = self.get_ee_local_pose()
        ee_local = ee_local[env_ids]  # (n, 3) armbase frame
        armbase_pos = self.armbase_pos_world[env_ids]
        armbase_quat = self.armbase_quat_world[env_ids]

        is_local = rng.random(n) < float(goal_cfg.local_fraction)
        self._goal_is_local[env_ids] = is_local

        def _sample_radial(r_lo: float, r_hi: float, size: int) -> np.ndarray:
            v = rng.standard_normal((size, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
            r = rng.uniform(r_lo, r_hi, size=size)
            return v * r[:, None]

        delta = np.zeros((n, 3), dtype=np.float64)
        n_loc = int(is_local.sum())
        n_ext = n - n_loc
        if n_loc > 0:
            delta[is_local] = _sample_radial(*goal_cfg.local_radius_range, n_loc)
        if n_ext > 0:
            delta[~is_local] = _sample_radial(*goal_cfg.extended_radius_range, n_ext)

        goal_local = ee_local + delta
        goals_world = armbase_pos + np_quat_apply_batched(armbase_quat, goal_local)
        goals_world[:, 2] = np.maximum(goals_world[:, 2], 0.15)

        bad = self._goal_infeasible(env_ids, goal_local, goals_world, require_ik=is_local)
        for _attempt in range(goal_cfg.num_resample_attempts):
            if not bad.any():
                break
            n_bad = int(bad.sum())
            bad_local = is_local[bad]
            n_bl = int(bad_local.sum())
            n_be = n_bad - n_bl
            new_delta = np.zeros((n_bad, 3), dtype=np.float64)
            if n_bl > 0:
                new_delta[bad_local] = _sample_radial(*goal_cfg.local_radius_range, n_bl)
            if n_be > 0:
                new_delta[~bad_local] = _sample_radial(*goal_cfg.extended_radius_range, n_be)
            delta[bad] = new_delta
            goal_local[bad] = ee_local[bad] + new_delta
            goals_world[bad] = armbase_pos[bad] + np_quat_apply_batched(
                armbase_quat[bad], goal_local[bad]
            )
            goals_world[bad, 2] = np.maximum(goals_world[bad, 2], 0.15)
            bad = self._goal_infeasible(env_ids, goal_local, goals_world, require_ik=is_local)

        if bad.any():
            # Fallback: small forward-down reach (guaranteed locally reachable).
            fb = np.array([0.10, 0.0, -0.05], dtype=np.float64)
            goal_local[bad] = ee_local[bad] + fb[None, :]
            goals_world[bad] = armbase_pos[bad] + np_quat_apply_batched(
                armbase_quat[bad], goal_local[bad]
            )
            goals_world[bad, 2] = np.maximum(goals_world[bad, 2], 0.15)

        self.world_ee_goal[env_ids] = goals_world
        self._arm_goal_timer[env_ids] = 0

    def _goal_infeasible(
        self,
        env_ids: np.ndarray,
        goal_local: np.ndarray,
        goals_world: np.ndarray,
        *,
        require_ik: np.ndarray,
    ) -> np.ndarray:
        """Return True where a goal must be rejected.

        For every goal: below floor or inside the chassis.  For LOCAL goals
        (``require_ik`` True): additionally the damped-LS correction from the
        current arm state must stay inside the soft range and move the EE
        toward the goal.  EXTENDED goals are NOT rejected for IK reachability —
        the base is expected to navigate them into the arm capture region.
        """
        base_pos_w = self._backend.get_base_pos()[env_ids]
        base_quat_w = self._backend.get_base_quat()[env_ids]

        # 1. Inside the chassis bounding box.
        goals_base = np_quat_apply_batched(
            np_quat_conjugate_batched(base_quat_w), goals_world - base_pos_w
        )
        inside_chassis = (
            (np.abs(goals_base[:, 0]) < 0.55)
            & (np.abs(goals_base[:, 1]) < 0.38)
            & (goals_base[:, 2] < 0.48)
        )
        # 2. Below floor.
        below_floor = goals_world[:, 2] < 0.15

        bad = inside_chassis | below_floor
        if not np.asarray(require_ik).any():
            return bad

        # 3. IK feasibility from the current (ready-pose) arm state: the
        #    damped-LS correction must stay inside the soft range and move the
        #    EE toward the goal.  Only applied to LOCAL goals.
        ee_local, _ = self.get_ee_local_pose()
        ee_local = ee_local[env_ids]
        q_actual = self.get_arm_dof_pos()[env_ids]
        pos_err = goal_local - ee_local
        jacp_w, _ = self._backend.get_site_jacobian_w(
            self._ee_site_id, self._arm_jacobian_dof_indices
        )
        ref_rot = np_matrix_from_quat(
            self._backend.get_sensor_data(self._cfg.sensor.arm_ref_world_quat)[env_ids]
        )
        jacp_b = np.matmul(np.swapaxes(ref_rot, 1, 2), jacp_w[env_ids])  # (n,3,6)
        eye3 = np.eye(3, dtype=jacp_b.dtype)[None, :, :]
        lhs = np.matmul(jacp_b, np.swapaxes(jacp_b, 1, 2)) + eye3 * (self._cfg.ik.damping ** 2)
        rhs = pos_err[:, :, None]
        dq = np.matmul(np.swapaxes(jacp_b, 1, 2), np.linalg.solve(lhs, rhs))[:, :, 0]

        margin = 0.05 * (self._arm_joint_upper - self._arm_joint_lower)
        lo = self._arm_soft_lower + margin
        hi = self._arm_soft_upper - margin
        q_target = q_actual + dq
        out_of_range = ((q_target < lo) | (q_target > hi)).any(axis=1)

        pred = np.matmul(jacp_b, dq[:, :, None])[:, :, 0]
        no_progress = (pred * pos_err).sum(axis=1) <= 0

        ik_bad = out_of_range | no_progress
        ik_bad = np.where(np.asarray(require_ik), ik_bad, False)
        return bad | ik_bad

    def _world_goal_to_armbase(self, world_goal, armbase_pos, armbase_quat):
        rel = world_goal - armbase_pos
        q_conj = np_quat_conjugate_batched(armbase_quat)
        return np_quat_apply_batched(q_conj, rel)

    def _compute_obs(
        self,
        info,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        ee_local_pos,
        armbase_ee_goal,
        *,
        add_noise=True,
    ):
        raw = self._compute_raw_obs(
            info,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            ee_local_pos,
            armbase_ee_goal,
            add_noise=add_noise,
        )
        return self._update_history(raw)
