# RangerBoxReach — 移动操作 EE 到达 设计文档

日期：2026-06-29 | 状态：v3 审阅修正完成，待审阅 | 任务：RangerboxCR10Lidar 移动底盘 + 臂 EE 目标到达

---

## 第一节：文件变更清单

### 新增代码/配置文件（5 个）

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/ranger_box/__init__.py` | 包文件，导入 `RangerBoxReachEnv` 触发注册 |
| `src/unilab/envs/locomotion/ranger_box/reach_env.py` | Env + Dataclass cfg、DR Provider、reward |
| `src/unilab/envs/locomotion/ranger_box/base_velocity_controller.py` | A+ 方案核心：延迟环 + 限幅 + 噪声 + 轮子 IK（全向量化） |
| `tests/test_ranger_box_reach.py` | 测试文件 |
| `conf/ppo/task/ranger_box_reach/mujoco.yaml` | PPO × MuJoCo owner YAML |

### 新增资产文件

| 目录 | 内容 |
|------|------|
| `src/unilab/assets/robots/ranger_box/` | `robot.xml`、`scene_flat.xml`、`meshes/*.obj` (26 个) |

### 需修改现有文件（3 个）

| 文件 | 修改内容 | 风险 |
|------|---------|------|
| `src/unilab/envs/locomotion/__init__.py` | `__unilab_registry_modules__` 加 `"unilab.envs.locomotion.ranger_box"`（包级别） | 低 |
| `src/unilab/base/backend/base.py` | 新增 `set_root_planar_velocity`、`set_joint_qpos`、`set_joint_qvel` 三个方法（非抽象，默认抛 `NotImplementedError`） | 中 |
| `src/unilab/base/backend/mujoco/backend.py` | 实现上述三个方法：只写 freejoint qvel 的 vx/vy/wz，保留 vz/wx/wy | 中 |

### 不修改的部分

| 文件 | 原因 |
|------|------|
| `scripts/train_rsl_rl.py` | Hydra config 驱动，不感知具体 task |
| `src/unilab/training/run.py` | 通用训练流程 |
| `src/unilab/base/registry.py` | `ensure_registries()` 自动发现 |
| `src/unilab/envs/locomotion/go2_arm/` | 仅作为父类被继承 |

---

## 第二节：robot.xml 适配设计

### 源 → 目标变更

| 问题 | 现状（foropenpi） | 目标（UniLab） |
|------|-------------------|----------------|
| 基座自由度 | 无 freejoint | 加 `<freejoint/>`（物理上锁定为 SE(2) planar，见 §2.1） |
| 臂执行器 | `<motor>`（力矩模式） | `<position>`（位置控制） |
| 传感器 | 仅 `force_ee` / `torque_ee` | RL 最小闭环传感器（见下表） |
| 夹爪执行器 | `<position>` finger1 + equality | **保留不变** |
| 转向/车轮 | 无 actuator | **保留无 actuator**；geom 加 `contype="0" conaffinity="0"` |
| 网格路径 | `meshes/xxx.obj` | 复制到 `src/unilab/assets/robots/ranger_box/meshes/` |

### 2.1 SE(2) Planar Lock（关键设计决策）

freejoint 提供 6-DOF，但 v1 明确采用 **SE(2) 运动学底盘**：

- 每步 `set_root_planar_velocity` 只写 vx/vy/wz（xy 速度 + yaw rate）
- vz、wx、wy **固定为 0**（z 固定为初始高度，roll/pitch 固定为 0）
- 不需要轮地接触模型支撑底盘

**同步调整：**
- `base_vel_z` reward scale → **0.0**（无控制输入）
- `base_orientation` reward scale → **0.0**（roll/pitch 恒为 0）
- `base_height` reward scale → **0.0**（z 固定）
- `push_robots` → **false**（无支撑接触）
- wheel geom 设 `contype="0" conaffinity="0"`

若后续 v2 需要倾斜/高度动力学，再引入轮地支撑接触模型。

### 执行器（`<actuator>` 块，共 7 个）

```xml
<actuator>
  <position name="cr10_joint1_act" joint="cr10_joint1" kp="100" ctrlrange="-3.92 0.94"/>
  <position name="cr10_joint2_act" joint="cr10_joint2" kp="110" ctrlrange="-1.57 1.57"/>
  <position name="cr10_joint3_act" joint="cr10_joint3" kp="95"  ctrlrange="-2.86 2.86"/>
  <position name="cr10_joint4_act" joint="cr10_joint4" kp="50"  ctrlrange="-3.14 3.14"/>
  <position name="cr10_joint5_act" joint="cr10_joint5" kp="50"  ctrlrange="-3.14 3.14"/>
  <position name="cr10_joint6_act" joint="cr10_joint6" kp="50"  ctrlrange="-3.14 3.14"/>
  <position name="gripper_finger1_joint_act" joint="gripper_finger1_joint" kp="500" ctrlrange="0 0.65"/>
</actuator>
```

- 6 arm + 1 gripper = **7 个 position actuators**
- `kp` 运行时通过 `position_actuator_gains` 注入

### 新增 Backend Contract 方法

| 方法 | 声明位置 | 实现位置 | 用途 |
|------|---------|---------|------|
| `set_root_planar_velocity(lin_vel_xy, yaw_rate, preserve_uncontrolled=True)` | `base/backend/base.py` | `mujoco/backend.py` | 只写 freejoint qvel 的 vx/vy/wz，保留 vz/wx/wy |
| `set_joint_qpos(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qpos（轮子可视化） |
| `set_joint_qvel(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qvel（轮子可视化） |

三个方法在 `SimBackend` 默认抛 `NotImplementedError`（非抽象），仅在 `MuJoCoBackend` 实现。

### 新增传感器（最小 RL 闭环）

| 类别 | sensor | 数量 | 用途 |
|------|--------|------|------|
| IMU（`imu` site on base body） | `gyro`、`framequat`、`velocimeter`、`framezaxis` | 4 | base 角速度、姿态、base-frame 线速度、Z 轴 |
| 臂关节状态 | backend `get_joint_dof_pos_indices/vel_indices` | 6+6 | 直接通过 joint index 读取 |
| 末端位姿（相对 armbase） | `framepos`、`framequat` | 2 | EE goal tracking |
| armbase 位姿（world） | `framepos`、`framequat` | 2 | world↔armbase 转换 |
| 夹爪 | backend joint index | 1 | 直接通过 index 读取 |

> **注意：** 单个 `<jointpos>` 只能对应一个 joint。arm/gripper/steering/wheel 状态统一通过 backend joint DOF index 获取，不在 XML 中添加聚合 joint sensor。`velocimeter`（非 `framelinvel`）输出 body-local 线速度。
>
> **v1 暂不加入：** `jointactuatorfrc`、`touch`、`accelerometer`。

### 轮子可视化

```
每步流程:
  策略输出 action[:, 0:3] = (vx_base, vy_base, vyaw_base)   ← base-frame，(N,3)

  1. Scale → clip → latency ring → accel limit → first-order → noise → final clip
     （详见第三节，全向量化）

  2. 轮子 IK（base-frame，参数来自 cfg.asset）:
       4 个轮子位置：(+0.445,±0.28), (-0.445,±0.28), wheel_radius=0.152m

       vx_i = vx_base - vyaw_base * y_i     (N,4)
       vy_i = vy_base + vyaw_base * x_i     (N,4)
       steer = wrap_to_pi(atan2(vy_i, vx_i))
       omega = sqrt(vx_i² + vy_i²) / wheel_radius

  3. backend 写入（wheel geom contype=0 conaffinity=0）:
       backend.set_joint_qpos(steering_names, steer)
       backend.set_joint_qvel(wheel_names, omega)

  4. World-frame 转换:
       v_world = quat_rotate(base_quat, [vx, vy, 0])
       w_world = quat_rotate(base_quat, [0, 0, vyaw])
       backend.set_root_planar_velocity(v_world[:, :2], w_world[:, 2:],
                                         preserve_uncontrolled=True)
```

### Keyframe（位于 `scene_flat.xml`，不在 `robot.xml`）

```xml
<keyframe>
  <key name="home"
    qpos="0 0 0.278  1 0 0 0  0 0 0 0 0 0 0 0  0 -0.3 0.75 0 0.45 0  0 0 0 0 0 0 0 0"
    ctrl="0 -0.3 0.75 0 0.45 0  0"
  />
</keyframe>
```

- `qpos` = 7(freejoint) + 8(base) + 6(arm) + 8(gripper) = **29**（XML 属性内不能有注释）
- `ctrl` = 6(arm) + 1(gripper) = **7**

---

## 第三节：BaseVelocityController 设计

### 3.1 为什么是"方案 A+"？

- MuJoCo freejoint 提供 6-DOF 自由体运动学
- 训练效率远高于全接触动力学
- Domain randomization 在控制器层实现
- v1 锁定为 SE(2) planar

### 3.2 控制器架构（全向量化）

```
action[:, 0:3] = (vx_des, vy_des, vyaw_des)    ← (N,3)

     │
     ▼
┌──────────────────────────────────────────────────────┐
│  1. scale → 2. clip → 3. latency ring (L,N,3)       │
│  → 4. accel limit → 5. first-order → 6. noise       │
│  → 7. final clip → 8. wheel IK (N,4)                │
│  → 9. world-frame convert                           │
│                                                      │
│  Output: set_root_planar_velocity() + wheel viz      │
└──────────────────────────────────────────────────────┘
```

### 3.3 Step 1-2：Scale + Clip

```python
v_cmd[:, 0:2] = np.clip(action[:, 0:2] * action_scale_lin, -max_lin_vel, max_lin_vel)
v_cmd[:, 2]   = np.clip(action[:, 2]   * action_scale_ang, -max_ang_vel, max_ang_vel)
```

| 参数 | 默认值 |
|------|--------|
| `max_lin_vel` | 1.5 m/s |
| `max_ang_vel` | 1.0 rad/s |
| `action_scale_lin` | 1.5 |
| `action_scale_ang` | 1.0 |

### 3.4 Step 3：延迟环（NumPy ring buffer，无 Python deque）

```python
# 初始化
self.latency_ring = np.zeros((max_latency_steps + 1, num_envs, 3), dtype=np.float32)
self.latency_steps = np.zeros(num_envs, dtype=np.int32)
self.latency_write_ptr = np.zeros(num_envs, dtype=np.int32)

# reset
new_steps = rng.integers(0, max_latency_steps + 1, size=len(env_ids))
self.latency_steps[env_ids] = new_steps
self.latency_write_ptr[env_ids] = 0
self.latency_ring[:, env_ids, :] = 0.0

# step
L = ring.shape[0]; wp = write_ptr
ring[wp, np.arange(N), :] = v_cmd
rp = (wp - latency_steps) % L
v_cmd_delayed = ring[rp, np.arange(N), :]
write_ptr = (wp + 1) % L
```

- `max_latency_steps` 默认 4（80ms @ 50Hz）

### 3.5 Step 4：加速度限幅

```python
dv[:, 0:2] = np.clip(dv[:, 0:2], -max_lin_acc * dt, max_lin_acc * dt)
dv[:, 2]   = np.clip(dv[:, 2],   -max_ang_acc * dt, max_ang_acc * dt)
v_target = v_real + dv
```

| 参数 | 默认值 |
|------|--------|
| `max_lin_acc` | 1.5 m/s² |
| `max_ang_acc` | 3.0 rad/s² |

### 3.6 Step 5：一阶速度响应

```python
alpha = dt / (tau + dt)
v_real = v_real + alpha * (v_target - v_real)
```

- `tau` 默认 0.05s

### 3.7 Step 6-7：噪声 + Final Clip

```python
noise = rng.standard_normal((N, 3), dtype=np.float32)
noise[:, 0:2] *= noise_scale * max_lin_vel
noise[:, 2]   *= noise_scale * max_ang_vel
v_real = v_real + noise
v_real = np.clip(v_real, -max_vel_arr, max_vel_arr)
```

噪声在一阶滤波**之后**（否则会被平滑）。注意使用 `rng.standard_normal()`，不是 `randn()`。

### 3.8 Step 8：轮子可视化

```python
vx_i = v_real[:, 0:1] - v_real[:, 2:3] * y[None, :]  # (N, 4)
vy_i = v_real[:, 1:2] + v_real[:, 2:3] * x[None, :]
steer = np.arctan2(vy_i, vx_i)
omega = np.sqrt(vx_i**2 + vy_i**2) / wheel_radius
```

### 3.9 Step 9：World-Frame 转换

```python
v_body = np.concatenate([v_real[:, 0:2], np.zeros((N, 1))], axis=1)
w_body = np.concatenate([np.zeros((N, 2)), v_real[:, 2:3]], axis=1)
v_world = quat_rotate(base_quat, v_body)
w_world = quat_rotate(base_quat, w_body)
self._backend.set_root_planar_velocity(v_world[:, :2], w_world[:, 2:],
                                        preserve_uncontrolled=True)
```

### 3.10 控制器生命周期

```python
class BaseVelocityController:
    def __init__(self, cfg, dt, backend, asset_cfg, num_envs):
        self.v_real = np.zeros((num_envs, 3), dtype=np.float32)
        self.latency_ring = np.zeros((cfg.max_latency_steps + 1, num_envs, 3), dtype=np.float32)
        self.latency_steps = np.zeros(num_envs, dtype=np.int32)
        self.latency_write_ptr = np.zeros(num_envs, dtype=np.int32)

    def reset(self, env_ids, rng):
        self.latency_steps[env_ids] = rng.integers(0, cfg.max_latency_steps + 1, size=len(env_ids))
        self.latency_write_ptr[env_ids] = 0
        self.latency_ring[:, env_ids, :] = 0.0
        self.v_real[env_ids] = 0.0

    def step(self, action_base_vel):
        """全向量化流水线."""
        ...
```

### 3.11 参数汇总

| 参数 | 类型 | 默认值 |
|------|------|--------|
| `max_lin_vel` | float | 1.5 |
| `max_ang_vel` | float | 1.0 |
| `action_scale_lin` | float | 1.5 |
| `action_scale_ang` | float | 1.0 |
| `tau` | float | 0.05 |
| `max_lin_acc` | float | 1.5 |
| `max_ang_acc` | float | 3.0 |
| `max_latency_steps` | int | 4 |
| `action_noise_scale` | float | 0.05 |
| `enable_latency` | bool | true |
| `enable_noise` | bool | true |
| `enable_wheel_visualization` | bool | true |

---

## 第四节：Observation / Action 空间设计

### 4.1 设计约束

| | Go2_arm | RangerBox |
|---|---|---|
| 策略输出维度 | 18 | **10**（3 base + 6 arm + 1 gripper） |
| backend 执行器数 | 18 | **7**（6 arm + 1 gripper） |
| base 控制 | 腿足运动学 | **BaseVelocityController → freejoint qvel**（SE(2) planar） |
| task 命令 | velocity command | **world-frame EE goal** |
| 步态 | feet_phase | **无** |

### 4.2 Action 空间

```
Box(low=-1.0, high=1.0, shape=(10,), dtype=float32)

action[0]  = vx_des       base-frame 前进
action[1]  = vy_des       base-frame 横向
action[2]  = vyaw_des     base-frame 偏航
action[3:9] = Δcr10_j1~j6  arm delta
action[9]  = grip_cmd     gripper（v1 固定 open）
```

### 4.3 `apply_action` 流程

```python
def apply_action(self, actions, state):
    state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
    state.info["current_actions"] = actions

    # base latency 由 controller 内部处理
    self._base_controller.step(actions[:, 0:3])

    # arm/gripper latency（可选，仅 action[3:10]）
    arm_gripper_action = (state.info["last_actions"][:, 3:10]
                          if self._cfg.control_config.simulate_action_latency
                          else actions[:, 3:10])

    # world goal → armbase 每步动态转换
    armbase_ee_goal = self._world_goal_to_armbase(
        self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world)

    # arm IK + residual
    dq_ik = self.compute_arm_ik_delta(
        armbase_ee_goal, self.get_ee_local_pose()[0],
        self.ee_goal_orn_quat, self.get_ee_local_pose()[1])
    arm_ctrl = (self.get_arm_dof_pos()
                + arm_gripper_action[:, 0:6] * self._cfg.control_config.arm_action_scale
                + self._cfg.ik.gain * dq_ik)

    # gripper fixed open
    grip_ctrl = np.zeros((actions.shape[0], 1), dtype=np.float32)
    ctrl = np.concatenate([arm_ctrl, grip_ctrl], axis=1)
    return np.clip(ctrl, self._ctrl_low, self._ctrl_high)
```

### 4.4 Raw Observation 布局（41 维）

```
linvel(3) + gyro(3) + (-gravity_body)(3) + arm_diff(6) + arm_dof_vel(6) +
ee_local_pos(3) + armbase_ee_goal(3) + ee_error(3) +
gripper_pos(1) + last_actions(10) = 41 维
```

| 段 | 维度 | 来源 |
|---|---|---|
| `linvel` | 3 | `velocimeter`（base-frame 局部线速度） |
| `gyro` | 3 | `gyro` |
| `-gravity_body` | 3 | `framequat` 计算 |
| `arm_diff` | 6 | jointpos arm − default_angles[:6] |
| `arm_dof_vel` | 6 | jointvel arm |
| `ee_local_pos` | 3 | `framepos` (EE in armbase) |
| `armbase_ee_goal` | 3 | world goal → armbase 每步转换 |
| `ee_error` | 3 | `armbase_ee_goal − ee_local_pos` |
| `gripper_pos` | 1 | gripper joint index |
| `last_actions` | 10 | 上一帧 current_actions |

> **不包含 `world_ee_goal`：** 绝对坐标破坏平移不变性。`armbase_ee_goal` + `ee_error` 已完整表达相对关系。

### 4.5 历史堆叠

```python
_RAW_OBS_DIM = 41

def _update_history(self, raw_obs, env_ids=None, *, critic_raw_obs=None):
    """沿特征轴（axis=1）滚动，不沿 env 轴（axis=0）。"""
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
```

- Buffer: `(N, H×D)`。`np.roll(..., -D, axis=1)` 沿特征轴滚动
- Actor 不剔除 linvel（v1 不做 privileged 拆分）

### 4.6 观测噪声

| 参数 | 默认值 | 作用于 |
|------|--------|--------|
| `noise_config.level` | 1.0 | 总开关 |
| `scale_linvel` | 0.1 | linvel |
| `scale_gyro` | 0.2 | gyro |
| `scale_gravity` | 0.05 | gravity_body |
| `scale_joint_angle` | 0.03 | arm_diff, gripper_pos |
| `scale_joint_vel` | 0.5 | arm_dof_vel |
| `scale_ee_pos` | 0.02 | ee_local_pos |
| `scale_ee_goal` | 0.01 | armbase_ee_goal |

### 4.7 与 Go2_arm 对齐

| 对齐项 | 方式 |
|--------|------|
| `compute_arm_ik_delta` / `get_ee_local_pose` / `get_arm_dof_pos/vel` | 直接复用 |
| `_init_action_space` | **重写**：10-dim policy |
| `_update_history` | **重写**：不剔除 linvel，沿 axis=1 滚动 |
| `apply_action` | **重写**：base→controller / arm→IK / gripper→fixed |

---

## 第五节：Reward 设计

### 5.1 任务定义

**Free-space EE reaching**：策略控制 base + arm 移动末端到 world-frame 目标。

**目标采样策略（关键）：**
- **30%** 在机械臂当前可达范围内
- **70%** 超出当前可达、底盘移动后可达

world-frame 目标固定后，每步转 armbase 系供 IK。base 运动直接缩短 world EE→goal 距离。

> 所有公式返回正值，负号只在 YAML `scale` 中。

### 5.2 奖励项清单

#### 主任务（world-frame）

| 名称 | 公式 | scale |
|------|------|-------|
| `ee_distance` | `exp(−‖ee_pos_world − world_ee_goal‖² / σ²)` | **4.0** |
| `ee_distance_l2` | `‖ee_pos_world − world_ee_goal‖²` | **−1.0** |

`σ_ee=0.15`。`ee_pos_world = armbase_pos + quat_rotate(armbase_quat, ee_local)`。

#### 底盘效率

| 名称 | 公式 | scale | 说明 |
|------|------|-------|------|
| `base_vel_xy` | `vx² + vy²` | **−0.05** | |
| `base_vel_z` | — | **0.0** | SE(2) planar |
| `base_vel_yaw` | `vyaw²` | **−0.01** | |

#### 臂运动平滑

| 名称 | 公式 | scale | 说明 |
|------|------|-------|------|
| `arm_dof_vel` | `‖q̇_arm‖²` | **−0.001** | |
| `arm_dof_acc` | `‖(q̇_t − q̇_{t−1}) / dt‖²` | **−1e-6** | 相邻步 velocity 差分 |
| `torques` | — | **0.0** | 不暴露 |

#### 安全

| 名称 | scale | 说明 |
|------|-------|------|
| `base_orientation` | **0.0** | SE(2) planar |
| `base_height` | **0.0** | SE(2) planar |
| `arm_joint_limits` | **−1.0** | margin=0.01 |
| `arm_collision` | **0.0** | 无 touch sensor |

#### 存活/正则

| 名称 | scale |
|------|-------|
| `alive` | **0.3** |
| `action_rate` | **−0.01** |
| `similar_to_default` | **−0.005** (‖q_arm − q_default_arm‖₁, 6 维) |

### 5.3 终止条件

| 条件 | 阈值 | 类型 |
|------|------|------|
| `gravity_x² + gravity_y² > sin²(1.0)` | tilt > 1.0 rad | terminated |
| arm joint 超出硬限位 | hard limits | terminated |
| `steps >= max_episode_steps` | 超时 | truncated |

（SE(2) planar 无 base_height 终止）

### 5.4 RewardContext

```python
@dataclass
class RewardContext:
    info: dict
    linvel, gyro, gravity: np.ndarray     # (N,3) each
    arm_pos, arm_vel: np.ndarray           # (N,6) each
    prev_arm_vel: np.ndarray               # (N,6)
    gripper_pos: np.ndarray                # (N,1)
    num_envs: int
    default_arm_angles: np.ndarray         # (6,)
    armbase_pos_world, armbase_quat_world: np.ndarray
    ee_local_pos, ee_pos_world: np.ndarray
    world_ee_goal, armbase_ee_goal: np.ndarray
    sigma_ee: float
    arm_joint_upper, arm_joint_lower: np.ndarray
    joint_limit_margin: float
```

### 5.5 YAML reward 配置

```yaml
reward_config:
  scales:
    ee_distance: 4.0
    ee_distance_l2: -1.0
    base_vel_xy: -0.05
    base_vel_z: 0.0
    base_vel_yaw: -0.01
    arm_dof_vel: -0.001
    arm_dof_acc: -1.0e-6
    torques: 0.0
    base_orientation: 0.0
    base_height: 0.0
    arm_joint_limits: -1.0
    arm_collision: 0.0
    action_rate: -0.01
    similar_to_default: -0.005
    alive: 0.3
  sigma_ee: 0.15
```

---

## 第六节：Config YAML 设计

### 6.1 `conf/ppo/task/ranger_box_reach/mujoco.yaml`

```yaml
# @package _global_
training:
  task_name: RangerBoxReach
  sim_backend: mujoco
algo:
  num_envs: 256
  max_iterations: 200
  empirical_normalization: true
  obs_groups:
    actor:
      - actor
  policy:
    init_noise_std: 0.5
    actor_hidden_dims: [256, 128, 64]
    critic_hidden_dims: [256, 128, 64]
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 1.0e-3
    num_mini_batches: 4
reward_config:
  scales:
    ee_distance: 4.0
    ee_distance_l2: -1.0
    base_vel_xy: -0.05
    base_vel_z: 0.0
    base_vel_yaw: -0.01
    arm_dof_vel: -0.001
    arm_dof_acc: -1.0e-6
    torques: 0.0
    base_orientation: 0.0
    base_height: 0.0
    arm_joint_limits: -1.0
    arm_collision: 0.0
    action_rate: -0.01
    similar_to_default: -0.005
    alive: 0.3
  sigma_ee: 0.15
env:
  max_episode_seconds: 30.0
  init_state:
    pos: [0.0, 0.0, 0.278]
  control_config:
    arm_action_scale: 0.03
    simulate_action_latency: false
    arm_kp: [100.0, 110.0, 95.0, 50.0, 50.0, 50.0]
    arm_kd: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
    gripper_kp: 500.0
    gripper_kd: 10.0
  noise_config:
    level: 1.0
    scale_linvel: 0.1
    scale_gyro: 0.2
    scale_gravity: 0.05
    scale_joint_angle: 0.03
    scale_joint_vel: 0.5
    scale_ee_pos: 0.02
    scale_ee_goal: 0.01
  ik:
    damping: 0.05
    gain: 1.0
    dq_clip: 0.2
    use_orientation: false
    orientation_mode: target
  goal_ee:
    sphere_l_range: [0.20, 0.50]
    sphere_phi_range: [-1.20, 1.00]
    sphere_theta_range: [-2.00, 2.00]
    reachable_fraction: 0.30
    extended_l_range: [0.50, 1.20]
    extended_fraction: 0.70
    traj_time_range: [1.0, 3.0]
    hold_time_range: [0.5, 2.0]
    collision_upper_limits: [0.30, 0.15, -0.10]
    collision_lower_limits: [-0.20, -0.15, -0.50]
    underground_limit: -0.50
    num_collision_check_samples: 10
    num_resample_attempts: 10
    default_orn_roll: 1.5708
    arm_induced_pitch: 0.0
    delta_orn_r: [0.0, 0.0]
    delta_orn_p: [0.0, 0.0]
    delta_orn_y: [0.0, 0.0]
    init_ee_cart: [0.30, 0.0, 0.30]
  base_velocity_controller:
    max_lin_vel: 1.5
    max_ang_vel: 1.0
    action_scale_lin: 1.5
    action_scale_ang: 1.0
    tau: 0.05
    max_lin_acc: 1.5
    max_ang_acc: 3.0
    max_latency_steps: 4
    action_noise_scale: 0.05
    enable_latency: true
    enable_noise: true
    enable_wheel_visualization: true
  domain_rand:
    randomize_kp: true
    kp_multiplier_range: [0.9, 1.1]
    randomize_kd: true
    kd_multiplier_range: [0.9, 1.1]
    randomize_body_mass: true
    body_mass_multiplier_range: [0.9, 1.1]
    random_com: true
    com_offset_x: [-0.03, 0.03]
    randomize_ground_friction: false
    randomize_dof_armature: true
    dof_armature_multiplier_range: [0.8, 1.2]
    push_robots: false
  history:
    num_actor_history: 1
    num_critic_history: 1
  arm_stage:
    freeze_arm_joints: false
    disable_ee_goal_trajectory: false
    fixed_ee_goal_cart: [0.30, 0.0, 0.30]
```

### 6.2 YAML → Dataclass 映射

| YAML 路径 | Dataclass |
|-----------|-----------|
| `reward_config.*` | `RangerBoxRewardConfig`（不是 `reward`） |
| `env.control_config.*` | `RangerBoxControlConfig` |
| `env.sensor.*` | `RangerBoxSensor`（继承 `Go2ArmSensor`） |
| `env.noise_config.*` | `RangerBoxNoiseConfig` |
| `env.goal_ee.*` | `EEGoalConfig`（扩展字段） |
| `env.base_velocity_controller.*` | `BaseVelocityControllerConfig` |
| `env.domain_rand.*` | `RangerBoxDomainRandConfig` |
| `env.history.*` / `env.ik.*` / `env.arm_stage.*` | 复用 |

### 6.3 关键差异 vs Go2_arm

| 字段 | Go2_arm | RangerBox |
|------|---------|-----------|
| `reward_config` | Go2 arm | RangerBox（训练适配器注入） |
| `env.commands` | velocity ranges | **无** |
| `push_robots` | true | **false** |
| `randomize_ground_friction` | true | **false** |
| `leg_kp/kd` | 60/2 | **无** |
| `base_vel_z/base_orientation/base_height` | 非零 | **0.0** |

---

## 第七节：Env 类层次与文件组织

### 7.1 类层次

```
NpEnv → LocomotionBaseEnv → Go2ArmBaseEnv → RangerBoxReachEnv
```

### 7.2 方法重写矩阵

| 方法 | 父类 | RangerBox |
|------|------|-----------|
| `_init_action_space` | backend ctrl_range | `Box(-1,1,(10,))` |
| `_init_buffers` | qpos tail | 显式常量 7 维 |
| `apply_action` | 全局 action_scale | base→ctrl / arm→IK / gripper→fixed |
| `update_state` | gait+command+EE | world goal→armbase + 受控 DOF |
| `_compute_raw_obs` | 79 维 | 41 维 |
| `_update_history` | 去 linvel | 全 41 维，axis=1 滚动 |
| `_compute_reward` | 15 项 | 11 项（含 5 项 0.0） |
| `obs_groups_spec` | H*76/H*79 | H*41/H*41 |
| termination | gravity_z + base_h | tilt_sq + arm limits |

### 7.3 Env 构造流程

```python
class RangerBoxReachEnv(Go2ArmBaseEnv):
    def __init__(self, cfg, num_envs=1, backend_type="mujoco"):
        # 1. resolve scene
        scene = _resolve_ranger_box_scene(cfg)

        # 2. position gains as dict
        position_actuator_gains = build_ranger_box_position_gains(cfg.control_config)
        # → {"kp": np.array(7,), "kd": np.array(7,)}

        # 3. create_backend (correct signature)
        backend = create_backend(
            backend_type, scene, num_envs, cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            position_actuator_gains=position_actuator_gains,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
            motrix_max_iterations=getattr(cfg, 'motrix_max_iterations', None),
        )

        # 4. parent init (cfg, backend, num_envs order)
        super().__init__(cfg, backend, num_envs)

        # 5. ctrl bounds from backend (7 dims)
        ctrl_range = self._backend.get_actuator_ctrl_range()
        self._ctrl_low = np.asarray(ctrl_range[:, 0])
        self._ctrl_high = np.asarray(ctrl_range[:, 1])

        # 6. gripper DOF indices
        self._gripper_dof_pos_idx = self._backend.get_joint_dof_pos_indices(
            [cfg.asset.gripper_joint_name])
        self._gripper_dof_vel_idx = self._backend.get_joint_dof_vel_indices(
            [cfg.asset.gripper_joint_name])

        # 7. controller
        self._base_controller = BaseVelocityController(
            cfg.base_velocity_controller, cfg.ctrl_dt, backend, cfg.asset, num_envs)

        # 8. world goal + armbase cache
        self.world_ee_goal = np.zeros((num_envs, 3))
        self.armbase_pos_world = np.zeros((num_envs, 3))
        self.armbase_quat_world = np.zeros((num_envs, 4))

        # 9. prev arm vel for acc computation
        self._prev_arm_vel = np.zeros((num_envs, 6))

        # 10. init goals, history, reward fns
        self._init_ee_goals()
        self._init_history_buffers()
        self._init_reward_functions()


def build_ranger_box_position_gains(cc: RangerBoxControlConfig) -> dict[str, np.ndarray]:
    """{'kp': np.array(7,), 'kd': np.array(7,)} — 6 arm + 1 gripper."""
    return {
        "kp": np.concatenate([np.asarray(cc.arm_kp), [cc.gripper_kp]]),
        "kd": np.concatenate([np.asarray(cc.arm_kd), [cc.gripper_kd]]),
    }
```

### 7.4 `update_state` 流程

```python
def update_state(self, state):
    self._update_ee_goal_trajectory()
    linvel = self.get_local_linvel()
    gyro = self.get_gyro()
    gravity = self._get_projected_gravity()  # via cfg.sensor.framequat
    ee_local_pos, ee_local_quat = self.get_ee_local_pose()

    # controlled DOF only
    arm_pos = self.get_arm_dof_pos()
    arm_vel = self.get_arm_dof_vel()
    gripper_pos = self.get_gripper_dof_pos()

    # armbase pose (via cfg.sensor, no hardcoded strings)
    armbase_pos_world = self._backend.get_sensor_data(self._cfg.sensor.armbase_world_pos)
    armbase_quat_world = self._backend.get_sensor_data(self._cfg.sensor.arm_ref_world_quat)

    # world goal → armbase
    armbase_ee_goal = self._world_goal_to_armbase(
        self.world_ee_goal, armbase_pos_world, armbase_quat_world)
    ee_pos_world = armbase_pos_world + quat_rotate(armbase_quat_world, ee_local_pos)

    # termination
    tilt_sq = gravity[:, 0]**2 + gravity[:, 1]**2
    terminated = (tilt_sq > np.sin(1.0)**2) | self._arm_joint_hard_limits_violated(arm_pos)

    # arm acc
    arm_acc = (arm_vel - self._prev_arm_vel) / self._cfg.ctrl_dt
    self._prev_arm_vel = arm_vel.copy()

    reward = self._compute_reward(...)
    obs = self._compute_obs(...)
    return state.replace(obs=obs, reward=reward, terminated=terminated)
```

### 7.5 Projected Gravity

```python
def _get_projected_gravity(self):
    quat = self._backend.get_sensor_data(self._cfg.sensor.framequat)  # via cfg
    R_wb = np_matrix_from_quat(quat)
    return np.einsum("nij,j->ni", np.swapaxes(R_wb, 1, 2), [0, 0, -1])
```

### 7.6 DR Provider

```python
class RangerBoxReachDRProvider(LocomotionDRProvider):
    def __init__(self, backend, cfg):
        base_kp, base_kd = backend.get_actuator_gains()
        self._base_kp = base_kp       # (7,)
        self._base_kd = base_kd       # (7,)
        self._base_body_mass = ...      # cached
        self._base_dof_armature = ...

    def _get_base_actuator_gains(self, env):
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(self, env):
        return self._base_body_mass, None, None, self._base_dof_armature

    def _sample_commands(self, env, num_reset):
        return np.zeros((num_reset, 3))  # (num_reset, 3), NOT dict

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        env._arm_goal_timer[env_ids] = 0
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._prev_arm_vel[env_ids] = 0.0
        env._base_controller.reset(env_ids, np.random)
        return plan

    def _compute_reset_obs(self, env, env_ids, info_updates, ...):
        # backend.set_state() already done — safe to generate world goal
        env.reset_ee_goals(env_ids)
        ...
```

**关键：** `reset_ee_goals()` 在 `_compute_reset_obs()` 中调用（`set_state` 之后），不在 `build_reset_plan()` 中。

### 7.7 Dataclass 汇总

```python
# Asset — inherits go2_arm.Asset, overrides defaults
@dataclass
class RangerBoxAsset(Asset):
    base_name: str = "base"; ground: str = "floor"
    ee_site_name: str = "right_center"; ee_body_name: str = "cr10_Link6"
    arm_joint_names: tuple[str, ...] = ("cr10_joint1", ..., "cr10_joint6")
    gripper_joint_name: str = "gripper_finger1_joint"
    steering_joint_names: tuple[str, ...] = (...)
    wheel_joint_names: tuple[str, ...] = (...)
    wheel_positions: tuple[tuple[float, float], ...] = ((0.445,-0.28), ...)
    wheel_radius: float = 0.152

# Sensor — inherits Go2ArmSensor, overrides field values only
@dataclass
class RangerBoxSensor(Go2ArmSensor):
    local_linvel: str = "imu-velocimeter"
    gyro: str = "imu-gyro"; framequat: str = "imu-framequat"
    ee_local_pos: str = "endpoint-framepos"; ee_local_quat: str = "endpoint-framequat"
    arm_ref_world_quat: str = "armbasepoint-framequat"
    armbase_world_pos: str = "armbasepoint-framepos"

# Control
@dataclass
class RangerBoxControlConfig(ControlConfig):
    arm_action_scale: float = 0.03
    arm_kp: tuple = (100., 110., 95., 50., 50., 50.)
    arm_kd: tuple = (3.5, 3.8, 2.5, 1.5, 1.5, 1.5)
    gripper_kp: float = 500.0; gripper_kd: float = 10.0

# DomainRand — explicit kp/kd fields
@dataclass
class RangerBoxDomainRandConfig(DomainRandConfig):
    randomize_ground_friction: bool = False
    randomize_kp: bool = True; kp_multiplier_range = (0.9, 1.1)
    randomize_kd: bool = True; kd_multiplier_range = (0.9, 1.1)
    push_robots: bool = False

# Noise
@dataclass
class RangerBoxNoiseConfig(NoiseConfig):
    scale_ee_goal: float = 0.01

# Reward
@dataclass
class RangerBoxRewardConfig:
    scales: dict = field(default_factory=lambda: {
        "ee_distance": 4.0, "ee_distance_l2": -1.0,
        "base_vel_xy": -0.05, "base_vel_z": 0.0, "base_vel_yaw": -0.01,
        "arm_dof_vel": -0.001, "arm_dof_acc": -1e-6,
        "torques": 0.0, "base_orientation": 0.0, "base_height": 0.0,
        "arm_joint_limits": -1.0, "arm_collision": 0.0,
        "action_rate": -0.01, "similar_to_default": -0.005, "alive": 0.3,
    })
    sigma_ee: float = 0.15

# EnvCfg
@registry.envcfg("RangerBoxReach")
@dataclass
class RangerBoxReachCfg(Go2ArmBaseCfg):
    scene: SceneCfg = field(default_factory=_default_ranger_box_scene)
    model_file: str = field(default_factory=_default_ranger_box_model_file)
    max_episode_seconds: float = 30.0
    init_state: InitState = field(default_factory=InitState)
    goal_ee: EEGoalConfig = field(default_factory=EEGoalConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    arm_stage: ArmStageConfig = field(default_factory=ArmStageConfig)
    reward_config: RangerBoxRewardConfig = field(default_factory=RangerBoxRewardConfig)
    domain_rand: RangerBoxDomainRandConfig = field(default_factory=RangerBoxDomainRandConfig)
    base_velocity_controller: BaseVelocityControllerConfig = field(
        default_factory=BaseVelocityControllerConfig)
    # NOTE: 训练适配器注入 reward_config，不是 reward
```

### 7.8 注册集成

`locomotion/__init__.py`:
```python
__unilab_registry_modules__ = (
    # ...
    "unilab.envs.locomotion.ranger_box",   # 包级别
)
```

`ranger_box/__init__.py` 导入 `RangerBoxReachEnv` 触发注册。

---

## 第八节：测试计划

| 测试 | 验证 |
|------|------|
| XML 维度 | `nq=29`, `nu=7` |
| Hydra compose | `task=ranger_box_reach/mujoco` 不抛异常 |
| Registry 创建 | `gym.make("RangerBoxReach-mujoco", num_envs=1)` |
| Action→Ctrl | `(4,10)` → `(4,7)` |
| Mixed latency | 4 envs 不同 latency_steps |
| History H>1 | `obs["obs"].shape[1]==3*41` |
| Partial reset | env_ids 后 history 清零 |
| World goal 固定 | `world_ee_goal` 不变，`armbase_ee_goal` 随 base 变 |
| 远目标 | extended 目标仅 arm IK 不可达 |
| DR reset | randomize_kp=true 不崩溃 |
| SE(2) lock | z/roll/pitch 多步恒定 |

---

## 附录 A：CR10 关节硬限位

| 关节 | XML range (rad) | kp |
|------|-----------------|-----|
| cr10_joint1 | [-3.92, 0.94] | 100 |
| cr10_joint2 | [-1.57, 1.57] | 110 |
| cr10_joint3 | [-2.86, 2.86] | 95 |
| cr10_joint4 | [-3.14, 3.14] | 50 |
| cr10_joint5 | [-3.14, 3.14] | 50 |
| cr10_joint6 | [-3.14, 3.14] | 50 |
| gripper_finger1 | [0, 0.65] | 500 |

## 附录 B：底盘转向/车轮关节

| 关节 | steering | wheel | (x, y) |
|------|----------|-------|--------|
| FR | `fr_steering_joint` | `fr_wheel_joint` | (+0.445, -0.28) |
| FL | `fl_steering_wheel_joint` | `fl_wheel_joint` | (+0.445, +0.28) |
| RL | `rl_steering_wheel_joint` | `rl_wheel_joint` | (-0.445, +0.28) |
| RR | `rr_steering_wheel_joint` | `rr_wheel_joint` | (-0.445, -0.28) |

## 附录 C：关键风险

| 风险 | 缓解 |
|------|------|
| backend planar setter 竞态 | 只写 planar 分量，保留非受控分量 |
| `reward_config` vs `reward` | `RangerBoxReachCfg.reward_config` 匹配训练适配器 |
| `_sample_commands` 返回值类型 | 返回 `(num_reset,3)` 数组 |
| DR kp/kd 无缓存 | Provider `__init__` 调 `backend.get_actuator_gains()` |
| world goal reset 读旧位姿 | `reset_ee_goals()` 在 `_compute_reset_obs` 中 |
| `_update_history` 沿 env 轴滚动 | 沿 axis=1，写入 `[:, -D:]` |
| Sensor 字段不匹配父类 | `RangerBoxSensor` 继承 `Go2ArmSensor` |
