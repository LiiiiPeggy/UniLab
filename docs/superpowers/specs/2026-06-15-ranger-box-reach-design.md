# RangerBoxReach — 移动操作 EE 到达 设计文档

日期：2026-06-29 | 状态：v2 审阅修正完成，待审阅 | 任务：RangerboxCR10Lidar 移动底盘 + 臂 EE 目标到达

---

## 第一节：文件变更清单

### 新增代码/配置文件（4 个）

| 文件 | 内容 |
|------|------|
| `src/unilab/envs/locomotion/ranger_box/__init__.py` | 包文件，暴露 `RangerBoxReachEnv` |
| `src/unilab/envs/locomotion/ranger_box/reach_env.py` | Env + Dataclass cfg、DR Provider、reward |
| `src/unilab/envs/locomotion/ranger_box/base_velocity_controller.py` | A+ 方案核心：一阶滤波 + 延迟 + 限幅 + 噪声 + 轮子 IK |
| `conf/ppo/task/ranger_box_reach/mujoco.yaml` | PPO × MuJoCo owner YAML |

### 新增资产文件

| 目录 | 内容 |
|------|------|
| `src/unilab/assets/robots/ranger_box/` | `robot.xml`、`scene_flat.xml`、`meshes/*.obj` (26 个) |

### 需修改现有文件（3 个）

| 文件 | 修改内容 | 风险 |
|------|---------|------|
| `src/unilab/envs/locomotion/__init__.py` | `__unilab_registry_modules__` 加 `"unilab.envs.locomotion.ranger_box"`（包级别，对齐 Go2_arm 模式） | 低——纯注册 |
| `src/unilab/base/backend/base.py` | 新增 `set_root_planar_velocity`、`set_joint_qpos`、`set_joint_qvel` 三个方法（非抽象，默认抛 `NotImplementedError`） | 中——SimBackend 接口扩展，影响所有后端子类 |
| `src/unilab/base/backend/mujoco/backend.py` | 实现上述三个方法的 MuJoCo 版本 | 中——只写 freejoint qvel 的 vx/vy/wz 分量，保留 vz/wx/wy |

### 不修改的部分

| 文件 | 原因 |
|------|------|
| `scripts/train_rsl_rl.py` | Hydra config 驱动，不感知具体 task |
| `src/unilab/training/run.py` | 通用训练流程，不过载 task 逻辑 |
| `src/unilab/base/registry.py` | `ensure_registries()` 自动发现新模块 |
| `src/unilab/envs/locomotion/go2_arm/` | 仅作为父类被 RangerBoxReachEnv 继承 |

---

## 第二节：robot.xml 适配设计

### 源 → 目标变更

| 问题 | 现状（foropenpi） | 目标（UniLab） |
|------|-------------------|----------------|
| 基座自由度 | 无 freejoint | 加 `<freejoint/>` |
| 臂执行器 | `<motor>`（力矩模式） | `<position>`（位置控制） |
| 传感器 | 仅 `force_ee` / `torque_ee` | RL 最小闭环传感器（见下表） |
| 夹爪执行器 | `<position>` finger1 + equality | **保留不变** |
| 转向/车轮 | 无 actuator | **保留无 actuator** |
| 网格路径 | `meshes/xxx.obj` | 复制到 `src/unilab/assets/robots/ranger_box/meshes/` |

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

- 6 arm + 1 gripper = **7 个 position actuators**（对齐 Go2_arm airbot_play 模式）
- `kp` 运行时通过 `position_actuator_gains` 注入，XML 中的值为默认 fallback
- 转向/车轮关节**不在 actuator 块中注册**

### 新增 Backend Contract 方法

| 方法 | 声明位置 | 实现位置 | 用途 |
|------|---------|---------|------|
| `set_root_planar_velocity(lin_vel_xy, yaw_rate, preserve_uncontrolled=True)` | `base/backend/base.py` | `mujoco/backend.py` | 只写 freejoint qvel 的 vx/vy/wz，保留 vz/wx/wy |
| `set_joint_qpos(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qpos（轮子可视化） |
| `set_joint_qvel(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qvel（轮子可视化） |

三个方法在 `SimBackend` 默认抛 `NotImplementedError`（非抽象方法），仅在 `MuJoCoBackend` 实现。
`set_root_planar_velocity` 只覆盖受控平面分量，避免清零 vz/wx/wy 与 base_height reward、push DR、倾覆终止冲突。

### 新增传感器（最小 RL 闭环）

| 类别 | sensor | 数量 | 用途 |
|------|--------|------|------|
| IMU（`imu` site on base body） | `gyro`、`framequat`、`velocimeter`、`framezaxis` | 4 | base 角速度、姿态、base-frame 局部线速度、Z 轴方向 |
| 臂关节 | `jointpos` + `jointvel` | 6+6 | arm 本体感知 |
| 底盘关节 | `jointpos` + `jointvel` | 8+8 | 转向/车轮状态（观测用） |
| 末端位姿（相对 armbase） | `framepos`、`framequat` | 2 | EE goal tracking（framelinvel 不在 endpoint site 用） |
| 夹爪 | `jointpos` finger1 | 1 | gripper state |

> **坐标系约定：** MuJoCo `<velocimeter>` 输出 body-local 线速度，`<framelinvel>` 默认 world-frame。Go2_arm XML 也用 `velocimeter name="local_linvel"` 获取 base-frame 线速度。RangerBox 统一使用 `velocimeter`，所有 sensor name 通过 `RangerBoxSensor` dataclass 管理，代码禁止硬编码字符串。
>
> **v1 暂不加入**（可通过 config 后续打开）：`jointactuatorfrc`、`touch`、`accelerometer`。accelerometer 含线加速度分量，gravity 方向改用 `framequat` 计算。**轮子 geom 需设置 `contype="0" conaffinity="0"`** 避免 wheel 可视化 qpos/qvel 写入影响物理接触。

### 轮子可视化：接口合规设计

```
每步流程:
  策略输出 action[0:3] = (vx_base, vy_base, vyaw_base)   ← base-frame

  1. scale → clip → latency → accel limit → first-order → noise → final clip
     （详见第三节 BaseVelocityController）

  2. 轮子 IK（base-frame 坐标，实际 wheel positions 来自 XML）:
       FR: ( 0.445, -0.28)   FL: ( 0.445,  0.28)
       RR: (-0.445, -0.28)   RL: (-0.445,  0.28)
       wheel_radius = 0.152 m（6 英寸）

       for each wheel at (x_i, y_i):
         vx_i = vx_base - vyaw_base * y_i
         vy_i = vy_base + vyaw_base * x_i
         steer_angle = wrap_to_pi(atan2(vy_i, vx_i))
         wheel_omega  = sqrt(vx_i² + vy_i²) / wheel_radius

  3. 通过 backend contract 写入（纯可视化，不影响 freejoint 物理）:
       backend.set_joint_qpos(steering_names, steer_values)
       backend.set_joint_qvel(wheel_names, wheel_vel_values)

  4. base-frame → world-frame 转换:
       R_wb = quat_to_matrix(base_quat)
       v_world = R_wb @ [vx_base, vy_base, 0]
       w_world = R_wb @ [0, 0, vyaw_base]

  5. 通过 backend contract 写入:
       backend.set_root_planar_velocity(
           v_world[:, 0:2], w_world[:, 2:3],
           preserve_uncontrolled=True,
       )   ← 物理生效（只写 vx/vy/wz，保留 vz/wx/wy）
```

- 轮子 IK 输入是 **base-frame**（策略直接输出），`set_root_planar_velocity` 写入 **world-frame**
- `preserve_uncontrolled=True` 确保 vz（底盘高度）、wx/wy（roll/pitch 角速度）不被清零，与 base_height reward、push DR、倾覆终止条件兼容
- 轮子 geom 需设置 `contype="0" conaffinity="0"`，避免 wheel qpos/qvel 的纯可视化写入干扰物理碰撞

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
- `ctrl` = 6(arm position) + 1(gripper position) = **7** ← 与 `<actuator>` 块严格一致

### 传感器命名约定（通过 `RangerBoxSensor` dataclass 统一管理）

| XML sensor name | `cfg.sensor.<field>` | 用途 |
|-----------------|----------------------|------|
| `imu-gyro` | `sensor.gyro` | base 角速度 |
| `imu-velocimeter` | `sensor.velocimeter` | base-frame 局部线速度 |
| `imu-framequat` | `sensor.framequat` | base 姿态 → 计算 projected gravity |
| `armbasepoint-framepos` | `sensor.armbase_framepos` | arm base 在 world 系位置 |
| `armbasepoint-framequat` | `sensor.armbase_framequat` | arm base 在 world 系姿态 |
| `endpoint-framepos` | `sensor.ee_framepos` | EE 在 armbase 系位置 |
| `endpoint-framequat` | `sensor.ee_framequat` | EE 在 armbase 系姿态 |
| `arm-jointpos` / `arm-jointvel` | `sensor.arm_jointpos` / `arm_jointvel` | arm 关节状态 |
| `steering-jointpos/vel` | `sensor.steering_jointpos/vel` | 转向关节（观测用） |
| `wheel-jointpos/vel` | `sensor.wheel_jointpos/vel` | 车轮关节（观测用） |
| `gripper-jointpos` | `sensor.gripper_jointpos` | 夹爪位置 |

**代码只允许通过 `self._cfg.sensor.xxx` 获取 sensor name，禁止硬编码字符串。**

---

## 第三节：BaseVelocityController 设计

### 3.1 为什么是"方案 A+"？

- MuJoCo freejoint 提供 6-DOF 自由体运动学，无需求解轮地摩擦
- 训练效率远高于全接触动力学
- 对自由空间 EE tracking 任务，轮地接触细节不提供有意义的迁移信号
- Domain randomization 在控制器层实现（延迟、限幅、噪声）

### 3.2 控制器架构（全向量化）

控制器内部所有状态保持为 `(num_envs, ...)` 张量。不使用 Python `deque` per env —— 用 NumPy ring buffer `(max_latency+1, N, 3)` 实现延迟。

```
action[:, 0:3] = (vx_des, vy_des, vyaw_des)    ← 策略输出，base-frame，(N,3)

     │
     ▼
┌──────────────────────────────────────────────────────┐
│             BaseVelocityController (batched)           │
│                                                      │
│  1. action scale   ──► [-1,1] → m/s, rad/s  (N,3)   │
│  2. command clip   ──► 限幅到安全范围      (N,3)     │
│  3. latency ring   ──► 延迟缓冲 (L,N,3) + env-wise   │
│  4. accel limit    ──► 加速度限幅          (N,3)     │
│  5. first-order    ──► 一阶平滑响应       (N,3)     │
│  6. noise          ──► domain rand 加性噪声 (N,3)    │
│  7. final clip     ──► 最终安全限幅       (N,3)     │
│  8. wheel IK (viz) ──► wrap_to_pi + 轮子角 (N,4)    │
│  9. world-frame     ──► quat 旋转到世界    (N,3)     │
│                                                      │
│  输出: set_root_planar_velocity(N,3)                  │
│  输出: set_joint_qpos/qvel(steer, wheel) (N,4 each)  │
└──────────────────────────────────────────────────────┘
     │
     ▼
  backend.step() ──► MuJoCo 前向动力学
```

### 3.3 Step 1-2：Scale + Clip（向量化）

```python
# action_base_vel: (N, 3) in [-1, 1]
# scale_xy:       (2,) = [action_scale_lin, action_scale_lin]
# scale_yaw:      scalar = action_scale_ang
v_cmd = np.empty((N, 3), dtype=np.float32)
v_cmd[:, 0:2] = np.clip(action_base_vel[:, 0:2] * scale_xy, -max_lin_vel, max_lin_vel)
v_cmd[:, 2]   = np.clip(action_base_vel[:, 2]   * scale_yaw, -max_ang_vel, max_ang_vel)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_lin_vel` | 1.5 m/s | Ranger 最大线速度 |
| `max_ang_vel` | 1.0 rad/s | 约 57°/s |
| `action_scale_lin` | 1.5 | [-1,1] → [-1.5, 1.5] |
| `action_scale_ang` | 1.0 | [-1,1] → [-1, 1] |

### 3.4 Step 3：延迟环（Latency Ring Buffer）

不使用 Python `deque` per env（256+ env 时成为热点）。用单个 NumPy ring buffer：

```python
# 初始化
self.latency_ring = np.zeros((max_latency_steps + 1, N, 3), dtype=np.float32)
self.latency_steps = np.zeros(N, dtype=np.int32)     # 每个 env 的延迟步数
self.latency_write_ptr = np.zeros(N, dtype=np.int32)  # 每个 env 的 ring write head

# reset（per env_ids）
new_steps = rng.integers(0, max_latency_steps + 1, size=len(env_ids))
self.latency_steps[env_ids] = new_steps
self.latency_write_ptr[env_ids] = 0
self.latency_ring[:, env_ids, :] = 0.0

# step
if self.enable_latency:
    L = self.latency_ring.shape[0]
    # 写入当前 cmd 到 ring
    wp = self.latency_write_ptr  # (N,)
    self.latency_ring[wp, np.arange(N), :] = v_cmd
    # 读取延迟后的 cmd：ring index = (wp - latency_steps) mod L
    rp = (wp - self.latency_steps) % L
    v_cmd_delayed = self.latency_ring[rp, np.arange(N), :]  # (N, 3)
    # 推进 write pointer
    self.latency_write_ptr = (wp + 1) % L
else:
    v_cmd_delayed = v_cmd  # bypass
```

- `max_latency_steps` 默认 4（80ms @ 50Hz），受 `enable_latency` 开关控制
- 单次 step 只做数组索引 + 取模，无 Python loop

### 3.5 Step 4：加速度限幅（向量化）

```python
acc_limits_xy = np.full((N, 2), max_lin_acc * dt, dtype=np.float32)
acc_limit_yaw = np.full((N, 1), max_ang_acc * dt, dtype=np.float32)
dv = v_cmd_delayed - v_real                             # (N, 3)
dv[:, 0:2] = np.clip(dv[:, 0:2], -acc_limits_xy, acc_limits_xy)
dv[:, 2]   = np.clip(dv[:, 2],   -acc_limit_yaw.flatten(), acc_limit_yaw.flatten())
v_target = v_real + dv
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_lin_acc` | 1.5 m/s² | 线加速度硬限幅 |
| `max_ang_acc` | 3.0 rad/s² | 角加速度硬限幅 |

### 3.6 Step 5：一阶速度响应（向量化）

```python
alpha = dt / (tau + dt)                        # 稳定标量
v_real = v_real + alpha * (v_target - v_real)  # (N, 3)
```

- `tau` 默认 0.05s（50ms）

### 3.7 Step 6-7：噪声 + Final Clip（向量化）

```python
if self.enable_noise:
    noise_xy = noise_scale * max_lin_vel * rng.randn(N, 2)
    noise_yaw = noise_scale * max_ang_vel * rng.randn(N, 1)
    noise = np.concatenate([noise_xy, noise_yaw], axis=1)     # (N, 3)
    v_real = v_real + noise

v_real[:, 0:2] = np.clip(v_real[:, 0:2], -max_lin_vel, max_lin_vel)
v_real[:, 2]   = np.clip(v_real[:, 2],   -max_ang_vel, max_ang_vel)
```

噪声作用在一阶滤波**之后**（否则滤波会把噪声平滑掉），final clip 兜底。`enable_noise` 开关允许 eval 时关闭。

### 3.8 Step 8：轮子可视化 + 角度 Wrap（向量化）

```python
# wheel_positions: (4, 2) from cfg.asset — base-frame
# v_real: (N, 3)  base-frame linear vx,vy + angular vyaw
if self.enable_wheel_visualization:
    x = wheel_positions[:, 0]  # (4,)
    y = wheel_positions[:, 1]  # (4,)
    vx_i = v_real[:, 0:1] - v_real[:, 2:3] * y[None, :]  # (N, 4)
    vy_i = v_real[:, 1:2] + v_real[:, 2:3] * x[None, :]  # (N, 4)
    steer = np.arctan2(vy_i, vx_i)                         # (N, 4)
    omega = np.sqrt(vx_i**2 + vy_i**2) / wheel_radius     # (N, 4)
    # write via backend contract（纯可视化）
    backend.set_joint_qpos(steering_names, steer)
    backend.set_joint_qvel(wheel_names, omega)
```

受 `enable_wheel_visualization` 开关控制。

### 3.9 Step 9：World-Frame 转换（向量化）

```python
# base_quat: (N, 4) wxyz
v_body = np.concatenate([v_real[:, 0:2], np.zeros((N, 1))], axis=1)  # (N, 3) pad z=0
w_body = np.concatenate([np.zeros((N, 2)), v_real[:, 2:3]], axis=1)  # (N, 3) only yaw

v_world = quat_rotate(base_quat, v_body)   # (N, 3)
w_world = quat_rotate(base_quat, w_body)   # (N, 3)

backend.set_root_planar_velocity(v_world[:, 0:2], w_world[:, 2:3],
                                  preserve_uncontrolled=True)
```

### 3.10 控制器生命周期

```python
class BaseVelocityController:
    def __init__(self, cfg: BaseVelocityControllerConfig, dt: float,
                 backend: SimBackend, asset_cfg: RangerBoxAsset,
                 num_envs: int):
        self.v_real = np.zeros((num_envs, 3), dtype=np.float32)
        self.latency_ring = np.zeros((cfg.max_latency_steps + 1, num_envs, 3),
                                      dtype=np.float32)
        ...

    def reset(self, env_ids: np.ndarray, rng: np.random.Generator):
        """重置各 env 的 latency_steps、ring buffer 和 v_real"""
        n = len(env_ids)
        self.latency_steps[env_ids] = rng.integers(0, self.cfg.max_latency_steps + 1, size=n)
        self.latency_write_ptr[env_ids] = 0
        self.latency_ring[:, env_ids, :] = 0.0
        self.v_real[env_ids] = 0.0

    def step(self, action_base_vel: np.ndarray) -> None:
        """全向量化：scale→clip→latency→accel→filter→noise→clip→IK→world"""
        ...  # 如上各步
```

### 3.11 与 Env 的关系

`BaseVelocityController` 是 env 的内部组件，在 `RangerBoxReachEnv.__init__` 中构造。`apply_action()` 中 base 延迟由 controller 独立处理，不经过 arm/gripper 的 `simulate_action_latency`：

```python
def apply_action(self, actions, state):
    # base: 向量化传入 (N,3)，controller 内部完成全流水线
    self._base_controller.step(actions[:, 0:3])

    # arm/gripper latency（独立于 base）
    arm_gripper_action = (state.info["last_actions"][:, 3:10]
                          if self._cfg.control_config.simulate_action_latency
                          else actions[:, 3:10])

    armbase_ee_goal = self._world_goal_to_armbase(
        self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world)
    dq_ik = self.compute_arm_ik_delta(
        armbase_ee_goal, self.get_ee_local_pose()[0],
        self.ee_goal_orn_quat, self.get_ee_local_pose()[1])
    arm_ctrl = (self.get_arm_dof_pos()
                + arm_gripper_action[:, 0:6] * self._cfg.control_config.arm_action_scale
                + ik_gain * dq_ik)
    grip_ctrl = (arm_gripper_action[:, 6] + 1.0) / 2.0 * 0.65

    ctrl = np.concatenate([arm_ctrl, grip_ctrl[:, None]], axis=1)
    return np.clip(ctrl, self._ctrl_low, self._ctrl_high)
```

### 3.12 参数汇总（YAML 路径：`env.base_velocity_controller`）

| 参数 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `max_lin_vel` | float | 1.5 | Step 2 clip + Step 7 final clip |
| `max_ang_vel` | float | 1.0 | Step 2 clip + Step 7 final clip |
| `action_scale_lin` | float | 1.5 | Step 1 |
| `action_scale_ang` | float | 1.0 | Step 1 |
| `tau` | float | 0.05 | Step 5 一阶响应时间常数 |
| `max_lin_acc` | float | 1.5 | Step 4 线加速度限幅 |
| `max_ang_acc` | float | 3.0 | Step 4 角加速度限幅 |
| `max_latency_steps` | int | 4 | Step 3 延迟 buffer 最大长度 |
| `action_noise_scale` | float | 0.05 | Step 6 加性噪声标准差 |
| `enable_latency` | bool | true | 开关 |
| `enable_noise` | bool | true | 开关 |
| `enable_wheel_visualization` | bool | true | 开关 |

---

## 第四节：Observation / Action 空间设计

### 4.1 设计约束

| | Go2_arm | RangerBox |
|---|---|---|
| 策略输出维度 | 18（12 leg + 6 arm） | **10**（3 base vel + 6 arm Δ + 1 gripper） |
| backend 执行器数 | 18 position actuators | **7**（6 arm + 1 gripper） |
| base 控制方式 | 腿足运动学 | **BaseVelocityController → freejoint qvel** |
| 外部 task 命令 | velocity command [vx,vy,vyaw] | **EE goal position** |
| 步态 | feet_phase 4 维 | **无** |
| base 延迟处理 | `simulate_action_latency`（全局） | **controller 内部独立** |

核心约束：策略输出 10 维，但 `backend.step()` 只接收 7 维（6 arm + 1 gripper），base 速度通过 `BaseVelocityController` 直接写入 freejoint qvel。必须重写 `_init_action_space`。

### 4.2 Action 空间

```
Action Space: Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

action[0]  = vx_des      ← base-frame 前进速度
action[1]  = vy_des      ← base-frame 横向速度
action[2]  = vyaw_des    ← base-frame 偏航角速度
action[3]  = Δcr10_j1    ← 臂关节 delta
action[4]  = Δcr10_j2
action[5]  = Δcr10_j3
action[6]  = Δcr10_j4
action[7]  = Δcr10_j5
action[8]  = Δcr10_j6
action[9]  = grip_cmd    ← 夹爪归一化命令
```

### 4.3 `apply_action` 流程

```python
def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
    # actions: (num_envs, 10) in [-1, 1], dtype=float32

    state.info["last_actions"] = state.info.get("current_actions",
                                                 np.zeros_like(actions))
    state.info["current_actions"] = actions

    # 1. base velocity → controller（内部有独立 latency，不经过 simulate_action_latency）
    base_action = actions[:, 0:3]
    self._base_controller.step(base_action)

    # 2. arm/gripper latency（可选，只对 action[3:10] 生效）
    if self._cfg.control_config.simulate_action_latency:
        arm_gripper_action = state.info["last_actions"][:, 3:10]
    else:
        arm_gripper_action = actions[:, 3:10]

    # 3. 将 world goal 转换到当前 armbase 系（每步动态转换）
    armbase_ee_goal = self._world_goal_to_armbase(
        self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world)

    # 4. arm IK + residual delta（使用 armbase 系 goal）
    dq_ik = self.compute_arm_ik_delta(
        armbase_ee_goal,
        self.get_ee_local_pose()[0],
        self.ee_goal_orn_quat,
        self.get_ee_local_pose()[1],
    )
    arm_ctrl = (
        self.get_arm_dof_pos()
        + arm_gripper_action[:, 0:6] * self._cfg.control_config.arm_action_scale
        + ik_gain * dq_ik
    )

    # 5. gripper position: [-1,1] → [0, 0.65]
    grip_norm = arm_gripper_action[:, 6]
    grip_ctrl = (grip_norm + 1.0) / 2.0 * 0.65

    # 6. return 7-dim ctrl for backend.step()
    ctrl = np.concatenate([arm_ctrl, grip_ctrl[:, None]], axis=1)
    return np.clip(ctrl, self._ctrl_low, self._ctrl_high)
```

### 4.4 单步 Raw Observation 布局

```
linvel(3) + gyro(3) + (-gravity_body)(3) + arm_diff(6) + arm_dof_vel(6) +
ee_local_pos(3) + armbase_ee_goal(3) + ee_error(3) + world_ee_goal(3) +
gripper_pos(1) + last_actions(10) = 44 维
```

| 段 | 维度 | 来源 | 说明 |
|---|---|---|---|
| `linvel` | 3 | `velocimeter` (base IMU) | base-frame 线速度（局部系） |
| `gyro` | 3 | `gyro` (base IMU) | base 角速度 |
| `-gravity_body` | 3 | `framequat` 计算 | base-frame 投影重力（gravity_world=[0,0,-1] 经 R_wb 旋转） |
| `arm_diff` | 6 | `jointpos` arm − default_angles | arm 偏离默认姿态 |
| `arm_dof_vel` | 6 | `jointvel` arm | arm 关节角速度 |
| `ee_local_pos` | 3 | `framepos` (endpoint/armbase) | EE 在 armbase 系下的位置 |
| `armbase_ee_goal` | 3 | `_world_goal_to_armbase()` 每步计算 | 当前段目标在 armbase 系（从 world_ee_goal 转换） |
| `ee_error` | 3 | `armbase_ee_goal − ee_local_pos` | armbase 系下指向目标的误差向量 |
| `world_ee_goal` | 3 | task 生成（世界系，固定） | 世界坐标系下的目标 EE 位置 |
| `gripper_pos` | 1 | `jointpos` finger1 | 夹爪开合位置 |
| `last_actions` | 10 | 上一帧 `current_actions` | 策略上一步全部输出 |

> **目标采样与观测流：** ① 任务在 world frame 采样 `world_ee_goal`，固定不变；② 每步 `update_state` 根据当前 armbase 位姿将 world goal 转换到 armbase 系得 `armbase_ee_goal`；③ IK 和 obs 中的 `ee_error` 使用 `armbase_ee_goal`；④ 策略通过 `world_ee_goal` 和 `armbase_ee_goal` 的差异感知 base 与 goal 的相对关系，从而学会移动底盘。
>
> **为什么不用 accelerometer 做 gravity：** 加速度计包含线加速度分量，机器人运动时 `accelerometer ≠ g`。改用 `framequat` 计算：`gravity_body = R_wb^T @ [0,0,-1]`。
>
> **`ee_error` 符号：** `armbase_ee_goal − ee_local_pos`（目标减当前），与 IK delta、reach reward 方向一致。

**不包含的观测：** steering/wheel joint 状态（纯可视化）、feet_phase（无腿足）、accelerometer（含线加速度）。v1 不做 privileged 拆分 —— actor 和 critic 都用相同 44 维。

### 4.5 历史堆叠与 `obs_groups_spec`

```python
_RAW_OBS_DIM = 44

@property
def obs_groups_spec(self) -> dict[str, int]:
    H_a = self._cfg.history.num_actor_history
    H_c = self._cfg.history.num_critic_history
    return {"obs": H_a * _RAW_OBS_DIM, "critic": H_c * _RAW_OBS_DIM}

def _compute_obs(self, ...) -> dict[str, np.ndarray]:
    actor_raw = self._compute_raw_obs(..., add_noise=True)
    critic_raw = self._compute_raw_obs(..., add_noise=False)
    history = self._update_history(actor_raw, critic_raw_obs=critic_raw)
    return {
        "obs": history["obs"].astype(np.float32),
        "critic": history["critic"].astype(np.float32),
    }
```

两个 key 始终都返回（与 Go2_arm 一致），core key 匹配 `obs_groups_spec`。默认 `num_actor_history=1`, `num_critic_history=1` → `{"obs": 44, "critic": 44}`。

### 4.6 观测噪声

| 参数 | 默认值 | 作用于 |
|------|--------|--------|
| `noise_config.level` | 0.0 | 总开关（train 时 1.0） |
| `scale_linvel` | 0.1 | linvel |
| `scale_gyro` | 0.2 | gyro |
| `scale_gravity` | 0.05 | gravity_body |
| `scale_joint_angle` | 0.03 | arm_diff, gripper_pos |
| `scale_joint_vel` | 0.5 | arm_dof_vel |
| `scale_ee_pos` | 0.02 | ee_local_pos |
| `scale_ee_goal` | 0.01 | ee_goal_cart, ee_goal_base（新增） |

### 4.7 与 Go2_arm 的对齐点

| 对齐项 | 方式 |
|--------|------|
| `obs_groups_spec` | `{"obs": H_a*D, "critic": H_c*D}`，返回 `dict[str, np.ndarray]` |
| `_compute_raw_obs` | 相同布局逻辑，维度不同（44 vs 79） |
| `_update_history` | **必须重写**（Go2Arm 固定去除前 3 维 linvel，导致 44→41 维冲突） |
| `compute_arm_ik_delta` | 直接复用（阻尼最小二乘 IK） |
| `get_ee_local_pose` | 直接复用（framepos + framequat sensor） |
| `simulate_action_latency` | 仅作用于 arm/gripper（index 3:10），base 走独立 controller |
| `_init_action_space` | **必须重写**（10-dim policy vs 7 backend actuators） |
| gravity 来源 | 从 `framequat` 计算，不用 accelerometer |

---

## 第五节：Reward 设计

### 5.1 任务定义（v1）

**Free-space EE reaching**：策略控制 base + arm + gripper 将末端从初始位姿移动到球面采样的目标位姿，到达后周期性重采样新目标。无物体交互，无抓取。方向跟踪默认关。

**目标坐标系（关键设计决策）：** 目标采样并固定在 **world frame**（不是 armbase 系），每步根据当前 armbase 位姿将 world goal 转换到 armbase 系供 IK 使用。这样 base 运动可以缩短 world-frame EE→goal 距离，策略才有 incentive 移动底盘。Go2_arm 的 armbase-frame 采样不能直接搬用到"底盘主动靠近目标"的任务。

**IK-only 验证 vs 策略训练：** `arm_action_scale` 默认 0.03（不是 0.0）。设为 0.0 会使 6 维 arm action 完全失效，策略退化为 3 维 base-only。训练时用 0.02~0.05 让 10 维 action 全部有效；纯 IK 验证时可临时 override 为 0.0 并缩减 action 空间到 3 维 base。

> 所有公式返回**正值（误差/惩罚量）**，负号只在 YAML `scale` 中。完全对齐 Go2_arm 的 `rew = fn(ctx)`, `reward += rew * scale` 模式。

### 5.2 奖励项清单

#### 5.2.1 主任务：EE 到达（world-frame）

EE 位置和 goal 都转换到 world frame 后计算距离 —— 这样 base 靠近 goal 时 reward 立即改善，直接激励 base-arm 协同。

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `ee_distance` | `exp(−‖ee_pos_world − world_ee_goal‖² / σ²)` | **4.0** | 高斯核到达，值域 [0,1]，scale 为正 |
| `ee_distance_l2` | `‖ee_pos_world − world_ee_goal‖²` | **−1.0** | L2 距离平方，远距离时梯度优于 exp |

`σ_ee = 0.15`（workspace 约 1.5m，比 Go2_arm 的 0.1 略大）。

其中 `ee_pos_world = armbase_pos_world + R_ab_w @ ee_local_pos`，每步从 armbase 位姿和 local EE 位置实时计算。

#### 5.2.2 底盘效率

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `base_vel_xy` | `vx² + vy²` | **−0.05** | 抑制不必要的底盘移动（scale 小，不妨碍任务必需的移动） |
| `base_vel_z` | `vz²` | **−1.0** | 策略不控 Z，vz 应为 0 |

#### 5.2.3 臂运动平滑

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `arm_dof_vel` | `‖q̇_arm‖²` | **−0.001** | L2 臂关节速度 |
| `arm_dof_acc` | `‖q̈_arm‖²` | **−1e-6** | L2 臂关节加速度。用相邻两步 `arm_vel` 差分实现（`info["qacc"]` 不存在），scale 极小 |
| `torques` | `‖τ‖₁` | **0.0** | 先关（backend 不暴露 torques，Go2_arm 同理） |

#### 5.2.4 安全 / 约束

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `base_orientation` | `gravity_x² + gravity_y²` | **−2.0** | 自定义实现，替代 `rewards.roll` |
| `base_height` | `(h − h_target)²` | **−20.0** | 底盘高度偏离 keyframe |
| `arm_joint_limits` | `∑ max(q−q_hi+m, 0)² + ∑ max(q_lo−q+m, 0)²` | **−1.0** | 臂关节软限位（margin=0.01 rad） |
| `arm_collision` | `∑ touch_forces` | **0.0** | 先关（v1 XML 无 touch sensor） |

#### 5.2.5 存活 / 正则

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `alive` | `1.0`（每步常数） | **0.3** | 存活奖励 |
| `action_rate` | `‖a_t − a_{t−1}‖²` | **−0.01** | 动作平滑 |
| `similar_to_default` | `‖q_arm − q_default‖₁` | **−0.005** | 臂姿态 L1 正则 |

### 5.3 不包含的奖励项

| Go2_arm 项 | 移除原因 |
|-------------|----------|
| `tracking_lin_vel` / `tracking_ang_vel` | base 速度是策略输出，不是外部命令 |
| `ang_vel_xy` / `stand_still` | 无腿足站立概念 |
| `swing_feet_z` / `foot_drag` / `contact` | 无腿足步态 |
| `leg_pose` / `dof_pos_limits` (leg) | 无腿关节 |
| `energy` | 已被 `torques` 覆盖（两者都关） |

### 5.4 RewardContext

```python
@dataclass
class RewardContext:
    info: dict
    linvel: np.ndarray            # (N, 3) base-frame linvel
    gyro: np.ndarray              # (N, 3) base angular vel
    gravity: np.ndarray           # (N, 3) projected gravity (from framequat)
    arm_pos: np.ndarray           # (N, 6) controlled arm positions
    arm_vel: np.ndarray           # (N, 6) controlled arm velocities
    gripper_pos: np.ndarray       # (N, 1) gripper position
    num_envs: int
    default_angles: np.ndarray    # (7,) arm home pose + gripper open
    armbase_pos_world: np.ndarray # (N, 3) arm base origin in world frame
    armbase_quat_world: np.ndarray# (N, 4) arm base quat (wxyz) in world
    ee_local_pos: np.ndarray      # (N, 3) current EE position in armbase frame
    ee_pos_world: np.ndarray      # (N, 3) current EE position in world frame
    world_ee_goal: np.ndarray     # (N, 3) goal EE position in world frame (canonical, fixed)
    armbase_ee_goal: np.ndarray   # (N, 3) world goal converted to armbase frame (for IK)
    sigma_ee: float               # 0.15
    base_height_target: float     # keyframe base Z
    base_height: np.ndarray       # (N,) current base Z
    arm_joint_upper: np.ndarray   # (6,) CR10 upper limits (rad)
    arm_joint_lower: np.ndarray   # (6,) CR10 lower limits (rad)
    joint_limit_margin: float     # 0.01
```

### 5.5 终止条件

| 条件 | 阈值 | 类型 |
|------|------|------|
| `gravity_x² + gravity_y² > sin²(1.0)` | roll/pitch > 1.0 rad (≈57°) | **terminated** |
| `base_height < 0.1 m` | 底盘触地 | **terminated** |
| `any arm_joint < hard_lower` 或 `> hard_upper` | 臂关节超出硬限位 | **terminated** |
| `steps >= max_episode_steps` | 超时 | **truncated** |

> 不用 `gravity[:, 2] ≤ 0.5`（Go2_arm 的翻转终止）—— freejoint 底盘没有"翻转"概念，直接用 tilt 平方。

### 5.6 计算骨架

```python
def _compute_reward(self, info, linvel, gyro, gravity,
                    dof_pos, dof_vel, ee_local_pos) -> np.ndarray:
    ctx = RewardContext(...)
    reward = np.zeros((self._num_envs,), dtype=get_global_dtype())
    for name, scale in self._reward_cfg.scales.items():
        if scale == 0 or name not in self._reward_fns:
            continue
        rew = self._reward_fns[name](ctx)       # 正值
        reward += rew * scale                   # scale 决定正负
    return reward * self._cfg.ctrl_dt
```

### 5.7 YAML reward 配置

```yaml
reward:
  scales:
    ee_distance: 4.0
    ee_distance_l2: -1.0
    base_vel_xy: -0.05
    base_vel_z: -1.0
    arm_dof_vel: -0.001
    arm_dof_acc: -1.0e-6
    torques: 0.0
    base_orientation: -2.0
    base_height: -20.0
    arm_joint_limits: -1.0
    arm_collision: 0.0
    action_rate: -0.01
    similar_to_default: -0.005
    alive: 0.3
  sigma_ee: 0.15
  base_height_target: 0.278
```

---

## 第六节：Config YAML 设计

### 6.1 文件结构

```
conf/ppo/task/ranger_box_reach/
└── mujoco.yaml    ← PPO × MuJoCo owner YAML（v1）
```

后续如需 Motrix 后端或 sim2sim，再提取 `base.yaml`。

### 6.2 `conf/ppo/task/ranger_box_reach/mujoco.yaml`

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
      - actor                    # RSL wrapper: obs["obs"] → TensorDict["actor"]
  policy:
    init_noise_std: 0.5
    actor_hidden_dims: [256, 128, 64]
    critic_hidden_dims: [256, 128, 64]
  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 1.0e-3
    num_mini_batches: 4
reward:
  scales:
    ee_distance: 4.0
    ee_distance_l2: -1.0
    base_vel_xy: -0.05
    base_vel_z: -1.0
    arm_dof_vel: -0.001
    arm_dof_acc: -1.0e-6
    torques: 0.0
    base_orientation: -2.0
    base_height: -20.0
    arm_joint_limits: -1.0
    arm_collision: 0.0
    action_rate: -0.01
    similar_to_default: -0.005
    alive: 0.3
  sigma_ee: 0.15
  base_height_target: 0.278
env:
  max_episode_seconds: 30.0
  init_state:
    pos: [0.0, 0.0, 0.278]
  control_config:
    arm_action_scale: 0.03       # 让 10 维 action 全部有效（不为 0.0）
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
    sphere_l_range: [0.20, 0.55]
    sphere_phi_range: [-1.20, 1.00]
    sphere_theta_range: [-2.00, 2.00]
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
    randomize_ground_friction: false  # freejoint, no wheel-ground contact
    randomize_dof_armature: true
    dof_armature_multiplier_range: [0.8, 1.2]
    push_robots: true
    push_interval: 500
    max_force: [1.2, 1.2, 0.6]
    push_body_name: base
  history:
    num_actor_history: 1
    num_critic_history: 1
  arm_stage:
    freeze_arm_joints: false
    disable_ee_goal_trajectory: false
    fixed_ee_goal_cart: [0.30, 0.0, 0.30]
```

### 6.3 YAML → Dataclass 映射

| YAML 路径 | Dataclass |
|-----------|-----------|
| `training.task_name` | registry key `"RangerBoxReach"` |
| `algo.*` | Hydra 顶层 → train script 消费 |
| `algo.obs_groups.actor` | RSL wrapper 约定 `["actor"]`（obs dict → TensorDict） |
| `reward.scales` + `sigma_ee` + `base_height_target` | `RangerBoxRewardConfig` |
| `env.control_config.*` | `RangerBoxControlConfig`（含 `gripper_kd`） |
| `env.asset.*` | `RangerBoxAsset`（arm_joint_names, ee_site_name, wheel 信息） |
| `env.sensor.*` | `RangerBoxSensor`（统一传感器名称映射） |
| `env.noise_config.*` | `RangerBoxNoiseConfig`（含 `scale_ee_goal`） |
| `env.goal_ee.*` | `EEGoalConfig`（复用 Go2_arm，但 goal 采样到 world frame） |
| `env.base_velocity_controller.*` | `BaseVelocityControllerConfig` |
| `env.ik.*` | `IKConfig`（复用 Go2_arm） |
| `env.domain_rand.*` | `RangerBoxDomainRandConfig`（含 `randomize_kp/kd`） |
| `env.history.*` | `HistoryConfig`（复用 Go2_arm） |
| `env.position_actuator_gains` | 运行时由 `build_ranger_box_position_gains()` 填充 |

### 6.4 与 Go2_arm YAML 的关键差异

| 字段 | Go2_arm | RangerBox | 原因 |
|------|---------|-----------|------|
| `env.commands` | velocity command ranges | **无** | base 速度是策略输出 |
| `env.curriculum` | velocity curriculum | **无** | 无速度命令 |
| `env.domain_rand.randomize_ground_friction` | true | **false** | 无轮地接触 |
| `env.control_config.leg_kp/kd` | 60/2 | **无** | 无腿关节 |
| `env.base_velocity_controller` | **无** | 新增 | A+ 方案特有 |
| `reward.tracking_sigma` | 0.25 | **无** | 无速度跟踪 |
| `reward.object_sigma` | 0.1 | **sigma_ee: 0.15** | 改名，值略大 |
| `reward.scales` 项数 | 15（含大量 0.0） | **13**（2 项暂时置零） | 精简 |
| `env.ik.use_orientation` | true | **false** | v1 只做位置 tracking |
| `algo.obs_groups.actor` | `["actor"]` | `["actor"]` | 一致 |

### 6.5 eval 覆盖示例

```bash
uv run eval \
  --algo ppo --task ranger_box_reach --sim mujoco \
  --load-run <run_dir> \
  env.base_velocity_controller.enable_noise=false \
  env.base_velocity_controller.enable_latency=false \
  env.noise_config.level=0.0
```

---

## 第七节：Env 类层次与文件组织

### 7.1 类层次

```
NpEnv                                   # base/np_env.py
  └── LocomotionBaseEnv                 # locomotion/common/base.py
        └── Go2ArmBaseEnv               # locomotion/go2_arm/base.py
              └── RangerBoxReachEnv     # locomotion/ranger_box/reach_env.py  [NEW]
```

继承 `Go2ArmBaseEnv` 复用：`compute_arm_ik_delta`、`get_ee_local_pose`、`get_arm_dof_pos/vel`、Jacobian DOF 索引、`_obs_noise`。

**不继承** `Go2ArmManipLocoEnv` —— 含腿足 gait、command resampling、feet sensor 等，不适用于 RangerBox。

### 7.2 关键方法重写矩阵

| 方法 | 父类行为 | RangerBox 重写 |
|------|---------|---------------|
| `_init_action_space` | 从 `backend.get_actuator_ctrl_range()` 推 dim | 固定 `Box(-1,1,(10,), float32)` |
| `_init_buffers` | `default_angles = init_qpos[-_num_action:]` | 从 CR10 home pose 常量构建 7 维 |
| `apply_action` | 全局 action_scale + default_angles | 拆分 base→controller / arm→IK / gripper→绝对 |
| `update_state` | gait + command resample + EE trajectory | 去掉 gait/command，保留 EE trajectory |
| `_compute_raw_obs` | 79 维 | 44 维（§4.4 布局） |
| `_update_history` | `actor_step = raw_obs[:, 3:]`（删除前 3 维 linvel） | **必须重写**，actor 和 critic 都保留完整 44 维 |
| `_compute_obs` | 依赖 `_update_history` | 同模式，维度不同 |
| `_compute_reward` | 15 项（含 locomotion） | 13 项（§5.2） |
| `obs_groups_spec` | `{"obs": H_a*76, "critic": H_c*79}` | `{"obs": H_a*44, "critic": H_c*44}` |
| `_init_reward_functions` | 含 legged rewards | 替换为 RangerBox 专用 |
| termination logic | `gravity[:,2] ≤ 0.5` | `gravity_x² + gravity_y² > sin²(1.0)` |

### 7.3 Env 构造流程

Registry 创建环境时调用 `env_cls(env_cfg, num_envs=num_envs, backend_type=sim_backend)`。`RangerBoxReachEnv.__init__` 负责完整初始化链：

```python
class RangerBoxReachEnv(Go2ArmBaseEnv):
    def __init__(self, cfg: RangerBoxReachCfg, num_envs: int = 1,
                 backend_type: str = "mujoco"):
        # 1. resolve scene（对齐 Go2_arm 模式）
        scene = _resolve_ranger_box_scene(cfg)

        # 2. 构建 7 维 position actuator gains（6 arm + 1 gripper）
        kp, kd = build_ranger_box_position_gains(cfg)
        cfg.position_actuator_gains = tuple(zip(kp, kd))

        # 3. create backend
        backend = create_backend(backend_type, scene, cfg)

        # 4. 调用 Go2ArmBaseEnv.__init__
        super().__init__(cfg, num_envs, backend)

        # 5. RangerBox 特有组件
        self._base_controller = BaseVelocityController(
            cfg.base_velocity_controller,
            cfg.ctrl_dt,
            backend,
            cfg.asset,             # wheel positions, radius, joint names
        )
        self.world_ee_goal = np.zeros((num_envs, 3), dtype=get_global_dtype())
        self.armbase_pos_world = np.zeros((num_envs, 3), dtype=get_global_dtype())
        self.armbase_quat_world = np.zeros((num_envs, 4), dtype=get_global_dtype())

        # 6. 初始化 goal、history buffer、reward fns、DR
        self._init_ee_goals()
        self._init_history_buffers()
        self._init_reward_functions()


def build_ranger_box_position_gains(cfg) -> tuple[np.ndarray, np.ndarray]:
    """模块级函数：返回 Kp (7,) 和 Kd (7,)：6 arm + 1 gripper."""
    cc = cfg.control_config
    kp = np.array([
        *cc.arm_kp,         # 6 floats
        cc.gripper_kp,      # 1 float
    ], dtype=get_global_dtype())
    kd = np.array([
        *cc.arm_kd,         # 6 floats
        cc.gripper_kd,      # 1 float
    ], dtype=get_global_dtype())
    return kp, kd
```

**关键：** `position_actuator_gains` 是 7 维 tuple，严格对应 robot.xml 中 7 个 `<position>` actuator 的顺序（cr10_j1~j6 然后 gripper_finger1）。Go2_arm 同一模式用 `build_go2_arm_position_gains()` 构造。

### 7.4 `default_angles` 处理

父类从 keyframe **qpos** 尾部取 `_num_action` 维（qpos 尾部含所有 joints，不是 arm+gripper）。重写 `_init_buffers`：

```python
# CR10 home pose + gripper open (rad) — 对齐 keyframe ctrl
_ARM_DEFAULT = np.array([0.0, -0.3, 0.75, 0.0, 0.45, 0.0], dtype=np.float64)
_GRIPPER_DEFAULT = np.array([0.0], dtype=np.float64)

def _init_buffers(self) -> None:
    dtype = get_global_dtype()
    raw_qpos = self._backend.get_keyframe_qpos(self._keyframe_name)
    self._init_qpos = np.asarray(raw_qpos, dtype=dtype)
    self.default_angles = np.concatenate([
        np.asarray(self._ARM_DEFAULT, dtype=dtype),
        np.asarray(self._GRIPPER_DEFAULT, dtype=dtype),
    ])
    raw_qvel = self._backend.get_init_qvel()
    self._init_qvel = np.asarray(raw_qvel, dtype=dtype)
```

### 7.5 `_update_history` 重写

Go2Arm 的 `_update_history()` 固定执行 `actor_step = raw_obs[:, 3:]`（删除前 3 维 linvel），与 44 维完整 obs 冲突。RangerBox 必须重写，actor 和 critic 都保留全部 44 维：

```python
def _update_history(self, obs, critic_raw_obs=None):
    self._history_obs_buf = np.roll(self._history_obs_buf, -1, axis=0)
    self._history_obs_buf[-1] = obs          # full 44 dims, no slicing
    history = {"obs": self._history_obs_buf.reshape(self._num_envs, -1).astype(np.float32)}
    if critic_raw_obs is not None:
        self._history_critic_buf = np.roll(self._history_critic_buf, -1, axis=0)
        self._history_critic_buf[-1] = critic_raw_obs  # also full 44 dims
        history["critic"] = self._history_critic_buf.reshape(self._num_envs, -1).astype(np.float32)
    return history
```

### 7.6 `update_state` 流程

```python
def update_state(self, state: NpEnvState) -> NpEnvState:
    # 1. EE goal trajectory（world-frame 球面插值）
    self._update_ee_goal_trajectory()

    # 2. 传感器读取
    linvel = self.get_local_linvel()              # velocimeter: base-frame (N,3)
    gyro = self.get_gyro()                        # gyro: (N,3)
    gravity = self._get_projected_gravity()       # from framequat: (N,3)
    ee_local_pos, ee_local_quat = self.get_ee_local_pose()  # framepos + framequat

    # 3. 只取受控 DOF（arm 6 + gripper 1），不用 get_dof_pos() 全量
    arm_pos = self.get_arm_dof_pos()              # Go2ArmBaseEnv 方法，(N,6)
    arm_vel = self.get_arm_dof_vel()
    gripper_pos = self.get_gripper_dof_pos()      # (N,1)
    gripper_vel = self.get_gripper_dof_vel()

    # 4. armbase 位姿（用于 world↔armbase 转换）
    armbase_pos_world, armbase_quat_world = self._get_armbase_world_pose()

    # 5. world goal → armbase 系
    armbase_ee_goal = self._world_goal_to_armbase(
        self.world_ee_goal, armbase_pos_world, armbase_quat_world)

    # 6. world-frame EE 位置（用于 reward）
    ee_pos_world = armbase_pos_world + quat_rotate(armbase_quat_world, ee_local_pos)

    # 7. 终止判断
    tilt_sq = gravity[:, 0]**2 + gravity[:, 1]**2
    base_z = self._backend.get_base_pos()[:, 2]
    terminated = (
        (tilt_sq > np.sin(1.0)**2) |
        (base_z < 0.1) |
        self._arm_joint_hard_limits_violated(arm_pos)
    )

    # 8. Reward + Obs
    reward = self._compute_reward(
        state.info, linvel, gyro, gravity,
        arm_pos, arm_vel, gripper_pos, gripper_vel,
        armbase_pos_world, armbase_quat_world,
        ee_local_pos, ee_pos_world, self.world_ee_goal, armbase_ee_goal)

    obs = self._compute_obs(
        state.info, linvel, gyro, gravity,
        arm_pos, arm_vel, gripper_pos,
        ee_local_pos, armbase_ee_goal, self.world_ee_goal)

    return state.replace(obs=obs, reward=reward, terminated=terminated)
```

**受控 DOF 切片（重要）：** `get_dof_pos()/get_dof_vel()` 返回全部非 freejoint DOF（包括 4 steering + 4 wheel + 8 gripper mimic joints）。reward、obs、joint limit 和 default_angles 都只应使用 6 arm + 1 gripper 受控 DOF。Go2ArmBaseEnv 已提供 `get_arm_dof_pos()/get_arm_dof_vel()`，gripper 需要新增 `get_gripper_dof_pos/vel()`。

### 7.7 Projected Gravity 计算

```python
def _get_projected_gravity(self) -> np.ndarray:
    """World gravity [0,0,-1] expressed in base frame. Returns (N, 3)."""
    quat = self._backend.get_sensor_data("framequat")    # (N, 4) wxyz
    R_wb = np_matrix_from_quat(quat)                     # (N, 3, 3) body→world
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=get_global_dtype())
    gravity_body = np.einsum("nij,j->ni", np.swapaxes(R_wb, 1, 2), gravity_world)
    return gravity_body
```

### 7.8 Scene / Model 解析

```python
def _default_ranger_box_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "ranger_box" / "scene_flat.xml")

def _default_ranger_box_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_ranger_box_model_file())

def _resolve_ranger_box_scene(cfg: RangerBoxReachCfg) -> SceneCfg:
    """对齐 _resolve_go2_arm_scene 模式."""
    scene = cfg.scene
    default_model_file = _default_ranger_box_model_file()
    if scene is None:
        scene = SceneCfg(model_file=cfg.model_file)
    elif cfg.model_file != default_model_file and scene.model_file == default_model_file:
        scene = SceneCfg(
            model_file=cfg.model_file,
            fragment_files=list(scene.fragment_files),
            terrain=scene.terrain,
        )
    cfg.scene = scene
    return scene
```

### 7.9 注册集成

`src/unilab/envs/locomotion/__init__.py` 加一条：

```python
__unilab_registry_modules__ = (
    # ... 现有模块保持不变 ...
    "unilab.envs.locomotion.ranger_box",   # [NEW] — 包级别，对齐 Go2_arm 模式
)
```

`ranger_box/__init__.py` 负责导入 `RangerBoxReachEnv`，触发 `@registry.envcfg("RangerBoxReach")` 和 `@registry.env("RangerBoxReach", sim_backend="mujoco")` 注册。

### 7.10 DR Provider

```python
class RangerBoxReachDRProvider(LocomotionDRProvider):
    """Domain randomization provider for RangerBoxReach.

    Differs from Go2_arm: no velocity command sampling, no gait phase,
    includes BaseVelocityController reset. Must override _sample_commands
    to return zero commands (parent calls env.cfg.commands which doesn't exist).
    """

    def _sample_commands(self, env, env_ids, rng):
        """Override: no velocity commands in this task."""
        return {
            "vel_x": np.zeros(len(env_ids)),
            "vel_y": np.zeros(len(env_ids)),
            "vel_yaw": np.zeros(len(env_ids)),
        }

    def build_reset_plan(self, env, env_ids):
        plan = super().build_reset_plan(env, env_ids)
        env.reset_ee_goals(env_ids)
        env._arm_goal_timer[env_ids] = 0
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._base_controller.reset(env_ids, env._np_rng)
        return plan

    def _compute_reset_obs(self, env, env_ids, info_updates, ...):
        # 同 Go2_arm 模式，raw_obs 维度为 44
        ...
```

### 7.11 新增 Dataclass 汇总

```python
# ─── Asset ───────────────────────────────────────────────
@dataclass
class RangerBoxAsset:
    model_file: str = ""                                           # Hydra 注入
    arm_joint_names: tuple[str, ...] = (
        "cr10_joint1", "cr10_joint2", "cr10_joint3",
        "cr10_joint4", "cr10_joint5", "cr10_joint6",
    )
    gripper_joint_name: str = "gripper_finger1_joint"
    ee_site_name: str = "right_center"
    armbase_body_name: str = "cr10_Link1"                          # arm ref body for framepos
    base_body_name: str = "base"
    steering_joint_names: tuple[str, ...] = (
        "fr_steering_joint", "fl_steering_wheel_joint",
        "rl_steering_wheel_joint", "rr_steering_wheel_joint",
    )
    wheel_joint_names: tuple[str, ...] = (
        "fr_wheel_joint", "fl_wheel_joint",
        "rl_wheel_joint", "rr_wheel_joint",
    )
    wheel_positions: tuple[tuple[float, float], ...] = (
        (0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28),
    )
    wheel_radius: float = 0.152

# ─── Sensor ───────────────────────────────────────────────
@dataclass
class RangerBoxSensor:
    """统一传感器名称映射，代码只读取 cfg.sensor.xxx，禁止硬编码字符串."""
    gyro: str = "imu-gyro"
    velocimeter: str = "imu-velocimeter"                           # base-frame 局部线速度
    framequat: str = "imu-framequat"                               # base 姿态 wxyz
    arm_jointpos: str = "arm-jointpos"
    arm_jointvel: str = "arm-jointvel"
    steering_jointpos: str = "steering-jointpos"
    steering_jointvel: str = "steering-jointvel"
    wheel_jointpos: str = "wheel-jointpos"
    wheel_jointvel: str = "wheel-jointvel"
    ee_framepos: str = "endpoint-framepos"                         # EE 在 armbase 系位置
    ee_framequat: str = "endpoint-framequat"                       # EE 在 armbase 系姿态
    armbase_framepos: str = "armbasepoint-framepos"                # arm base world pos
    armbase_framequat: str = "armbasepoint-framequat"              # arm base world quat
    gripper_jointpos: str = "gripper-jointpos"

# ─── Control ──────────────────────────────────────────────
@dataclass
class RangerBoxControlConfig(ControlConfigBase):
    arm_action_scale: float = 0.03
    arm_kp: tuple[float, ...] = (100.0, 110.0, 95.0, 50.0, 50.0, 50.0)
    arm_kd: tuple[float, ...] = (3.5, 3.8, 2.5, 1.5, 1.5, 1.5)
    gripper_kp: float = 500.0
    gripper_kd: float = 10.0

# ─── Domain Rand ──────────────────────────────────────────
@dataclass
class RangerBoxDomainRandConfig(DomainRandConfig):
    randomize_ground_friction: bool = False          # freejoint, no wheel-ground contact
    randomize_kp: bool = True
    kp_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = True
    kd_multiplier_range: tuple[float, float] = (0.9, 1.1)

# ─── Noise ────────────────────────────────────────────────
@dataclass
class RangerBoxNoiseConfig(NoiseConfig):
    scale_ee_goal: float = 0.01

# ─── Reward ───────────────────────────────────────────────
@dataclass
class RangerBoxRewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        "ee_distance": 4.0,
        "ee_distance_l2": -1.0,
        "base_vel_xy": -0.05,
        "base_vel_z": -1.0,
        "arm_dof_vel": -0.001,
        "arm_dof_acc": -1.0e-6,
        "torques": 0.0,
        "base_orientation": -2.0,
        "base_height": -20.0,
        "arm_joint_limits": -1.0,
        "arm_collision": 0.0,
        "action_rate": -0.01,
        "similar_to_default": -0.005,
        "alive": 0.3,
    })
    sigma_ee: float = 0.15
    base_height_target: float = 0.278

# ─── BaseVelocityController ───────────────────────────────
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

# ─── EnvCfg (顶层) ───────────────────────────────────────
@registry.envcfg("RangerBoxReach")
@dataclass
class RangerBoxReachCfg(Go2ArmBaseCfg):
    """继承 Go2ArmBaseCfg 以复用 task、ik、history、scene 字段."""
    asset: RangerBoxAsset = field(default_factory=RangerBoxAsset)
    sensor: RangerBoxSensor = field(default_factory=RangerBoxSensor)
    control_config: RangerBoxControlConfig = field(default_factory=RangerBoxControlConfig)
    noise_config: RangerBoxNoiseConfig = field(default_factory=RangerBoxNoiseConfig)
    domain_rand: RangerBoxDomainRandConfig = field(default_factory=RangerBoxDomainRandConfig)
    reward: RangerBoxRewardConfig = field(default_factory=RangerBoxRewardConfig)
    base_velocity_controller: BaseVelocityControllerConfig = field(
        default_factory=BaseVelocityControllerConfig)
    # 前向声明：env 构造后回填（Go2_arm 同模式）
    position_actuator_gains: tuple[tuple[float, float], ...] = ()
```

**关键点：**
- `RangerBoxAsset.arm_joint_names` 默认 `cr10_joint1~6`，不依赖 Go2ArmBaseEnv 的 `joint1~6` 默认值
- `RangerBoxSensor` 将所有 sensor name 集中管理，代码通过 `self._cfg.sensor.xxx` 读取，无硬编码字符串
- `RangerBoxDomainRandConfig` 显式声明 `randomize_kp/kd` 和 multiplier range，避免父类 `DomainRandConfig` 无此字段导致 Hydra 抛 `ValueError`
- `RangerBoxReachCfg` 继承 `Go2ArmBaseCfg`（包含 `task`、`ik`、`history`、`scene`），同时声明自己的子 config

---

## 附录 A：CR10 关节硬限位

| 关节 | XML range (rad) | 默认 kp | 说明 |
|------|-----------------|---------|------|
| cr10_joint1 | [-3.92, 0.94] | 100 | base rotation |
| cr10_joint2 | [-1.57, 1.57] | 110 | shoulder |
| cr10_joint3 | [-2.86, 2.86] | 95 | elbow |
| cr10_joint4 | [-3.14, 3.14] | 50 | wrist |
| cr10_joint5 | [-3.14, 3.14] | 50 | wrist |
| cr10_joint6 | [-3.14, 3.14] | 50 | flange |
| gripper_finger1_joint | [0, 0.65] | 500 (fixed) | grip open→close |

## 附录 B：底盘转向/车轮关节名称

| 关节 | steering name | wheel name | XML position (x, y) relative to base |
|------|---------------|------------|--------------------------------------|
| 右前 FR | `fr_steering_joint` | `fr_wheel_joint` | (+0.445, -0.28) |
| 左前 FL | `fl_steering_wheel_joint` | `fl_wheel_joint` | (+0.445, +0.28) |
| 左后 RL | `rl_steering_wheel_joint` | `rl_wheel_joint` | (-0.445, +0.28) |
| 右后 RR | `rr_steering_wheel_joint` | `rr_wheel_joint` | (-0.445, -0.28) |

> 注意：源 XML 中 steering 关节命名不一致（`fr_steering_joint` vs `fl_steering_wheel_joint`），适配后的 `robot.xml` 应统一命名。

## 附录 C：关键风险清单

| 风险 | 缓解 |
|------|------|
| freejoint qvel 角速度坐标系约定 | 确认 MuJoCoBackend 实现与 `mjData.qvel` world-frame 一致 |
| `default_angles` 从 keyframe 推导错误 | 显式常量 `_ARM_DEFAULT` + `_GRIPPER_DEFAULT`，不依赖 qpos 维度假定 |
| `set_root_planar_velocity` → `mj_step()` 竞态 | `backend.step()` 在 `mj_step()` 之前检查 qvel，只写 planar 分量，保留非受控分量 |
| arm IK 对移动底盘 numerical stability | `Go2ArmBaseEnv.compute_arm_ik_delta` 已验证 stability，复用 |
| freejoint + push DR 导致 base drift | `base_height` reward + termination 双重防护 |
| 观测噪声注入遗漏维度 | `_compute_raw_obs(add_noise=True)` 覆盖所有 44 维 |
