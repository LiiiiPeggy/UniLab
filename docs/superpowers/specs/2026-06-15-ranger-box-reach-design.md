# RangerBoxReach — 移动操作 EE 到达 设计文档

日期：2026-06-15 | 状态：设计完成，待审阅 | 任务：RangerboxCR10Lidar 移动底盘 + 臂 EE 目标到达

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
| `src/unilab/envs/locomotion/__init__.py` | `__unilab_registry_modules__` 加 `"unilab.envs.locomotion.ranger_box.reach_env"` | 低——纯注册 |
| `src/unilab/base/backend/base.py` | 新增 `set_root_velocity`、`set_joint_qpos`、`set_joint_qvel` 三个抽象方法 | 中——SimBackend 接口扩展，影响所有后端子类 |
| `src/unilab/base/backend/mujoco/backend.py` | 实现上述三个方法的 MuJoCo 版本 | 中——直接写 `mjData.qvel`/`qpos`，需确认 freejoint 角速度 world-frame 约定 |

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
| `set_root_velocity(lin_vel, ang_vel)` | `base/backend/base.py` | `mujoco/backend.py` | world-frame 写入 freejoint qvel[0:6] |
| `set_joint_qpos(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qpos（轮子可视化） |
| `set_joint_qvel(joint_names, values)` | `base/backend/base.py` | `mujoco/backend.py` | 按 joint 名写入 qvel（轮子可视化） |

三个方法在 `SimBackend` 默认抛 `NotImplementedError`，仅在 `MuJoCoBackend` 实现。

### 新增传感器（最小 RL 闭环）

| 类别 | sensor | 数量 | 用途 |
|------|--------|------|------|
| IMU（`imu` site on base body） | `gyro`、`framequat`、`framelinvel`、`framezaxis` | 4 | base 角速度、姿态、局部线速度、Z 轴方向 |
| 臂关节 | `jointpos` + `jointvel` | 6+6 | arm 本体感知 |
| 底盘关节 | `jointpos` + `jointvel` | 8+8 | 转向/车轮状态（观测用） |
| 末端位姿（相对 armbase） | `framepos`、`framequat`、`framelinvel` | 3 | EE goal tracking |
| 夹爪 | `jointpos` finger1 | 1 | gripper state |

> **v1 暂不加入**（可通过 config 后续打开）：`jointactuatorfrc`、`touch`、`accelerometer`。accelerometer 含线加速度分量，gravity 方向改用 `framezaxis` 或从 `framequat` 计算。

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
       backend.set_root_velocity(v_world, w_world)   ← 物理生效
```

关键：轮子 IK 输入是 **base-frame**（策略直接输出），`set_root_velocity` 写入 **world-frame**（由 base quat 旋转得到）。

### Keyframe（位于 `scene_flat.xml`，不在 `robot.xml`）

```xml
<keyframe>
  <key name="home"
    qpos="
      0 0 0.278            <!-- base xyz (freejoint) -->
      1 0 0 0              <!-- base quat wxyz -->
      0 0 0 0 0 0 0 0      <!-- 4 steering + 4 wheel = neutral -->
      0 -0.3 0.75 0 0.45 0 <!-- cr10_j1~j6 home pose (rad) -->
      0 0 0 0 0 0 0 0"     <!-- 8 gripper joints (finger1=0 open) -->
    ctrl="
      0 -0.3 0.75 0 0.45 0 <!-- cr10_j1~j6 position targets -->
      0"                   <!-- gripper_finger1 = open -->
  />
</keyframe>
```

- `qpos` = 7(freejoint) + 8(base) + 6(arm) + 8(gripper) = **29**
- `ctrl` = 6(arm position) + 1(gripper position) = **7** ← 与 `<actuator>` 块严格一致

### 传感器命名约定（对齐 Go2_arm）

| XML sensor name | Go2_arm cfg field | RangerBox 用途 |
|-----------------|-------------------|---------------|
| `imu-gyro` | `sensor.gyro` | base 角速度 |
| `imu-framequat` | `sensor.??` | base 姿态 → 计算 projected gravity |
| `imu-framelinvel` | `sensor.local_linvel` | base 局部线速度 |
| `armbasepoint-framepos` | — | EE ref 系原点 world pos |
| `armbasepoint-framequat` | `sensor.arm_ref_world_quat` | EE ref 系 world 姿态 |
| `endpoint-framepos` | `sensor.ee_local_pos` | EE 在 armbase 系下的位置 |
| `endpoint-framequat` | `sensor.ee_local_quat` | EE 在 armbase 系下的姿态 |

---

## 第三节：BaseVelocityController 设计

### 3.1 为什么是"方案 A+"？

- MuJoCo freejoint 提供 6-DOF 自由体运动学，无需求解轮地摩擦
- 训练效率远高于全接触动力学
- 对自由空间 EE tracking 任务，轮地接触细节不提供有意义的迁移信号
- Domain randomization 在控制器层实现（延迟、限幅、噪声）

### 3.2 控制器架构

```
action[0:3] = (vx_des, vy_des, vyaw_des)    ← 策略输出，base-frame

     │
     ▼
┌──────────────────────────────────────────────────┐
│             BaseVelocityController                │
│                                                  │
│  1. action scale   ──► [-1,1] → m/s, rad/s      │
│  2. command clip   ──► 限幅到安全范围              │
│  3. latency buffer ──► FIFO 延迟（模拟 CAN 传输）  │
│  4. accel limit    ──► 加速度限幅，防跳变          │
│  5. first-order    ──► 一阶平滑响应               │
│  6. noise / slip   ──► domain rand 加性噪声       │
│  7. final clip     ──► 最终安全限幅               │
│  8. wheel IK (viz) ──► wrap_to_pi + 轮子可视化    │
│  9. world-frame     ──► quat 旋转到世界坐标系      │
│                                                  │
│  输出: set_root_velocity(v_world, w_world)        │
│  输出: set_joint_qpos/qvel(steer, wheel)          │
└──────────────────────────────────────────────────┘
     │
     ▼
  backend.step() ──► MuJoCo 前向动力学
```

### 3.3 Step 1-2：Scale + Clip

```python
vx_cmd = clip(action[0] * action_scale_lin, -max_lin_vel, max_lin_vel)
vy_cmd = clip(action[1] * action_scale_lin, -max_lin_vel, max_lin_vel)
vyaw_cmd = clip(action[2] * action_scale_ang, -max_ang_vel, max_ang_vel)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_lin_vel` | 1.5 m/s | Ranger 最大线速度 |
| `max_ang_vel` | 1.0 rad/s | 约 57°/s |
| `action_scale_lin` | 1.5 | [-1,1] → [-1.5, 1.5] |
| `action_scale_ang` | 1.0 | [-1,1] → [-1, 1] |

### 3.4 Step 3：延迟缓冲（Latency Buffer）

```python
# reset:
latency_steps = rng.integers(0, max_latency_steps + 1)
latency_buffer = deque(
    [np.zeros(3, dtype=np.float32) for _ in range(latency_steps + 1)],
    maxlen=latency_steps + 1,
)

# step:
v_cmd = np.array([vx_cmd, vy_cmd, vyaw_cmd], dtype=np.float32)
latency_buffer.append(v_cmd)
v_cmd_delayed = latency_buffer[0]  # oldest, deque(maxlen) auto-drops
```

- 不手动 `popleft()`，`deque(maxlen=...)` 自动丢弃旧值
- `max_latency_steps` 默认 4（80ms @ 50Hz），受 `enable_latency` 开关控制
- eval 时通过 `env.base_velocity_controller.enable_latency=false` 关闭

### 3.5 Step 4：加速度限幅

```python
dv = v_cmd_delayed - v_real
dv = clip(dv, -acc_limits * dt, acc_limits * dt)
v_target = v_real + dv
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_lin_acc` | 1.5 m/s² | 线加速度硬限幅 |
| `max_ang_acc` | 3.0 rad/s² | 角加速度硬限幅 |

### 3.6 Step 5：一阶速度响应（稳定形式）

```python
alpha = dt / (tau + dt)           # 稳定，dt >= tau 也不会 > 1
v_real = v_real + alpha * (v_target - v_real)
```

- `tau` 默认 0.05s（50ms）

### 3.7 Step 6-7：噪声 + Final Clip

```python
if enable_noise:
    v_real[0] += noise_scale * max_lin_vel * randn()   # vx
    v_real[1] += noise_scale * max_lin_vel * randn()   # vy
    v_real[2] += noise_scale * max_ang_vel * randn()   # vyaw

v_real = clip(v_real, -max_vel, max_vel)  # final safety clip
```

噪声作用在一阶滤波**之后**（否则滤波会把噪声平滑掉），final clip 兜底。`enable_noise` 开关允许 eval 时关闭。

### 3.8 Step 8：轮子可视化 + 角度 Wrap

```python
# wheel positions from robot.xml (base-frame):
wheel_positions = {
    "FR": (+0.445, -0.28),
    "FL": (+0.445, +0.28),
    "RR": (-0.445, -0.28),
    "RL": (-0.445, +0.28),
}
wheel_radius = 0.152  # 6 英寸

for name, (x, y) in wheel_positions.items():
    vx_i = v_real[0] - v_real[2] * y
    vy_i = v_real[1] + v_real[2] * x
    steer_angle = wrap_to_pi(atan2(vy_i, vx_i))
    wheel_omega = sqrt(vx_i**2 + vy_i**2) / wheel_radius
```

受 `enable_wheel_visualization` 开关控制。

### 3.9 Step 9：World-Frame 转换

```python
R_wb = quat_to_matrix(base_quat)

v_body = np.array([v_real[0], v_real[1], 0.0])
w_body = np.array([0.0, 0.0, v_real[2]])

v_world = R_wb @ v_body
w_world = R_wb @ w_body

backend.set_root_velocity(v_world, w_world)
```

`set_root_velocity` 写入 freejoint qvel[0:6]（world-frame body linear velocity + world-frame body angular velocity，MuJoCo 标准约定）。

### 3.10 控制器生命周期

```python
class BaseVelocityController:
    def __init__(self, cfg: BaseVelocityControllerConfig, dt: float, backend: SimBackend):
        ...

    def reset(self, env_ids: np.ndarray, rng):
        # 随机采样 latency_steps，初始化 latency_buffer（0 填充），重置 v_real

    def step(self, action_base_vel: np.ndarray):
        # 1. scale → 2. clip → 3. latency → 4. accel limit
        # → 5. first-order → 6. noise → 7. final clip
        # → 8. wheel IK → 9. world-frame convert + set_root_velocity
```

### 3.11 与 Env 的关系

`BaseVelocityController` 是 env 的内部组件（composition），在 `RangerBoxReachEnv.__init__` 中构造。`apply_action()` 中 base 延迟由 controller 独立处理，不经过 arm/gripper 的 `simulate_action_latency`：

```python
def apply_action(self, actions, state):
    base_action = actions[:, 0:3]          # 不经过 simulate_action_latency
    self.base_controller.step(base_action)  # latency 在 controller 内部

    arm_gripper_action = exec_actions[:, 3:10]  # simulate_action_latency（可选）
    arm_ctrl = self.get_arm_dof_pos() + arm_gripper_action[:, 0:6] * arm_action_scale + ik_gain * dq_ik
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
    self.base_controller.step(base_action)

    # 2. arm/gripper latency（可选，只对 action[3:10] 生效）
    if self._cfg.control_config.simulate_action_latency:
        arm_gripper_action = state.info["last_actions"][:, 3:10]
    else:
        arm_gripper_action = actions[:, 3:10]

    # 3. arm IK + residual delta
    dq_ik = self.compute_arm_ik_delta(
        self.curr_ee_goal_cart,
        self.get_ee_local_pose()[0],
        self.ee_goal_orn_quat,
        self.get_ee_local_pose()[1],
    )
    arm_ctrl = (
        self.get_arm_dof_pos()
        + arm_gripper_action[:, 0:6] * arm_action_scale
        + ik_gain * dq_ik
    )

    # 4. gripper position: [-1,1] → [0, 0.65]
    grip_norm = arm_gripper_action[:, 6]
    grip_ctrl = (grip_norm + 1.0) / 2.0 * 0.65

    # 5. return 7-dim ctrl for backend.step()
    ctrl = np.concatenate([arm_ctrl, grip_ctrl[:, None]], axis=1)
    return np.clip(ctrl, self._ctrl_low, self._ctrl_high)
```

### 4.4 单步 Raw Observation 布局

```
linvel(3) + gyro(3) + (-gravity_body)(3) + arm_diff(6) + arm_dof_vel(6) +
ee_local_pos(3) + ee_goal_cart(3) + ee_error(3) + ee_goal_base(3) +
gripper_pos(1) + last_actions(10) = 44 维
```

| 段 | 维度 | 来源 | 说明 |
|---|---|---|---|
| `linvel` | 3 | `framelinvel` (base IMU) | base-frame 线速度 |
| `gyro` | 3 | `gyro` (base IMU) | base 角速度 |
| `-gravity_body` | 3 | `framequat` 计算 | base-frame 投影重力（gravity_world=[0,0,-1] 经 R_wb 旋转） |
| `arm_diff` | 6 | `jointpos` arm − default_angles | arm 偏离默认姿态 |
| `arm_dof_vel` | 6 | `jointvel` arm | arm 关节角速度 |
| `ee_local_pos` | 3 | `framepos` (endpoint/armbase) | EE 在 arm 基座系下的位置 |
| `ee_goal_cart` | 3 | task 生成（arm 基座系） | 当前段目标 EE 位置（球面插值） |
| `ee_error` | 3 | `ee_goal_cart − ee_local_pos` | **目标减当前**，指向目标的误差向量 |
| `ee_goal_base` | 3 | 常量变换 armbase→base | 目标在 base 坐标系下的位置 |
| `gripper_pos` | 1 | `jointpos` finger1 | 夹爪开合位置 |
| `last_actions` | 10 | 上一帧 `current_actions` | 策略上一步全部输出 |

> **为什么不用 accelerometer 做 gravity：** 加速度计包含线加速度分量，机器人运动时 `accelerometer ≠ g`。改用 `framequat` 计算：`gravity_body = R_wb^T @ [0,0,-1]`。`framezaxis` 始终为 `[0,0,1]`（body Z 在 body frame），不能直接用作 gravity。
>
> **`ee_error` 符号：** `ee_goal_cart − ee_local_pos`（目标减当前），与 IK delta、reach reward 方向一致。

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
| `_update_history` | 直接复用父类 rolling buffer |
| `compute_arm_ik_delta` | 直接复用（阻尼最小二乘 IK） |
| `get_ee_local_pose` | 直接复用（framepos + framequat sensor） |
| `simulate_action_latency` | 仅作用于 arm/gripper（index 3:10），base 走独立 controller |
| `_init_action_space` | **必须重写**（10-dim policy vs 7 backend actuators） |
| gravity 来源 | 从 `framequat` 计算，不用 accelerometer |

---

## 第五节：Reward 设计

### 5.1 任务定义（v1）

**Free-space EE reaching**：策略控制 base + arm + gripper 将末端从初始位姿移动到球面采样的目标位姿，到达后周期性重采样新目标。无物体交互，无抓取。方向跟踪默认关。

> 所有公式返回**正值（误差/惩罚量）**，负号只在 YAML `scale` 中。完全对齐 Go2_arm 的 `rew = fn(ctx)`, `reward += rew * scale` 模式。

### 5.2 奖励项清单

#### 5.2.1 主任务：EE 到达

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `ee_distance` | `exp(−‖ee_pos − ee_goal‖² / σ²)` | **4.0** | 高斯核到达，值域 [0,1]，scale 为正 |
| `ee_distance_l2` | `‖ee_pos − ee_goal‖²` | **−1.0** | L2 距离平方，远距离时梯度优于 exp |

`σ_ee = 0.15`（workspace 约 1.5m，比 Go2_arm 的 0.1 略大）。

#### 5.2.2 底盘效率

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `base_vel_xy` | `vx² + vy²` | **−0.05** | 抑制不必要的底盘移动（scale 小，不妨碍任务必需的移动） |
| `base_vel_z` | `vz²` | **−1.0** | 策略不控 Z，vz 应为 0 |

#### 5.2.3 臂运动平滑

| 名称 | 公式 | 默认 scale | 说明 |
|------|------|------------|------|
| `arm_dof_vel` | `‖q̇_arm‖²` | **−0.001** | L2 臂关节速度 |
| `arm_dof_acc` | `‖q̈_arm‖²` | **−1e-6** | L2 臂关节加速度（值域大，scale 极小） |
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
    linvel: np.ndarray          # (N, 3) base-frame linvel
    gyro: np.ndarray            # (N, 3) base angular vel
    gravity: np.ndarray         # (N, 3) projected gravity (from framequat)
    dof_pos: np.ndarray         # (N, 7) arm+gripper positions
    dof_vel: np.ndarray         # (N, 7) arm+gripper velocities
    num_envs: int
    default_angles: np.ndarray  # (7,) arm home pose + gripper open
    ee_local_pos: np.ndarray    # (N, 3) current EE in arm frame
    ee_goal_cart: np.ndarray    # (N, 3) target EE in arm frame
    sigma_ee: float             # 0.15
    base_height_target: float   # keyframe base Z
    base_height: np.ndarray     # (N,) current base Z
    arm_joint_upper: np.ndarray # (6,) CR10 upper limits (rad)
    arm_joint_lower: np.ndarray # (6,) CR10 lower limits (rad)
    joint_limit_margin: float   # 0.01
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
    arm_action_scale: 0.0        # IK-only 初期验证；策略残差训练时改 0.02~0.05
    simulate_action_latency: false
    arm_kp: [100.0, 110.0, 95.0, 50.0, 50.0, 50.0]
    arm_kd: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5]
    gripper_kp: 500.0
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
| `reward.scales` | `RewardConfig.scales` |
| `env.control_config.*` | `RangerBoxControlConfig` |
| `env.noise_config.*` | `RangerBoxNoiseConfig`（含 `scale_ee_goal`） |
| `env.goal_ee.*` | `EEGoalConfig` |
| `env.base_velocity_controller.*` | `BaseVelocityControllerConfig` |
| `env.ik.*` | `IKConfig`（复用 Go2_arm） |
| `env.domain_rand.*` | `RangerBoxDomainRandConfig` |
| `env.history.*` | `HistoryConfig`（复用 Go2_arm） |

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
| `_compute_obs` | 复用 `_update_history` | 同模式，维度不同 |
| `_compute_reward` | 15 项（含 locomotion） | 13 项（§5.2） |
| `obs_groups_spec` | `{"obs": H_a*76, "critic": H_c*79}` | `{"obs": H_a*44, "critic": H_c*44}` |
| `_init_reward_functions` | 含 legged rewards | 替换为 RangerBox 专用 |
| termination logic | `gravity[:,2] ≤ 0.5` | `gravity_x² + gravity_y² > sin²(1.0)` |

### 7.3 `default_angles` 处理

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

### 7.4 `update_state` 流程

```python
def update_state(self, state: NpEnvState) -> NpEnvState:
    # 1. EE goal trajectory（复用到球面插值逻辑）
    self._update_ee_goal_trajectory()

    # 2. 传感器读取
    linvel = self.get_local_linvel()
    gyro = self.get_gyro()
    gravity = self._get_projected_gravity()   # from framequat
    dof_pos = self.get_dof_pos()
    dof_vel = self.get_dof_vel()
    ee_local_pos, _ = self.get_ee_local_pose()

    # 3. 终止判断
    tilt_sq = gravity[:, 0]**2 + gravity[:, 1]**2
    base_z = self._backend.get_base_pos()[:, 2]
    terminated = (
        (tilt_sq > np.sin(1.0)**2) |
        (base_z < 0.1) |
        self._arm_joint_hard_limits_violated(dof_pos)
    )

    # 4. Reward + Obs
    reward = self._compute_reward(state.info, linvel, gyro, gravity,
                                   dof_pos, dof_vel, ee_local_pos)
    obs = self._compute_obs(state.info, linvel, gyro, gravity,
                            dof_pos, dof_vel, ee_local_pos,
                            self.curr_ee_goal_cart)

    return state.replace(obs=obs, reward=reward, terminated=terminated)
```

### 7.5 Projected Gravity 计算

```python
def _get_projected_gravity(self) -> np.ndarray:
    """World gravity [0,0,-1] expressed in base frame. Returns (N, 3)."""
    quat = self._backend.get_sensor_data("framequat")    # (N, 4) wxyz
    R_wb = np_matrix_from_quat(quat)                     # (N, 3, 3) body→world
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=get_global_dtype())
    gravity_body = np.einsum("nij,j->ni", np.swapaxes(R_wb, 1, 2), gravity_world)
    return gravity_body
```

### 7.6 Scene / Model 解析

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

### 7.7 注册集成

`src/unilab/envs/locomotion/__init__.py` 加一条：

```python
__unilab_registry_modules__ = (
    # ... 现有模块保持不变 ...
    "unilab.envs.locomotion.ranger_box.reach_env",   # [NEW]
)
```

导入时 `@registry.envcfg("RangerBoxReach")` 和 `@registry.env("RangerBoxReach", sim_backend="mujoco")` 自动触发注册。

### 7.8 DR Provider

```python
class RangerBoxReachDRProvider(LocomotionDRProvider):
    """Domain randomization provider for RangerBoxReach.

    Differs from Go2_arm: no velocity command sampling, no gait phase,
    includes BaseVelocityController reset.
    """

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

### 7.9 新增 Dataclass 汇总

```python
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

@dataclass
class RangerBoxControlConfig(ControlConfigBase):
    arm_action_scale: float = 0.0
    arm_kp: list[float] = field(
        default_factory=lambda: [100.0, 110.0, 95.0, 50.0, 50.0, 50.0])
    arm_kd: list[float] = field(
        default_factory=lambda: [3.5, 3.8, 2.5, 1.5, 1.5, 1.5])
    gripper_kp: float = 500.0

@dataclass
class RangerBoxNoiseConfig(NoiseConfig):
    scale_ee_goal: float = 0.01

@dataclass
class RangerBoxDomainRandConfig(DomainRandConfig):
    randomize_ground_friction: bool = False  # freejoint, no wheel-ground contact
```

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
| `set_root_velocity` → `mj_step()` 竞态 | `backend.step()` 在 `mj_step()` 之前检查 qvel，确保不覆盖 |
| arm IK 对移动底盘 numerical stability | `Go2ArmBaseEnv.compute_arm_ik_delta` 已验证 stability，复用 |
| freejoint + push DR 导致 base drift | `base_height` reward + termination 双重防护 |
| 观测噪声注入遗漏维度 | `_compute_raw_obs(add_noise=True)` 覆盖所有 44 维 |
