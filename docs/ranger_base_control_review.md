# Ranger 底盘控制链路审查（RangerBoxReach）

审查对象：PPO `action[0:3]` → `BaseVelocityController` → `set_root_planar_velocity()`。
生成于 ranger 分支 `aa40fd58` 之后（本批改造前）。

---

## 1. 当前控制流程图

```
PPO action[0:3]  (≈ [-1, 1] 的归一化动作)
   │  × base_weight (= 1 - arm_weight, capture handoff)
   ▼
BaseVelocityController.step(action)
   1. scale      v_cmd = action × [lin_scale, lin_scale, ang_scale]
   2. clip       v_cmd → ±max_lin_vel / ±max_ang_vel
   3. latency    ring buffer 延迟读取（enable_latency，默认关）
   4. accel      dv = clip(v_cmd - v_real, ±max_lin_acc·dt, ±max_ang_acc·dt)
   5. 一阶滤波    v_real += α·(v_target - v_real),  α = dt/(tau+dt)
   6. noise      v_exec = v_real + 噪声（enable_noise，默认关；不进状态累积）
   7. final clip v_exec → ±max_vel_arr
   │
   ▼
apply_velocity()
   8. wheel viz  swerve IK → steering qpos + wheel qvel（enable_wheel_visualization）
   9. 世界转换    v_body=[vx,vy,0], w_body=[0,0,wz] → 经 imu quat 旋转
   ▼
set_root_planar_velocity(v_world_xy, w_world_z, preserve_uncontrolled=True)
   → 写 freejoint qvel（vz/wx/wy 保留不动，SE(2) 锁由 env 负责）
```

## 2. 当前链路结论

| 项 | 现状 | 说明 |
|---|---|---|
| `vx / vy / wz` 含义 | base 系前向 / 侧向 / 偏航角速度，经 imu quat 转世界 | 无 mode 概念 |
| deadband | **无**（命令级） | 仅 `_compute_wheel_ik` 内部有 0.03 m/s 的视觉死区 |
| hysteresis | **无** | — |
| acceleration limit | **有**（`max_lin_acc=1.5` m/s²、`max_ang_acc=3.0` rad/s²） | 按 dt 限 dv |
| jerk limit | **无** | — |
| command mode filtering | **无** | 任意 (vx,vy,wz) 组合都直接执行 |

## 3. 与 AgileX Ranger `/cmd_vel` 接口差异

真实 Ranger 通过 `geometry_msgs/Twist` 接收：`linear.x`(vx)、`linear.y`(vy)、
`angular.z`(wz)，由 `ranger_base_node` 解释为运动模式：

- `linear.y == 0`（≈）：**Ackermann / spinning**；
- `linear.y != 0`：**Parallel mode**（四轮平行横移/斜移）。

当前仿真差异：

1. **无 mode 决策**：仿真无条件执行任意 (vx,vy,wz)；真机会按 vy 触发 parallel mode。
2. **无 deadband / hysteresis**：微小命令直接透传；真机 base_node 对指令近零会停轮，
   且带回差（enter/exit 不同）避免临界抖动。
3. **无 jerk 限制**：仿真允许命令加速度突变（只有 1 阶 accel clip）；真机电机驱动
   有限制加速度变化率的物理约束。
4. **`max_ang_vel` 不匹配**：仿真 1.0 rad/s vs 真机约 0.785 rad/s（详见第 4 节）。
5. **wheel visualization 与真机模式不一致**：仿真用通用 swerve IK（每轮独立转向），
   真机是 Ackermann / Parallel / Spin 四轮转向；开启 viz 时视频中的轮胎运动不代表
   真机行为。

## 4. sim2real 风险

| 风险 | 影响 | 缓解（本批） |
|---|---|---|
| vy 无 mode 门控 | 策略可发出真机不支持的横向命令 | `RangerCommandAdapter` 按 vy 判 mode，Ackermann 下强制 vy=0 |
| 无 deadband/hysteresis | 近零命令抖动 → 真机行为发散 | adapter 增加 per-channel enter/exit 死区（可配） |
| 无 jerk 限制 | 命令突变真机跟不上 | 可选 jerk limiter（默认关，供 ablation） |
| `max_ang_vel` 过大 | 训练出的策略依赖真机达不到的转速 | 下调到 0.75~0.78 rad/s |
| 轮胎 viz 模式不匹配 | 视频误导 | mode-based wheel visualization |

## 5. 结论

当前仿真底盘命令接口是"裸速度指令"，与真机 `/cmd_vel` 语义有三处结构性差异
（无 mode 决策、无死区/滞回、无 jerk 限制）以及若干参数不匹配。本批引入
`RangerCommandAdapter` 作为命令接口层（deadband+hysteresis+mode），并保留
kinematic base controller（accel/tau 滤波不删），把仿真命令语义对齐到真机
`/cmd_vel`。
