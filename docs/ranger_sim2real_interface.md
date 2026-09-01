# Ranger sim2real 命令接口契约

本文档明确仿真与 AgileX Ranger 真机共享的 base 命令语义，以及本批引入的
`RangerCommandAdapter` 如何让两侧行为一致。

---

## 1. 命令链对比

```
Simulation (MuJoCo)                    Real Ranger (ranger_ros)
─────────────────────                  ──────────────────────────────
PPO policy                              PPO policy (部署后)
  │                                       │
  ▼                                       ▼
action[0:3] · base_weight               action[0:3] · base_weight
  │                                       │
  ▼                                       ▼
RangerCommandAdapter                    RangerCommandAdapter
  deadband + hysteresis                   deadband + hysteresis
  motion mode (STOP/ACKER/PARALLEL/SPIN)  motion mode
  vy gate (Ackermann → vy=0)              vy gate
  velocity clip                           velocity clip
  │                                       │
  ▼                                       ▼
geometry_msgs/Twist  (vx, vy, wz)  ←──┐   geometry_msgs/Twist
  │                                     │   │
  ▼                                     │   ▼
BaseVelocityController                  ranger_base_node
  accel limit / tau filter / jerk       mode switch + wheel command
  │                                     │
  ▼                                     ▼
set_root_planar_velocity                motor drivers
```

**两侧共享**（本批目标）：deadband / hysteresis、mode 决策、velocity limit、
（可选）smoothing。差异点只在"命令到执行"的物理层：MuJoCo 用 kinematic
root-velocity（freejoint），真机用轮速+转向经底盘动力学。

## 2. 共享语义

| 项 | 仿真 (`RangerCommandAdapter`) | 真机 (`ranger_base_node`) | 契约 |
|---|---|---|---|
| `linear.x` / `vx` | deadband enter 0.05 / exit 0.03 m/s | 节点内近零停轮 | 一致 |
| `linear.y` / `vy` | `|vy| > 0.08` 进入 PARALLEL；`|vy| < 0.03` 离开；中间保持（Schmitt） | `linear.y != 0` → Parallel mode | 一致 |
| `angular.z` / `wz` | deadband enter 0.05 / exit 0.03 rad/s；SPIN 判定 | 原地旋转模式 | 一致 |
| 速度上限 | `max_lin_vel=1.5`, `max_ang_vel=0.78` | 硬件限制（≈0.785 rad/s 最大转速） | 一致 |
| mode | STOP / ACKERMAN / PARALLEL / SPIN（含 min_mode_duration 防抖） | Ackermann / Parallel / Spin | 一致 |

## 2b. Command state machine

```
PPO output (raw action)
  │  scale to velocity
  ▼
RangerCommandAdapter
  ├─ deadband/hysteresis（vx/vy/wz）
  ├─ mode 判定 + mode 级 hysteresis（Schmitt）+ min_mode_duration 防抖
  └─ vy gate（ACKERMAN 时 vy=0） + velocity clip
  ▼
        ┌──────────────┐
        │     STOP     │  ← 三通道均 deadband 归零
        └──────┬───────┘
               │ 有命令
               ▼
        ┌──────────────┐   |wz|≥0.10 且 |v_xy|<0.05
        │     SPIN     │ ←─────────────────────────┐
        └──────┬───────┘                           │
               │ 非旋转                            │
               ▼                                   │
        ┌──────────────┐   |vy| > 0.08 (enter)     │
        │   ACKERMAN   │ ──────────────────────────┘
        │  (vy forced  │
        │   to 0)      │
        └──────┬───────┘
               │  |vy| < 0.03 (exit)
               ▼
        ┌──────────────┐
        │   PARALLEL   │   (vx + vy + wz 全通过)
        └──────────────┘
```

切换规则：
- **ACKERMAN ↔ PARALLEL**：Schmitt 触发（enter 0.08 / exit 0.03），且受
  `min_mode_duration`（0.2 s）约束——避免 vy 在阈值附近抖动造成频繁切换。
- **STOP/SPIN** 进入即时（响应快），不参与 dwell。

**必须与真机保持一致（部署契约）**：`parallel_enter_vy` / `parallel_exit_vy`、
`min_mode_duration`、`spin_angular/linear_threshold`、deadband enter/exit、
`max_lin_vel` / `max_ang_vel`。这些参数在仿真 adapter 与真机 base_node 中
取值一致，策略部署后行为才不会漂移。

## 3. 残留差异（已知，非本批范围）

1. **执行层**：仿真 kinematic root-velocity（无轮地接触动力学），真机轮速驱动。
   若需更高 fidelity，未来可选 wheel-torque 模型 —— 但这是"另一个阶段"，
   本批刻意保留 freejoint abstraction。
2. **friction**：仿真未建模真机轮胎-地面摩擦特性；不作为第一方案修改。
3. **latency / noise**：本批保持训练契约关闭；真机部署时按需开启。

## 4. 部署路径

1. 训练/评测始终走 `RangerCommandAdapter`（YAML `command_adapter.enable: true`）。
2. 导出策略 → 同一 `RangerCommandAdapter`（Python 端）→ 发布
   `geometry_msgs/Twist`。
3. 关键：**策略输出是 raw action，adapter 在两侧执行相同变换**，因此部署时
   无需改策略。

## 5. 验证

- 单元：`tests/envs/test_ranger_command_adapter.py`（deadband/hysteresis、
  mode 决策、vy gate、wheel viz）。
- 数值：`scripts/manip_loco/benchmark_ranger_base_command.py`
  （raw→adapter→controller→root velocity 统计）。
- 策略级：`eval_ranger_box_reach.py --config-override env.command_adapter.enable=...`
  对比 adapter on/off（见 benchmark 输出 JSON）。
