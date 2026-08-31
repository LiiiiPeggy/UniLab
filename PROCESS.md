# RangerBoxReach Development Process

RangerBoxReach 是 Ranger 轮式底盘 + Dobot CR10 六轴机械臂 + AG95 夹爪的
移动操作 reaching 任务（MuJoCo，PPO/RSL-RL）。本文档记录开发过程中真正
重要的算法 / 控制 / 物理建模问题与最终方案，以及**不要再重复尝试**的方案。
每个问题给出证据、解决方案、效果与对应 commit。

---

## 1. System Overview

- **硬件模型**：AgileX Ranger（4 轮全向底盘，此处仅用 SE(2) 平面运动）、
  Dobot CR10（6 自由度关节臂）、AG95（两指被动夹爪，不进 action space）。
- **仿真/算法**：MuJoCo、PPO（RSL-RL）、Hydra + registry 配置。
- **Action**：`3 base + 6 arm`。base 走速度控制器（vx/vy/yaw），arm 走
  **validated resolved-rate IK**。
- **Goal 结构**：`LOCAL + EXTENDED` 混合。LOCAL 目标在 arm 可靠捕获半径内
  （纯 IK 可闭合）；EXTENDED 目标靠 base 导航 + arm 后段捕获。
- **关键数字**（训练/评测契约）：`capture_inner=0.15`、`capture_outer=0.20`、
  LOCAL fraction 0.30、EXTENDED planar `r_xy∈[0.30,0.70]`、`dz∈[-0.10,0.10]`、
  `arm_action_scale=0.0`（有效 Run 4B）。

---

## 2. CR10 wrist / joint oscillation

### Problem

早期 CR10 尤其 wrist / joint6 出现明显持续振荡。

### Diagnosis / Experiments

用 `benchmark_ranger_box_physics.py` 系统测过三类手段：

- **gravcomp** `[1, 0.8, 0.5, 0.0]`：几乎不影响 j6 振荡（0.54–1.14 rad），
  只把 first-step drift 减少约 15%。在 freejoint 模型下 gravcomp 很弱。
- **actuator kd** `[1x, 2x, 4x]`：**提高 kd 反而让振荡更严重**（j6 osc → 6.7），
  属数值不稳定，不是修正方向。
- **joint damping** `[1, 5, 10, 20]`：效果明显。

### Solution

CR10 6 个关节全部加 `damping="1"`。

### Result

joint6 稳态振荡约 **0.539 rad → 0.004 rad**；j2–j5 保持静止（osc < 0.013）。

### Do not do

**Do not** 把提高 actuator kd 作为该振荡的主要修复手段——会加剧数值不稳定。

### Relevant commits

- `f2c8b56f` add joint damping=1 to CR10 arm joints — eliminates wrist oscillation

---

## 3. Initial arm pose / singularity

### Problem

早期机械臂初始姿态接近伸展状态：Jacobian conditioning 差、workspace 不适合
reaching、IK controller 难以稳定完成任务。

### Diagnosis

- 单纯把 gravity-sag equilibrium 当作 home 不可用（伸展位近似奇异，且 drift）。
- `find_ranger_box_ready_pose.py` 在全空间搜索**折叠、良态**的 manipulation pose，
  约束：EE 距 armbase 0.30–0.60 m、Jacobian condition < 20、joint-limit margin
  ≥ 0.2 rad、settle drift 小、无碰撞。

### Solution

最终选用 manipulation-ready folded pose：

```
q_ready = [-0.0793, 0.0031, -2.1214, -1.6912, 2.118, 1.1986]
```

关键性质：EE 约 **0.45 m** from armbase、Jacobian condition ≈ **1.6**、joint
margin 较好、无明显碰撞、settle drift 很小。

### Result

`init_arm_pose` 作为 reset 姿态 / obs 基线 / home-return 目标，IK 从良态位出发，
reaching 稳定收敛。

### Do not do

- 不要把 gravity sag equilibrium 直接当 home（reference 展开位会被推到伸展平衡）。
- 复位姿态不是简单"摆个好看角度"，是基于**可操作性**选出的。

### Relevant commits

- `5d29cbb0` manipulation-ready pose + local-EE goals + feasibility gate
- `613b6510` gravity-settle home, eval trajectory+text viz

---

## 4. IK target integration overshoot

> 本文档最重要的一节之一。

### Problem

旧控制逻辑**持续积分 persistent IK target**，导致 target chase-current 持续
领先实际关节，position actuator 不断被推着走 → 过冲 / 发散（final EE p50 达
1.12 m，hold 不住）。

### Experiment

比较三种控制器：

- **A. integrated target**（旧）：对 target 累计积分，常驻领先 q_actual。
- **B. resolved-rate position target**：`q_target = q_actual + gain·dq`，无积分。
- **C. resolved-rate + velocity damping**：B + `-kv·qdot`。

关键结果：

- **B 最好**（EE 收敛后保持，final p50 1.12 m → 0.063 m，无发散）。
- velocity damping `kv=0.05 / 0.2` **没有改善成功率，反而降低**。

### Solution

最终配置：

```
controller_mode = resolved_rate
damping   = 0.02
gain      = 1.5
dq_clip   = 0.3
velocity_damping = 0.0
```

核心：`q_target` 直接基于 `q_actual + arm_weight·gain·dq_cmd`（锚定实际位），
**不累计 persistent target**；soft-limit clip 作为 joint-limit 防积分饱和。

### Do not do

- **Do not** 恢复 integrated target 控制（persistent target lead）。
- **Do not** 在无新 ablation 的情况下加 velocity damping。

### Relevant commits

- `796c4f17` resolved-rate IK target controller — fixes overshoot/divergence
- `1bef5127` IK anti-windup — actual-q anchored target, tracking bound, dq blocking

---

## 5. Arm capture radius benchmark

### Problem

不能主观设定"机械臂可以 reach 0.3~0.4 m"——必须实测。

### Diagnosis / Evidence

`benchmark_ranger_box_ik_capture_radius.py` 分 bin 测量纯 IK 的
`once10 / hold10`：

- **0.10–0.15 m**：`once10 / hold10 > 0.95`（可靠捕获区）。
- **0.15–0.20 m**：成功率明显下降。
- 0.20–0.40 m：进一步衰减。

### Solution

据此定义：

```
capture_inner = 0.15   (fully engaged)
capture_outer = 0.20   (blend 边界)
LOCAL radius  = 0.10~0.15   (可靠捕获区内)
```

### Result

LOCAL 目标全部落在可靠捕获区内，IK 能稳定闭合。

### Do not do

不要拍脑袋改 capture radii；它们是经验 benchmark 得到的契约值。

### Relevant commits

- `a530f7c8` capture-region task — local/extended goals, capture gate; Pure IK gate PASSES
- `5d29cbb0` feasibility gate（LOCAL 目标 IK 可行性过滤）

---

## 6. Run 3 reward exploitation

### Problem

Run 3：training reward 持续增加，但 LOCAL hold / EXT hold 持续下降。典型现象是
策略反复"进入 10 cm → 离开 → 再进入"，而不是连续 hold。

### Root Cause

- 旧 `success_10cm / success_05cm` 是 **per-step reward**；
- 同时 **positive ee_distance reward** 可持续累积；
- 成功 hold 后 episode 提前结束，反而**失去后续累积奖励**。

⇒ 策略学会"延长 episode + 反复 crossing"，而不是完成任务。

### Solution

- `success_once_10cm / success_once_05cm` 改为 **first-entry event reward**；
- 新增 `success_hold_10cm` **terminal event bonus**；
- **event rewards 不乘 ctrl_dt**（continuous 才积分）；
- `ee_distance` 置 0（hovering 不再刷分）；
- 保留 `ee_progress` 与 negative L2 distance shaping。

### Sanity Check

`reward_return_sanity_check.py`（A 快速 hold / B crossing / C linger）：

```
hold    ≈ 7.03
crossing ≈ 1.94
linger  ≈ -0.04
```

必须满足 `hold > crossing > linger`。

### Result

奖励与终止态对齐：hold 10cm 0.5s 是唯一最优解，crossing 不再有收益。

### Do not do

- **Do not** 恢复 per-step success reward。
- **Do not** 用 positive per-step near-goal reward（会让 lingering 有利可图）。

### Relevant commits

- `f1a7fc97` Run-4A reward/base-handoff/gripper fixes（含 sanity check）
- `8c654593` run-3 evaluation diagnostics

---

## 7. Base 在 capture 内继续移动

### Problem

Run 3 中即使 EE 已靠近目标，base 仍继续移动；LOCAL 甚至出现约 **0.6 m**
base displacement，机械臂刚进入目标区域又被底盘带出去。

### Solution

引入 `arm_weight`，定义 **smooth complementary handoff**：

```
base_weight = 1 - arm_weight
arm_weight  = clip((capture_outer - ee_error) / (capture_outer - capture_inner), 0, 1)

far:       base_weight ≈ 1, arm_weight ≈ 0
capture:   base 渐停, arm 渐接管
near:      base_weight ≈ 0, arm_weight ≈ 1
```

base 命令 = 原命令 × base_weight；RL arm residual 也 × arm_weight。

### Result

Run 4A：LOCAL base displacement ≈ **0.001 m**，LOCAL hold 恢复 ≈ 1.0。

### Do not do

- 不要恢复 "base always fully active"。
- 不要简单 hard switch（`if d<thr: base=0`）——当前 smooth 互补 handoff 是
  验证过的版本，避免不连续。

### Relevant commits

- `f1a7fc97`（base/arm handoff）

---

## 8. RL arm residual 干扰稳定 IK

### Problem

Pure IK 下 LOCAL hold ≈ 1.0；加入 PPO arm residual 后 Run 3 LOCAL hold 明显
下降——RL residual 在干扰已经验证稳定的 IK。

### Solution

- 首先 residual × `arm_weight`（capture 外 residual=0）；
- Run 4A / Run 4B 进一步 **`arm_action_scale = 0.0`**：PPO 只负责 base
  navigation，validated IK 负责最后 reaching。

### Result

进入 capture 后 `succ_after_capture ≈ 0.94–1.0`。当前阶段无必要恢复 arm
residual。

### Do not do

- **Do not** 默认把 `arm_action_scale` 恢复为 0.01。
- 未来若重新启用，必须单独做 `0 / 0.002 / 0.003 / ...` 的 ablation。

### Relevant commits

- `f1a7fc97`（residual × arm_weight、Run 4A scale=0）
- `a789d282` eval `--arm-action-scale` 契约旗标

---

## 9. EXTENDED 3D radial geometry conflict

> 目前最有价值的问题之一。

### Problem

原 EXTENDED goal 用 **3D random unit direction × radius 0.30–0.70 m**。但
Ranger base 是 SE(2)（只能改 XY + yaw，**不能解决 vertical dz**），而 arm 在
`d > capture_outer` 时保持 ready pose。若 `|dz| ≥ 0.20 m`，base-only 理论上
永远无法把目标送入 `capture_outer=0.20 m`。

### Evidence（benchmark）

`benchmark_ranger_box_extended_geometry.py`（各 1008 eps，EXT-only）：

- 旧 sampler 约 **53.7%（m199）/ 57.2%（m150）** 的 EXTENDED goals 满足
  `|dz| ≥ 0.20 m`（impossible）。
- possible 子集 `capture ≈ 0.964`；impossible 子集 `capture ≈ 0.351`。
- `|dz| > 0.30 m` 时 capture / hold ≈ **0.18**。

⇒ 这不是"PPO 没训练够"，而是**任务定义与 base DOF 存在几何冲突**。

### Solution

- LOCAL：保持 3D radial sampling。
- EXTENDED：改为 **planar** ——

```
theta ~ U(-pi, pi)
r_xy  ~ U(0.30, 0.70)
dx = r_xy·cos(theta),  dy = r_xy·sin(theta)
dz   ~ U(-0.10, 0.10)          (bounded vertical offset)
```

即 planar base displacement + bounded vertical offset，仍只做物理过滤
（floor/chassis），不加 IK feasibility。

### Result

- 新 sampler：`base_only_capture_impossible fraction = 0`。
- 旧 Run 4A checkpoint **无需重训**，在新 sampler 下 EXT capture：**0.595 →
  1.000**。
- 最终 Run 4B：EXT capture ≈ 1.0、EXT hold ≈ 1.0。

### Do not do

**Do not** 恢复 3D radial EXTENDED sampler。

### Relevant commits

- `591497a1` planar EXTENDED goal sampler with bounded |dz|
- `a789d282` goal-sampler validator（10k+ goals 统计验证）

---

## 10. AG95 passive linkage jitter

### Problem

AG95 夹爪在 reaching 视频中持续小幅乱动。

### Diagnosis

action 是 9D 且**不含 gripper**，所以不是 policy 学出来的。实际来源：master
gripper actuator + 多个 passive joints + hard equality constraints，passive
joints **缺 damping**，arm 一动就振荡。

### Solution

- `gripper_hold_position` 固定开度（gripper 明确不进 action space）；
- 8 个 gripper joints 加 `damping="0.05"`。

### Result

动态摆动 benchmark：gripper `rms_qdot` 约 **0.032 → 0.0022**。

### Do not do

当前 reaching task 不需要增加 gripper action dimension。

### Relevant commits

- `f1a7fc97`（AG95 damping + gripper_hold_position）
- `046b48af` gripper kp 500→50 / kd 10→5

---

## 11. CR10 actuator saturation（Known Issue / Future Work）

> 此项**未宣称解决**，仅已 benchmark / diagnostic。

### Problem

ready pose 下部分关节静态 actuator force 已接近 / 达到 force limit，尤其
**j3 / j4 / j5 / j6**。

### 现状

`benchmark_ranger_box_arm_torque_margin.py` 已量化饱和比例，但**未正式修改
force limits**。

### Do not do

**Do not** 随便放大 actuator force limits。未来需综合：
`qfrc_bias`（重力/惯性补偿需求）、required hold torque、真实 CR10 actuator
capability、dynamic torque margin 后再决定。

### Relevant commits

- `49b88755` gravcomp + force sensors（`jointactuatorfrc` 诊断通道）

---

## 12. Run 演进

| Run | 关键问题 | 核心结果 |
|---|---|---|
| Run 3 | reward exploitation（per-step success → crossing）；base 进 capture 不停；arm residual 干扰 IK | training reward 升但 LOCAL/EXT hold 降；LOCAL base displacement ~0.6 m |
| Run 4A | 修复 reward（event）、base/arm handoff（smooth blend）、arm residual 隔离（scale=0）、AG95 damping | LOCAL hold = 1.0、succ_after_capture ≈0.94；**发现 EXT 3D-radial 几何瓶颈**（EXT cap 0.595） |
| Run 4B | EXTENDED 改 planar sampler（bounded dz） | EXT capture ≈1.0、EXT hold ≈1.0；LOCAL hold ≥0.985、col/jl=0 |

（完整指标见 `debug.md`；此处只保留能说明架构演进的数字。）

---

## 13. Key Do-nots 汇总

1. 不要用提高 actuator kd 修 CR10 振荡。
2. 不要把 gravity sag equilibrium 当 home（用可操作性选 q_ready）。
3. 不要恢复 integrated persistent IK target。
4. 不要无 ablation 加 velocity damping。
5. 不要恢复 per-step success reward 或 positive per-step near-goal reward。
6. 不要用 hard-switch 替代 smooth base/arm handoff。
7. 不要默认恢复 `arm_action_scale=0.01`（必须重新 ablation）。
8. 不要恢复 3D radial EXTENDED sampler。
9. 不要随便放大 CR10 actuator force limits。
10. 不要给 reaching task 加 gripper action dimension。
