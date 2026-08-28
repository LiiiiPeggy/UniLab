# RangerBoxReach 训练记录

## Run 4B — 有效 run：planar EXTENDED sampler + 无 arm residual（2026-08-27）

提交 `22f87681`（含 `591497a1` planar sampler、`a789d282` 诊断/validator）。
run dir `logs/rsl_rl_ppo/RangerBoxReach/2026-08-27_20-59-29_mujoco`（`run4b_clean_planar_sampler`）。

配置：300 iters、256 envs × 128 steps、seed 1、save_interval 50、noise/latency/DR=0、
**arm_action_scale=0.0**（YAML 默认 0.01，靠 CLI override 压到 0 并已复核 run_config）。
goal_ee：LOCAL 0.30 3D radial 0.10~0.15；EXTENDED 0.70 **planar**（theta~U(-π,π)、
r_xy~U[0.30,0.70]、dz~U[-0.10,0.10]）；capture 0.15/0.20。

Stage-A 硬指标（deterministic mean-action eval，n=208 eps/ckpt，train-matched arm_action_scale=0.0）：

| iter | rew | LOConce | LOChold | EXTcap | EXThold | EXT p50 | EXT disp | succAfterCap | escape | path/disp | col/jl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 7.34 | 1.000 | 1.000 | 1.000 | 0.920 | 0.055 | 0.499 | 0.920 | 0.014 | .523/.499 | 0/0 |
| 100 | 7.74 | 1.000 | 1.000 | 1.000 | 1.000 | 0.044 | 0.398 | 1.000 | 0.000 | .409/.398 | 0/0 |
| 150 | 7.76 | 0.981 | 0.981 | 1.000 | 1.000 | 0.042 | 0.379 | 1.000 | 0.000 | .389/.379 | 0/0 |
| 200 | 7.82 | 1.000 | 1.000 | 1.000 | 1.000 | 0.051 | 0.401 | 1.000 | 0.000 | .413/.401 | 0/0 |
| 250 | 7.85 | 1.000 | 1.000 | 1.000 | 1.000 | 0.051 | 0.403 | 1.000 | 0.000 | .414/.403 | 0/0 |
| 299 | 7.73 | 1.000 | 0.985 | 1.000 | 1.000 | 0.047 | 0.372 | 1.000 | 0.000 | .383/.372 | 0/0 |

### 结论
- **Stage-A 全达标**：LOCAL hold ≥0.985、EXT capture_entry=1.000（远超 Run 4A 旧 sampler 的 0.595）、
  EXT hold ≥0.920、succ_after_capture ≥0.920、collision=0、joint-limit=0，后期无 hold 退化。
- base → capture → IK → hold 链路正常；机械臂末端可达目标点。
- 训练约 5.6 min wall（334s），final mean reward 7.73。

---

## Run 4A — 旧 3D-radial sampler 基线（2026-08-27）

提交 `f1a7fc97`。run dir `2026-08-27_18-36-07_mujoco`（`run4a_clean_basehandoff`）。
200 iters、256×128、seed 1、arm_action_scale=0.0（CLI override，隔离 base 逼近 + validated-IK）。

EXTENDED 目标是 **true-3D radial**（`extended_radius_range=[0.30,0.70]`）——几何冲突：
SE(2) 锁定 base 只能改 x/y/yaw，~54% 目标 |dz|≥capture_outer(0.20) 构不成 capture。
dz 分箱 benchmark（1008 eps/ckpt）：hold10 随 |dz| 单调崩塌 1.00→1.00→0.89→0.61→0.65→0.18（>0.30）。
possible 子集 cap 0.964/hold10 0.889 vs impossible 子集 0.351/0.349。
Run 4A 在旧 sampler 上：EXT capture_entry 0.595、hold 0.557。旧 ckpt 换新 sampler 后：
cap 1.000 / hold 1.000（model_199）、0.987（model_150）——证实冲突在 sampler 而非策略。

### 结论
- 数据证实几何冲突后才改 sampler（Task 10 gate），LOCAL/capture radii/物理过滤均未动。
- 首启 Run 4B 曾漏带 arm_action_scale override（实际 0.01），违反"不恢复 arm residual"约束，
  已废弃（`logs/rsl_rl_ppo/_archive_invalid_run4b_armscale0p01`）并以完整配置重训。

---

## Run 2 — 建模/奖励重设计后（2026-08-11）

提交 `f8c3e6e9`：armbasepoint 移到 base body、signed-distance 碰撞奖励、
action 10→9 / obs 41→39、velocimeter linvel、progress+success+stop-near 奖励、
alive 奖励置零、goal rejection sampling、history=5、obs_groups 加 critic。

```
Final (iter 999/1000):
  Total steps:        32768000
  Steps per second:   39119
  Mean value loss:    0.0139
  Mean entropy loss:  -6.8991
  Mean reward:        3.71   (best: 45.50)
  Mean episode length: 500.00
  Mean action std:    0.12
  Time elapsed:       00:13:56  (837.7s wall)
  Actor:  195 → [256,128,64] → 9   (obs = 39 × history 5)
  Critic: 195 → [256,128,64] → 1
```

### 结论
- 网络 I/O 正确（195=39×5 obs，9 action），无 critic 警告。
- mean reward 3.71 远低于旧 2000-iter 的 7~8，但这是**去除 alive 0.3 常量奖励之后
  的真实任务 reward**（旧值约 40% 是无信息固定奖励），数值不可直接比较。
- best reward 45.50 显著高于旧 best 34.7，说明新 reward 结构下策略确实
  学到了更强的 reaching 行为。
- action std 0.12、value loss 0.0139、episode 稳定 500，训练收敛无发散。
- 仍缺 episode success rate / EE distance / 碰撞率等硬指标，下轮 eval 补齐。

---

## Run 1 — 修复前 baseline（2026-08-06）

```
Final (iter 1998/1999):
  Mean reward:        8.44 / 7.15
  Mean episode length: 500.00
  Mean action std:    0.10
  Actor: 41 → 10, Critic: 41 → 1  (history=1, 无 critic key 警告)
```
该 run 基于可穿模+会下垂的错误动力学，仅作行为参考。
