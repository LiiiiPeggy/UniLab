# RangerBoxReach 训练记录

## Run 2 — 建模/奖励重设计后（2026-08-11）

提交 `f8c3e6e9`：armbasepoint 移到 base body、signed-distance 碰撞奖励、
action 10→9 / obs 41→39、velocimeter linvel、progress+success+stop-near 奖励、
alive=0、goal rejection sampling、history=5、obs_groups 加 critic。

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
- mean reward 3.71 远低于旧 2000-iter 的 7~8，但这是**去除 alive=0.3 之后
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
