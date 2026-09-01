# Ranger base command pipeline 顺序审查

审查对象：PPO base action → `RangerCommandAdapter` → `BaseVelocityController`
→ `set_root_planar_velocity` 的各处理阶段执行顺序。

---

## 1. 当前 pipeline（实际代码）

```
PPO action[0:3] · base_weight          （raw action ∈ [-1,1]）
  │  apply_action 内换算到速度单位
  ▼
RangerCommandAdapter.process(v_cmd)     ── 命令语义层 ──
  ├─ 1. deadband + hysteresis（per-channel Schmitt：vx/vy/wz enter>exit）
  ├─ 2. mode decision（STOP/ACKERMAN/PARALLEL/SPIN，含 mode 级 hysteresis + dwell）
  ├─ 3. vy → 0（ACKERMAN 时）
  └─ 4. velocity clip（max_lin / max_ang）
  ▼
BaseVelocityController.step_from_velocity(v_cmd)  ── 平滑层 ──
  ├─ 2. clip（再次，防御）
  ├─ 3. latency ring（enable_latency，默认关）
  ├─ 4. acceleration limit（dv ≤ max_lin_acc·dt / max_ang_acc·dt）
  ├─ 4b. jerk limit（可选，da/dt bound，默认关）
  ├─ 5. first-order lag（v_real += α·(v_target − v_real)，α = dt/(τ+dt)）
  ├─ 6. noise（enable_noise，默认关，仅执行速度）
  └─ 7. final clip
  ▼
apply_velocity()
  ├─ 8. wheel viz（mode 感知：STOP/ACKERMAN/PARALLEL/SPIN）
  └─ 9. world-frame 旋转 → set_root_planar_velocity
```

## 2. 推荐 pipeline（用户目标）

```
action normalized
  → velocity scaling
  → RangerCommandAdapter（deadband / mode decision / velocity limit）
  → BaseVelocityController
        ├─ first-order lag
        ├─ acceleration limit
        └─ jerk limit
  → set_root_velocity()
```

**唯一差异**在 controller 内部两阶段顺序：

| 阶段 | 当前 | 推荐 |
|---|---|---|
| first-order lag | 最后（accel/jerk 之后） | **最前** |
| acceleration limit | accel → jerk → lag | lag → accel → jerk |
| jerk limit | accel 之后 | accel 之后 |

## 3. 修改风险

把 `first-order lag` 移到 `acceleration limit` 之前会改变速度响应的动态特性：

1. **行为改变**：当前顺序是先对命令做速率限制（accel/jerk），再低通滤波（τ）。
   目标顺序是先滤波再限速率。两者稳态相同，但瞬态不同（先滤波会平滑掉尖峰后
   再限幅，先限幅再滤波则限幅本身也可能被滤波平滑）。对**已训练策略**是
   行为改变 —— Run 4B 是在当前顺序下训练/验证的，重排需要重新训练。
2. **Task 10 已验证**：adapter 在当前顺序下对 model_299 **零性能下降**
   （success 1.000 / ep_len 65.5 / EE p50 0.050），说明当前顺序是行为安全的。
3. **本批约束**：禁止重训。因此**不修改顺序**，仅记录差异。

### 建议

- 保持当前顺序（accel → jerk → lag），与 Run 4B 训练契约一致。
- 若未来追求目标顺序，应与重训一起做，并做 A/B 对比。
- `jerk limit` 当前默认关，属于 ablation knob；开启时位置在 accel 之后
  （限制 da/dt），与目标顺序下"accel → jerk"一致。
