# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Always use `uv run`, never bare `python`.**

| Task | Command |
|------|---------|
| Sync deps | `uv sync` (append `--extra motrix` for Motrix backend) |
| Setup (sync + shell completion) | `make setup` |
| Setup Motrix (Quick Demo) | `make setup-motrix` (`uv sync --extra motrix` + shell completion) |
| ROCm sync | `make sync-rocm` (Linux AMD / ROCm ≥ 7.1) |
| Intel GPU sync | `make sync-xpu` (Intel Arc / Xe) |
| Format + lint | `make format` (ruff format + ruff check --fix) |
| Type check | `make type` (mypy + pyright) |
| Full check | `make check` (format + type) |
| Run tests (fast) | `make test` (`pytest -m "not slow"`) |
| Run tests + coverage | `make test-cov` |
| Run slow tests | `make test-slow` (`pytest -m "slow" -v`) |
| Full CI gate | `make test-all` (check + test-cov) |
| Run a single test | `uv run pytest tests/path/to/test_file.py::test_name -v` |
| Run a test file | `uv run pytest tests/path/to/test_file.py -v` |
| Train (CLI) | `uv run train --algo ppo --task g1_walk_flat --sim mujoco` |
| Eval (CLI) | `uv run eval --algo ppo --task g1_walk_flat --sim mujoco --load-run <run_dir>` |
| Demo (CLI) | `uv run demo <demo_name>` |
| Check docs | `uv run pytest tests/scripts/test_check_docs.py -q` |
| Check repo hygiene | `uv run pytest tests/scripts/test_repo_hygiene.py -q` |
| Clean artifacts | `make clean` |

Available demos: `teaser`, `dance`, `wallflip`, `boxtracking`, `locomani`, `inhandgrasp`, `sharpa_appo_student`.

For mainland China users, set `HF_ENDPOINT=https://hf-mirror.com` before running demos that download checkpoints.

## Architecture

UniLab is a **multi-backend, multi-algorithm** RL training infrastructure. It supports PPO-family (RSL-RL, MLX, APPO, HIM, HORA) and off-policy (SAC, TD3, FlashSAC) algorithms across MuJoCo and Motrix physics backends.

### Layer stack

```
CLI (cli.py) ──► scripts/train_*.py ──► training/ (run, experiment, sim2sim)
                                            │
     ┌──────────────────────────────────────┤
     ▼                                      ▼
  algos/{torch,mlx}/                     envs/{locomotion,motion_tracking,manipulation}/
  (learners, runners, networks)          (task envs: NpEnv subclasses)
                                            │
                                            ▼
                                         base/backend/{mujoco,motrix}/
                                         (SimBackend implementations)
```

- **`scripts/`**: thin entrypoints — assemble Hydra config + call training helpers. No business logic.
- **`training/`**: shared orchestration — run directory resolution, experiment tracking, sim2sim contract enforcement, reward building.
- **`algos/`**: algorithm implementations. Each algorithm has a runner (env loop), learner (gradient steps), and optional staged pipeline for async algorithms.
- **`envs/`**: task environments (locomotion, motion tracking, manipulation). Each env extends `NpEnv` and uses `SimBackend` for physics — never calls backend subclass methods directly.
- **`base/`**: abstract contracts — `NpEnv`, `SimBackend`, `EnvCfg`, registry.
- **`ipc/`**: inter-process communication — shared buffers, weight sync, async runner protocol.

### Key design patterns

**Config first (Hydra + registry).** Training is launched via `scripts/train_*.py` + Hydra `@hydra.main`. The config tree at `conf/<algo>/` composes: algorithm defaults (`config.yaml`) → task owner YAML (`task/<task>/<backend>.yaml`). Backend selection is `task=<task>/<backend>` — `training.sim_backend` is an identity field in the owner YAML, not a standalone override.

**Registry system.** Environments are registered via decorators at import time:

```python
@envcfg("g1_walk_flat")        # register config class
@dataclass
class G1WalkFlatCfg(LocomotionEnvCfg): ...

@env("g1_walk_flat", "mujoco")  # register env class for a backend
class G1WalkFlatEnv(LocomotionEnv): ...
```

Registry packages declare `__unilab_registry_modules__` to list their bootstrap modules. `ensure_registries()` imports these at startup.

**Env contract (`NpEnv`).** `reset()` returns `(obs_dict, info_dict)`. `step(action)` returns `NpEnvState` (obs, reward, terminated, truncated, info). `obs` must always be a `dict[str, np.ndarray]`. `obs_groups_spec` controls which obs groups are active and drives wrapper/learner dimension setup.

**Backend isolation.** Env code only accesses methods declared on `SimBackend` (in `base/backend/base.py`). If a method only exists on `MuJoCoBackend` or `MotrixBackend`, it must first be added to `SimBackend` (can raise `NotImplementedError`). Never call backend subclass methods or use `getattr`/`hasattr` to probe backend capabilities from env code.

### Algorithm landscape

The unified CLI (`uv run train --algo <algo>`) supports: `ppo`, `mlx_ppo`, `appo`, `sac`, `td3`, `flashsac`. APPO supports `--profile hora` for the HORA profile. The following algorithms require running their scripts directly (no CLI support):

| Algorithm | Script | Key characteristics |
|-----------|--------|---------------------|
| PPO (RSL-RL) | `scripts/train_rsl_rl.py` | Sync, RSL-RL based, MuJoCo-only |
| PPO (MLX) | `scripts/train_mlx_ppo.py` | Apple Silicon, macOS-only |
| APPO | `scripts/train_appo.py` | Async, decoupled collector/learner via IPC |
| HIM PPO | `scripts/train_him_ppo.py` | Hybrid inverse model for privileged-to-proprio distillation; script only |
| HORA | `scripts/train_hora_distill.py` | Hybrid off-policy RL with action distillation; script only |
| SAC / TD3 | `scripts/train_offpolicy.py` | Off-policy, replay-buffer based, multi-GPU support |
| FlashSAC | `scripts/train_offpolicy.py` | SAC with Flash Attention, high-throughput |

MuJoCo is the default backend. Motrix requires `uv sync --extra motrix` and uses `motrixsim-core`.

### Test layout

Tests mirror `src/unilab/` structure. Slow tests (tagged `@pytest.mark.slow`) cover training smoke tests, long-running integrations, and backend matrices — skipped in `make test`, run with `make test-slow`. Config tests validate the Hydra composition tree. NaN injection tests (`tests/nan_injection/`) use spawn-based subprocess testing for numerical stability.

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot body XML**（如 `g1.xml`、`go1.xml`）。robot body XML 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play 时，影响策略 I/O / 网络结构的字段必须跨后端一致；不一致即 `CrossBackendIncompatibleError`。详见下方 Sim2Sim 章节。 |

## Commit Conventions

Use Conventional Commits (from [CONTRIBUTING.md](CONTRIBUTING.md)):

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation update
- `style:` formatting only, no logic change
- `refactor:` code refactor
- `test:` test-related change
- `chore:` build or tooling

Do not add new owner logic under `src/unilab/utils/`. Current files there are transition shims scheduled for removal in 0.2.0.

## Sim2Sim 跨后端配置契约

`src/unilab/training/sim2sim.py` 按 dotted path 维护三类字段：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode`。`env.*` 子集对**任一方向**的不对称出现也 fail-closed；`algo` 专属字段目标缺省时按设计跳过（跨算法合法）。
- **WARNING_LIST**：`reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（自由覆盖）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

训练时 `ExperimentTracker.start()` 把上述字段写入 `run_config.json` 的 `contract_snapshot`（不改 checkpoint 格式，旧 run 无 snapshot 时 fallback + warning）；五个 play 入口在建 env 前调用 `resolve_sim2sim_config` 校验，并用 `policy_load_dim_guard` 包裹 checkpoint 加载以把维度不匹配的隐晦报错重抛为显式诊断。设 `training.sim2sim_strict=false` 可把 DENYLIST 差异降级为 warning（默认 `true`）。DENYLIST 字段应通过 task 的 `base.yaml` 共享（范例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`）；跨后端契约审计见 `scripts/audit_sim2sim_contracts.py`。

## Pointers

- PPO: `scripts/train_rsl_rl.py`
- MLX PPO: `scripts/train_mlx_ppo.py`
- APPO: `scripts/train_appo.py`
- SAC / TD3: `scripts/train_offpolicy.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

## Context

- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)
