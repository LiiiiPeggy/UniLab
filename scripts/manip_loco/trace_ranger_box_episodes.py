"""Trace typical episodes of a RangerBoxReach checkpoint for behavioral description.

Loads a checkpoint, runs a single batched rollout (autoreset OFF), classifies
each env by goal type / outcome, and prints a per-step timeline (d, aw, base
displacement, arm motion) for one success EXT, one fail EXT, one LOCAL episode
— a numeric proxy for video inspection (image reading unavailable here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from unilab.envs.common.rotation import np_quat_apply_batched  # noqa: E402
from unilab.training import BackendAdapter, create_env, ensure_registries  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402
from unilab.training.sim2sim import policy_load_dim_guard  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "logs/rsl_rl_ppo/RangerBoxReach/2026-08-27_16-50-34_mujoco"
ITER = int(sys.argv[2]) if len(sys.argv) > 2 else 299
N_ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 300

ensure_registries()
with initialize_config_dir(version_base="1.3", config_dir=str(ROOT_DIR / "conf" / "ppo")):
    cfg = compose(
        config_name="config",
        overrides=[
            "task=ranger_box_reach/mujoco",
            f"algo.num_envs={N_ENVS}",
            "env.noise_config.level=0.0",
            "env.base_velocity_controller.enable_latency=false",
            "env.base_velocity_controller.enable_noise=false",
            "env.base_velocity_controller.enable_wheel_visualization=false",
            "env.domain_rand.randomize_kp=false",
            "env.domain_rand.randomize_kd=false",
            "env.domain_rand.randomize_body_mass=false",
            "env.domain_rand.random_com=false",
            "env.domain_rand.randomize_dof_armature=false",
        ],
    )
env_cfg_override = BackendAdapter(cfg, root_dir="/home/ubuntu/locomani/UniLab").build_task_env_cfg_override()
env = create_env(cfg, num_envs=N_ENVS, env_cfg_override=env_cfg_override)
env.set_autoreset(False)
capture_outer = float(getattr(env._cfg.goal_ee, "capture_outer", 0.20))
capture_inner = float(getattr(env._cfg.goal_ee, "capture_inner", 0.15))
aw_span = max(capture_outer - capture_inner, 1e-6)

device = "cuda" if torch.cuda.is_available() else "cpu"
wrapped = RslRlVecEnvWrapper(env, device=device)
rl_cfg = OmegaConf.to_container(cfg.algo, resolve=True)
train_cfg = normalize_ppo_train_cfg(rl_cfg)
train_cfg.setdefault("runner", {})["logger"] = "none"
from rsl_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
with policy_load_dim_guard(
    env_obs_dim=getattr(wrapped, "num_obs", None),
    env_action_dim=getattr(wrapped, "num_actions", None),
    algo_name="ppo",
):
    runner.load(f"{RUN}/model_{ITER}.pt", map_location=device)
policy = runner.get_inference_policy(device=device)


def ee_world():
    ee_local, _ = env.get_ee_local_pose()
    return env.armbase_pos_world + np_quat_apply_batched(env.armbase_quat_world, ee_local)


obs = wrapped.reset()[0]
env.state.info["steps"][:] = 0
is_local = env._goal_is_local.copy()
base_start = env._backend.get_base_pos()[:, :2].copy()

# One batched rollout, recording per-env timelines.
trace_d: list[list[float]] = [[] for _ in range(N_ENVS)]
trace_aw: list[list[float]] = [[] for _ in range(N_ENVS)]
trace_base: list[list[float]] = [[] for _ in range(N_ENVS)]
trace_arm: list[list[float]] = [[] for _ in range(N_ENVS)]
held = np.zeros(N_ENVS, dtype=bool)
entered = np.zeros(N_ENVS, dtype=bool)
dones = np.zeros(N_ENVS, dtype=bool)
reason = [""] * N_ENVS
q_prev = env.get_arm_dof_pos().copy()
base_prev = base_start.copy()

for t in range(STEPS):
    act = policy(obs)
    obs, rew, dones_now, infos = wrapped.step(act)
    d = np.linalg.norm(ee_world() - env.world_ee_goal, axis=1)
    aw = np.clip((capture_outer - d) / aw_span, 0.0, 1.0)
    base_now = env._backend.get_base_pos()[:, :2]
    q = env.get_arm_dof_pos()
    arm_move = np.linalg.norm(q - q_prev, axis=1)
    dnp = dones_now.detach().cpu().numpy().astype(bool)
    to = infos.get("time_outs")
    to_np = to.detach().cpu().numpy().astype(bool) if to is not None else np.zeros(N_ENVS, bool)
    for i in range(N_ENVS):
        if dnp[i]:
            reason[i] = "timeout" if to_np[i] else "terminal"
            dones[i] = True
    entered |= d <= capture_outer
    held |= env._success_hold.astype(bool)
    for i in range(N_ENVS):
        trace_d[i].append(float(d[i]))
        trace_aw[i].append(float(aw[i]))
        trace_base[i].append(float(np.linalg.norm(base_now[i] - base_start[i])))
        trace_arm[i].append(float(arm_move[i]))
    q_prev = q
    base_prev = base_now
    if all(dones):
        break

print("\nepisode summary:")
for i in range(N_ENVS):
    print(f"  env{i}: {'LOCAL' if is_local[i] else 'EXT '} held={held[i]} "
          f"entered={entered[i]} done={dones[i]} reason={reason[i] or 'running'}")


def show(ep_idx: int, label: str) -> None:
    g0 = np.linalg.norm(
        ee_world() if False else env.world_ee_goal[ep_idx] - (env.armbase_pos_world[ep_idx])
    )
    print(f"\n--- {label} (env{ep_idx}, {'LOCAL' if is_local[ep_idx] else 'EXTENDED'}, "
          f"g0={g0:.2f}m) held={held[ep_idx]} ---")
    print(f"{'step':>5} {'d':>6} {'aw':>5} {'base_disp':>8} {'arm_move':>8}")
    prev = -10
    for t in range(len(trace_d[ep_idx])):
        if t - prev >= 10 or t == len(trace_d[ep_idx]) - 1:
            print(f"{t:>5} {trace_d[ep_idx][t]:6.3f} {trace_aw[ep_idx][t]:5.2f} "
                  f"{trace_base[ep_idx][t]:8.3f} {trace_arm[ep_idx][t]:8.3f}")
            prev = t


ext_ok = [i for i in range(N_ENVS) if not is_local[i] and held[i]]
ext_fail = [i for i in range(N_ENVS) if not is_local[i] and not held[i]]
loc = [i for i in range(N_ENVS) if is_local[i]]
if ext_ok:
    show(ext_ok[0], "EXT success")
if ext_fail:
    show(ext_fail[0], "EXT fail")
if loc:
    show(loc[0], "LOCAL")

env.close()
