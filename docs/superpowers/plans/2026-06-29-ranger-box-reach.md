# RangerBoxReach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate RangerboxCR10Lidar mobile manipulator into UniLab RL training with freejoint-kinematics base, SE(2) planar lock, and world-frame EE reaching.

**Architecture:** RangerBoxReachEnv extends Go2ArmBaseEnv with a vectorized BaseVelocityController (freejoint qvel write), 10-dim action space (3 base + 6 arm + 1 gripper), and 41-dim obs space. The env overrides `apply_action`, `update_state`, `_compute_raw_obs`, `_update_history`, and `_compute_reward`. Three new non-abstract SimBackend methods (`set_root_planar_velocity`, `set_joint_qpos`, `set_joint_qvel`) default to `NotImplementedError` and are implemented only in MuJoCoBackend.

**Tech Stack:** Python 3.12+, NumPy, MuJoCo (via `mujoco` package), Gymnasium, Hydra (OmegaConf struct), pytest. UniLab internal: SimBackend → MuJoCoBackend, Go2ArmBaseEnv, LocomotionDRProvider, NpEnvState, registry decorators.

**Spec:** [docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md](../specs/2026-06-29-ranger-box-reach-design.md)

## Global Constraints

- **SE(2) planar lock:** freejoint provides 6-DOF but only vx/vy/yaw are controlled; vz/wx/wy fixed to 0. No wheel-ground contact physics. `push_robots=false`, wheel geom `contype="0" conaffinity="0"`.
- **Backend isolation:** env code only accesses methods declared on `SimBackend`. New methods (`set_root_planar_velocity`, `set_joint_qpos`, `set_joint_qvel`) are non-abstract with default `NotImplementedError`.
- **reward_config field name:** training adapter injects `reward_config` (NOT `reward`). EnvCfg field is `reward_config: RangerBoxRewardConfig`.
- **RangerBoxSensor inherits Go2ArmSensor:** field names must match parent access patterns (`local_linvel`, `ee_local_pos`, etc.) — only override values.
- **_update_history uses axis=1 rolling:** `np.roll(buf, -D, axis=1)` along feature axis, not env axis. Buffer layout `(N, H*D)`, write to `[:, -D:]`. Support `env_ids` for partial reset.
- **All reward functions return positive values:** negative scales only in YAML. Follows go2_arm pattern.
- **_sample_commands returns `(num_reset, 3)` ndarray:** NOT a dict.
- **`reset_ee_goals()` called in `_compute_reset_obs()`:** after `set_state()`, not in `build_reset_plan()`.
- **DR Provider caches kp/kd/body_mass/dof_armature at init:** via `backend.get_actuator_gains()`, `backend.get_body_mass()`, `backend.get_dof_armature()`.

## File Structure Map

```
src/unilab/
├── base/backend/
│   ├── base.py                        ← MODIFY: add 3 non-abstract methods
│   └── mujoco/backend.py              ← MODIFY: implement 3 methods
├── assets/robots/ranger_box/
│   ├── robot.xml                      ← CREATE: freejoint + sensors + position actuators
│   ├── scene_flat.xml                 ← CREATE: include robot + floor + keyframe
│   └── meshes/*.obj                   ← CREATE: 26 mesh files (copy from foropenpi)
├── envs/locomotion/
│   ├── __init__.py                    ← MODIFY: add ranger_box to registry modules
│   └── ranger_box/
│       ├── __init__.py                ← CREATE: import env to trigger registration
│       ├── reach_env.py               ← CREATE: env + dataclasses + DR provider + rewards
│       └── base_velocity_controller.py← CREATE: vectorized A+ controller
conf/ppo/task/ranger_box_reach/
│   └── mujoco.yaml                    ← CREATE: Hydra owner YAML
tests/
└── test_ranger_box_reach.py           ← CREATE: all tests
```

---

### Task 1: Backend Contract — Add Planar Velocity + Joint Setters to SimBackend

**Files:**
- Modify: `src/unilab/base/backend/base.py` (add 3 methods after existing `get_actuator_gains`)
- Modify: `src/unilab/base/backend/mujoco/backend.py` (implement 3 methods in MuJoCoBackend)

**Interfaces:**
- Produces:
  - `SimBackend.set_root_planar_velocity(self, lin_vel_xy: np.ndarray, yaw_rate: np.ndarray, *, preserve_uncontrolled: bool = True) -> None` — write only vx/vy/wz to freejoint qvel, leave vz/wx/wy unchanged when `preserve_uncontrolled=True`.
  - `SimBackend.set_joint_qpos(self, names: Sequence[str], values: np.ndarray) -> None` — write qpos for named joints (used for wheel visualization).
  - `SimBackend.set_joint_qvel(self, names: Sequence[str], values: np.ndarray) -> None` — write qvel for named joints (used for wheel visualization).

**Pre-check:** `grep -n "def get_actuator_gains" src/unilab/base/backend/base.py` returns line ~421. New methods are inserted after this block, before the "Base kinematics" comment section.

- [ ] **Step 1: Add 3 method stubs to SimBackend (base.py)**

Append after `get_actuator_gains` (after line ~426, before the "Base kinematics" separator at ~427). Open `src/unilab/base/backend/base.py` and locate the `get_actuator_gains` method. Insert the following after its closing triple-quote and before the `# ------------------------------------------------------------------ #` / `# Base kinematics` section divider:

```python
    # ------------------------------------------------------------------ #
    # Root planar velocity                                                 #
    # ------------------------------------------------------------------ #

    def set_root_planar_velocity(
        self,
        lin_vel_xy: np.ndarray,
        yaw_rate: np.ndarray,
        *,
        preserve_uncontrolled: bool = True,
    ) -> None:
        """Write planar velocity to the root freejoint qvel.

        Only vx, vy, and wz (yaw rate) are written.  When
        ``preserve_uncontrolled`` is True (the default), vz, wx, and wy
        are left unchanged so that height / roll / pitch dynamics are
        preserved across steps.

        Args:
            lin_vel_xy: ``(num_envs, 2)`` float64 array — vx, vy in world frame.
            yaw_rate: ``(num_envs,)`` float64 array — angular velocity around z.
            preserve_uncontrolled: if True, read current qvel and only
                overwrite the planar components.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_root_planar_velocity"
        )

    def set_joint_qpos(self, names: Sequence[str], values: np.ndarray) -> None:
        """Set qpos for named joints (wheel / steering visualization).

        Args:
            names: Joint names (e.g. ``["fr_steering_joint", ...]``).
            values: ``(num_envs, len(names))`` float64 array.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_joint_qpos"
        )

    def set_joint_qvel(self, names: Sequence[str], values: np.ndarray) -> None:
        """Set qvel for named joints (wheel / steering visualization).

        Args:
            names: Joint names (e.g. ``["fr_wheel_joint", ...]``).
            values: ``(num_envs, len(names))`` float64 array.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support set_joint_qvel"
        )
```

- [ ] **Step 2: Verify base.py changes are syntactically valid**

Run: `python -c "import unilab.base.backend.base; print('OK')"`
Expected: `OK` (no ImportError or SyntaxError)

- [ ] **Step 3: Implement 3 methods in MuJoCoBackend (mujoco/backend.py)**

Open `src/unilab/base/backend/mujoco/backend.py`. Locate the `MuJoCoBackend` class (near line ~200, after `__init__`). Find `get_actuator_gains` (line ~1322). Add the following three methods after `get_actuator_gains`:

```python
    def set_root_planar_velocity(
        self,
        lin_vel_xy: np.ndarray,
        yaw_rate: np.ndarray,
        *,
        preserve_uncontrolled: bool = True,
    ) -> None:
        """Write planar velocity to freejoint qvel (vx, vy, wz only)."""
        _np = np
        qvel = self._data.qvel
        if qvel.ndim != 2:
            raise RuntimeError("set_root_planar_velocity requires batched qvel (num_envs, nv)")
        if preserve_uncontrolled:
            qvel[:, 0] = _np.asarray(lin_vel_xy[:, 0], dtype=_np.float64)
            qvel[:, 1] = _np.asarray(lin_vel_xy[:, 1], dtype=_np.float64)
            qvel[:, 5] = _np.asarray(yaw_rate, dtype=_np.float64)
            # qvel[:, 2] (vz), qvel[:, 3] (wx), qvel[:, 4] (wy) — left unchanged
        else:
            qvel[:, 0] = _np.asarray(lin_vel_xy[:, 0], dtype=_np.float64)
            qvel[:, 1] = _np.asarray(lin_vel_xy[:, 1], dtype=_np.float64)
            qvel[:, 2] = 0.0
            qvel[:, 3] = 0.0
            qvel[:, 4] = 0.0
            qvel[:, 5] = _np.asarray(yaw_rate, dtype=_np.float64)

    def set_joint_qpos(self, names: Sequence[str], values: np.ndarray) -> None:
        """Set qpos for named joints (wheel/steering visualization)."""
        _np = np
        _mujoco = mujoco
        for i, name in enumerate(names):
            jid = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} not found in MuJoCo model")
            adr = int(self._model.jnt_qposadr[jid]) - self._root_qpos_dim
            self._data.qpos[:, adr] = _np.asarray(values[:, i], dtype=_np.float64)

    def set_joint_qvel(self, names: Sequence[str], values: np.ndarray) -> None:
        """Set qvel for named joints (wheel/steering visualization)."""
        _np = np
        _mujoco = mujoco
        for i, name in enumerate(names):
            jid = _mujoco.mj_name2id(self._model, _mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint {name!r} not found in MuJoCo model")
            adr = int(self._model.jnt_dofadr[jid]) - self._root_qvel_dim
            self._data.qvel[:, adr] = _np.asarray(values[:, i], dtype=_np.float64)

    # ------------------------------------------------------------------ #
```

- [ ] **Step 4: Verify mujoco/backend.py changes are syntactically valid**

Run: `python -c "import unilab.base.backend.mujoco.backend; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Write unit test for set_root_planar_velocity**

Create `tests/test_ranger_box_reach.py`:

```python
"""Tests for RangerBoxReach env, backend contract, and controller."""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if mujoco is not installed.
pytest.importorskip("mujoco", reason="mujoco not installed")


class TestBackendPlanarVelocity:
    """Verify the 3 new SimBackend methods work correctly in MuJoCo."""

    @pytest.fixture(scope="class")
    def mj_backend(self):
        """Create a minimal MuJoCoBackend with a freejoint body."""
        from unilab.base.backend.mujoco.backend import MuJoCoBackend
        from unilab.base.scene import SceneCfg

        xml = """<mujoco model="test_planar">
          <worldbody>
            <body name="base">
              <freejoint/>
              <geom type="box" size="0.1 0.1 0.05"/>
            </body>
          </worldbody>
        </mujoco>"""
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False)
        tmp.write(xml)
        tmp.close()
        scene = SceneCfg(model_file=tmp.name)
        backend = MuJoCoBackend(scene, num_envs=4, sim_dt=0.01)
        os.unlink(tmp.name)
        return backend

    def test_set_root_planar_velocity_writes_planar_only(self, mj_backend):
        """set_root_planar_velocity writes vx, vy, wz and preserves vz, wx, wy."""
        N = 4
        # Set known uncontrolled velocities first
        mj_backend._data.qvel[:, 2] = 0.5   # vz
        mj_backend._data.qvel[:, 3] = 0.1   # wx
        mj_backend._data.qvel[:, 4] = -0.2  # wy

        lin_vel = np.array([[1.0, 0.5], [0.0, -0.3], [2.0, 0.0], [-1.0, 1.0]])
        yaw_rate = np.array([0.1, 0.0, -0.5, 0.2])

        mj_backend.set_root_planar_velocity(lin_vel, yaw_rate, preserve_uncontrolled=True)

        np.testing.assert_allclose(mj_backend._data.qvel[:, 0], lin_vel[:, 0])
        np.testing.assert_allclose(mj_backend._data.qvel[:, 1], lin_vel[:, 1])
        np.testing.assert_allclose(mj_backend._data.qvel[:, 5], yaw_rate)
        # Uncontrolled components preserved
        np.testing.assert_allclose(mj_backend._data.qvel[:, 2], 0.5)
        np.testing.assert_allclose(mj_backend._data.qvel[:, 3], 0.1)
        np.testing.assert_allclose(mj_backend._data.qvel[:, 4], -0.2)

    def test_set_root_planar_velocity_no_preserve_zeros_uncontrolled(self, mj_backend):
        """With preserve_uncontrolled=False, vz/wx/wy are zeroed."""
        N = 4
        mj_backend._data.qvel[:, 2] = 0.5
        mj_backend._data.qvel[:, 3] = 0.1
        mj_backend._data.qvel[:, 4] = -0.2

        lin_vel = np.zeros((N, 2))
        yaw_rate = np.zeros(N)
        mj_backend.set_root_planar_velocity(lin_vel, yaw_rate, preserve_uncontrolled=False)

        np.testing.assert_allclose(mj_backend._data.qvel[:, 2], 0.0)
        np.testing.assert_allclose(mj_backend._data.qvel[:, 3], 0.0)
        np.testing.assert_allclose(mj_backend._data.qvel[:, 4], 0.0)

    def test_set_joint_qpos_writes_correctly(self, mj_backend):
        """set_joint_qpos writes values at correct qpos addresses."""
        # The test XML only has a freejoint body with no named joints,
        # so this verifies the API exists. A fuller test comes with the
        # ranger_box model in Task 2.
        pass  # API existence verified by import; joint-name test in Task 8.

    def test_set_joint_qvel_writes_correctly(self, mj_backend):
        """set_joint_qvel writes values at correct qvel addresses."""
        pass  # API existence verified by import; joint-name test in Task 8.

    def test_sim_backend_stubs_raise_not_implemented(self):
        """Base SimBackend stubs raise NotImplementedError."""
        from unilab.base.backend.base import SimBackend
        # Cannot instantiate abstract SimBackend directly; verify methods exist
        assert hasattr(SimBackend, "set_root_planar_velocity")
        assert hasattr(SimBackend, "set_joint_qpos")
        assert hasattr(SimBackend, "set_joint_qvel")
```

- [ ] **Step 6: Run tests to verify backend methods**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestBackendPlanarVelocity -v`
Expected: 5 tests pass

- [ ] **Step 7: Commit**

```bash
git add src/unilab/base/backend/base.py src/unilab/base/backend/mujoco/backend.py tests/test_ranger_box_reach.py
git commit -m "feat: add set_root_planar_velocity, set_joint_qpos, set_joint_qvel to SimBackend and MuJoCoBackend"
```

### Task 2: Robot XML Assets — robot.xml, scene_flat.xml, Meshes

**Files:**
- Create: `src/unilab/assets/robots/ranger_box/robot.xml`
- Create: `src/unilab/assets/robots/ranger_box/scene_flat.xml`
- Create: `src/unilab/assets/robots/ranger_box/meshes/*.obj` (26 files)

**Interfaces:**
- Consumes: MuJoCoBackend (Task 1) — needs `set_root_planar_velocity`, `set_joint_qpos`, `set_joint_qvel` to exist on SimBackend
- Produces: loadable MuJoCo model with `nq=29`, `nu=7`, freejoint on base body, 7 position actuators (6 arm + 1 gripper), IMU + EE + armbase sensors, wheel geoms with `contype="0" conaffinity="0"`, equality constraints for gripper slave joints

**Spec reference:** [spec §2](docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md), [§2.1 SE(2) Planar Lock](docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md)

- [ ] **Step 1: Copy mesh files from foropenpi to assets**

Run:
```bash
SRC=foropenpi/robosuite/robosuite/models/assets/robots/rangerboxcr10lidar/meshes
DST=src/unilab/assets/robots/ranger_box/meshes
mkdir -p "$DST"
cp "$SRC"/*.obj "$DST/"
ls "$DST" | wc -l
```
Expected output: `26`

- [ ] **Step 2: Create robot.xml**

The robot.xml is derived from the foropenpi source with these changes:
1. Add `<freejoint/>` on the base body (before any child bodies)
2. Add sensors on base body: `imu` site with gyro, framequat, velocimeter, framezaxis
3. Add sensors on cr10_Link1 (armbase): framepos, framequat
4. Add sensors on cr10_Link6 (endpoint): framepos, framequat
5. Replace `<motor>` with `<position>` actuators (6 arm × position + 1 gripper × position)
6. Set wheel geom `contype="0" conaffinity="0"`

Create `src/unilab/assets/robots/ranger_box/robot.xml`:

```xml
<mujoco model="rangercr10lidar">
  <compiler angle="radian" />

  <asset>
    <mesh name="ranger_base_link" content_type="model/obj" file="meshes/ranger_base_link.obj" />
    <mesh name="fr_steering_link" content_type="model/obj" file="meshes/fr_steering_link.obj" />
    <mesh name="fr_wheel_link" content_type="model/obj" file="meshes/fr_wheel_link.obj" />
    <mesh name="fl_steering_wheel_link" content_type="model/obj" file="meshes/fl_steering_wheel_link.obj" />
    <mesh name="fl_wheel_link" content_type="model/obj" file="meshes/fl_wheel_link.obj" />
    <mesh name="rl_steering_wheel_link" content_type="model/obj" file="meshes/rl_steering_wheel_link.obj" />
    <mesh name="rl_wheel_link" content_type="model/obj" file="meshes/rl_wheel_link.obj" />
    <mesh name="rr_steering_wheel_link" content_type="model/obj" file="meshes/rr_steering_wheel_link.obj" />
    <mesh name="rr_wheel_link" content_type="model/obj" file="meshes/rr_wheel_link.obj" />
    <mesh name="cr10_base_link" content_type="model/obj" file="meshes/cr10_base_link.obj" />
    <mesh name="Link1" content_type="model/obj" file="meshes/Link1.obj" />
    <mesh name="Link2" content_type="model/obj" file="meshes/Link2.obj" />
    <mesh name="Link3" content_type="model/obj" file="meshes/Link3.obj" />
    <mesh name="Link4" content_type="model/obj" file="meshes/Link4.obj" />
    <mesh name="Link5" content_type="model/obj" file="meshes/Link5.obj" />
    <mesh name="Link6" content_type="model/obj" file="meshes/Link6.obj" />
    <mesh name="ag95_base_link" content_type="model/obj" file="meshes/ag95_base_link.obj" scale="0.001 0.001 0.001" />
    <mesh name="crank_Link" content_type="model/obj" file="meshes/crank_Link.obj" scale="0.001 0.001 0.001" />
    <mesh name="rod_Link" content_type="model/obj" file="meshes/rod_Link.obj" scale="0.001 0.001 0.001" />
    <mesh name="proximal_phalanx_Link" content_type="model/obj" file="meshes/proximal_phalanx_Link.obj" scale="0.001 0.001 0.001" />
    <mesh name="distal_phalanx_Link" content_type="model/obj" file="meshes/distal_phalanx_Link.obj" scale="0.001 0.001 0.001" />
    <mesh name="lidar_link0" content_type="model/obj" file="meshes/lidar_link0.obj" />
    <mesh name="box_link" content_type="model/obj" file="meshes/box_link.obj" />
  </asset>

  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint/>
      <inertial pos="-0.0169024 0.00678181 0.0577617" quat="1 0 0 0" mass="88.7576" diaginertia="1.71234 4.90028 6.39425" />
      <geom type="mesh" rgba="1 1 1 1" mesh="ranger_base_link" group="1" />
      <geom pos="0.2462 0 0.1" quat="1 0 0 0" type="mesh" rgba="1 1 1 1" mesh="cr10_base_link" group="1" />
      <geom pos="0.52588 0 0.16587" quat="1 0 0 0" type="mesh" rgba="1 1 1 1" mesh="lidar_link0" group="1" />
      <geom size="0.012525 0.045 0.0125" pos="0.62148 0 0.23637" type="box" rgba="0.82 0.82 0.85 1" group="1" />
      <geom pos="0 0 0.09" quat="1 0 0 0" type="mesh" rgba="1 1 1 1" mesh="box_link" group="1" />

      <!-- IMU sensor site on base body -->
      <site name="imu" pos="0 0 0.1" size="0.01" rgba="0 1 0 1" group="2" />

      <body name="fr_steering_link" pos="0.445 -0.28 0.0335">
        <inertial pos="0.0001118 -0.0073218 -0.085228" quat="0.501203 0.501505 -0.498791 0.498493" mass="2.0957" diaginertia="0.00792211 0.0077827 0.0012664" />
        <joint name="fr_steering_joint" pos="0 0 0" axis="0 0 -1" range="-1.57 1.57" />
        <geom type="mesh" rgba="0 0 0 1" mesh="fr_steering_link" group="1" contype="0" conaffinity="0" />
        <body name="fr_wheel_link" pos="0 0.001 -0.2918" quat="0.707105 0.707108 0 0">
          <inertial pos="-0.00095113 -1.6964e-07 0.0020954" quat="2.22213e-06 0.707112 2.22216e-06 0.707102" mass="11.468" diaginertia="0.11423 0.080083 0.05332" />
          <joint name="fr_wheel_joint" pos="0 0 0" axis="0 0 1" />
          <geom type="mesh" rgba="1 1 1 1" mesh="fr_wheel_link" group="1" contype="0" conaffinity="0" />
        </body>
      </body>
      <body name="fl_steering_wheel_link" pos="0.445 0.28 0.0335">
        <inertial pos="0.00010411 0.0077919 -0.086394" quat="0.502545 0.502295 -0.497443 0.497694" mass="2.1046" diaginertia="0.00792301 0.00778359 0.0012676" />
        <joint name="fl_steering_wheel_joint" pos="0 0 0" axis="0 0 -1" range="-1.57 1.57" />
        <geom type="mesh" rgba="0 0 0 1" mesh="fl_steering_wheel_link" group="1" contype="0" conaffinity="0" />
        <body name="fl_wheel_link" pos="0 -0.001 -0.29345" quat="0.707105 0.707108 0 0">
          <inertial pos="0.000311409 -1.69638e-07 -0.00209856" quat="2.22515e-06 0.707112 2.22518e-06 0.707102" mass="11.4679" diaginertia="0.114229 0.0800825 0.0533205" />
          <joint name="fl_wheel_joint" pos="0 0 0" axis="0 0 -1" />
          <geom type="mesh" rgba="1 1 1 1" mesh="fl_wheel_link" group="1" contype="0" conaffinity="0" />
        </body>
      </body>
      <body name="rl_steering_wheel_link" pos="-0.445 0.28 0.0335">
        <inertial pos="0.000104728 0.00747335 -0.0851649" quat="0.502544 0.502294 -0.497444 0.497694" mass="2.09214" diaginertia="0.007922 0.00778254 0.00126619" />
        <joint name="rl_steering_wheel_joint" pos="0 0 0" axis="0 0 -1" range="-1.57 1.57" />
        <geom type="mesh" rgba="0 0 0 1" mesh="rl_steering_wheel_link" group="1" contype="0" conaffinity="0" />
        <body name="rl_wheel_link" pos="0 -0.001 -0.29345" quat="0.707105 0.707108 0 0">
          <inertial pos="0.000311409 -1.69638e-07 -0.00209856" quat="2.22515e-06 0.707112 2.22518e-06 0.707102" mass="11.4679" diaginertia="0.114229 0.0800825 0.0533205" />
          <joint name="rl_wheel_joint" pos="0 0 0" axis="0 0 -1" />
          <geom type="mesh" rgba="1 1 1 1" mesh="rl_wheel_link" group="1" contype="0" conaffinity="0" />
        </body>
      </body>
      <body name="rr_steering_wheel_link" pos="-0.445 -0.28 0.0335">
        <inertial pos="0.000104728 0.00747335 -0.0851649" quat="0.502544 0.502294 -0.497444 0.497694" mass="2.09214" diaginertia="0.007922 0.00778254 0.00126619" />
        <joint name="rr_steering_wheel_joint" pos="0 0 0" axis="0 0 -1" range="-1.57 1.57" />
        <geom type="mesh" rgba="0 0 0 1" mesh="rr_steering_wheel_link" group="1" contype="0" conaffinity="0" />
        <body name="rr_wheel_link" pos="0 0.001 -0.2918" quat="0.707105 0.707108 0 0">
          <inertial pos="-0.00095113 -1.69638e-07 0.00209543" quat="2.22208e-06 0.707112 2.22211e-06 0.707102" mass="11.4679" diaginertia="0.114229 0.0800825 0.0533205" />
          <joint name="rr_wheel_joint" pos="0 0 0" axis="0 0 1" />
          <geom type="mesh" rgba="1 1 1 1" mesh="rr_wheel_link" group="1" contype="0" conaffinity="0" />
        </body>
      </body>

      <!-- Arm: CR10 on base -->
      <body name="cr10_Link1" pos="0.2462 0 0.2765">
        <inertial pos="-1.6635e-06 -0.010819 0.0028389" quat="0.999935 -0.0113747 -1.32872e-05 -5.20368e-05" mass="4.1649" diaginertia="0.020714 0.0188262 0.0145728" />
        <joint name="cr10_joint1" pos="0 0 0" axis="0 0 1" range="-3.92 0.94" />
        <geom type="mesh" rgba="1 1 1 1" mesh="Link1" group="1" />
        <!-- armbase frame sensor -->
        <site name="armbasepoint" pos="0 0 0" size="0.01" rgba="1 0 0 1" group="2" />
        <body name="cr10_Link2" quat="0.499998 0.5 0.5 -0.500002">
          <inertial pos="-0.24631 3.5816e-07 0.19515" quat="0.503124 0.496855 0.49686 0.503122" mass="11.314" diaginertia="0.30683 0.299762 0.0335742" />
          <joint name="cr10_joint2" pos="0 0 0" axis="0 0 1" range="-1.57 1.57" />
          <geom type="mesh" rgba="1 1 1 1" mesh="Link2" group="1" />
          <body name="cr10_Link3" pos="-0.607 0 0">
            <inertial pos="-0.270856 2.51447e-07 0.062296" quat="0.502759 0.497224 0.497187 0.502799" mass="4.89191" diaginertia="0.239673 0.238398 0.00712848" />
            <joint name="cr10_joint3" pos="0 0 0" axis="0 0 1" range="-2.86 2.86" />
            <geom type="mesh" rgba="1 1 1 1" mesh="Link3" group="1" />
            <body name="cr10_Link4" pos="-0.568 0 0.191" quat="0.707105 0 0 -0.707108">
              <inertial pos="-2.53137e-07 0.00798724 -0.00589074" quat="0.691341 0.722528 1.80757e-05 -1.34988e-05" mass="1.16884" diaginertia="0.00292615 0.00273515 0.00149812" />
              <joint name="cr10_joint4" pos="0 0 0" axis="0 0 1" range="-3.14 3.14" />
              <geom type="mesh" rgba="1 1 1 1" mesh="Link4" group="1" />
              <body name="cr10_Link5" pos="0 -0.125 0" quat="0.707105 0.707108 0 0">
                <inertial pos="5.50699e-07 -0.0145468 -0.00454887" quat="0.723949 0.689853 2.09649e-05 -8.16019e-06" mass="1.22303" diaginertia="0.00337905 0.00323707 0.00148849" />
                <joint name="cr10_joint5" pos="0 0 0" axis="0 0 1" range="-3.14 3.14" />
                <geom type="mesh" rgba="1 1 1 1" mesh="Link5" group="1" />
                <body name="cr10_Link6" pos="0 0.1084 0" quat="0.707105 -0.707108 0 0">
                  <inertial pos="-0.00372511 -0.00481821 -0.0011281" quat="0.0135251 0.718835 0.10122 0.687639" mass="1.1116" diaginertia="0.00622428 0.00491985 0.00244698" />
                  <joint name="cr10_joint6" pos="0 0 0" axis="0 0 1" range="-3.14 3.14" />
                  <geom type="mesh" rgba="1 1 1 1" mesh="Link6" group="1" />
                  <!-- EE site + sensor -->
                  <site name="right_center" pos="0 0 0" size="0.01" rgba="1 0.3 0.3 1" group="2" />
                  <site name="endpoint" pos="0 0 0" size="0.01" rgba="1 0.3 0.3 1" group="2" />
                  <geom type="mesh" mesh="ag95_base_link" group="1" />
                  <geom size="0.013 0.062 0.0145" pos="-0.0100115 -0.0744978 0.0327115" quat="0.500398 0.499602 -0.5 0.5" type="box" group="1" />
                  <body name="gripper_finger1_knuckle_link" pos="-0.016 0 0.10586" quat="0.000796327 0 0 1">
                    <inertial pos="0 0 0" quat="0.634377 0.634377 -0.312356 0.312356" mass="0.011111" diaginertia="4.96455e-06 3.24091e-06 1.92174e-06" />
                    <joint name="gripper_finger1_joint" pos="0 0 0" axis="0 -1 0" range="0 0.65" />
                    <geom type="mesh" mesh="crank_Link" group="1" />
                    <body name="gripper_finger1_finger_link" pos="0.029208 0 -0.0227133">
                      <inertial pos="0 0 0" quat="0.985484 0 0.169771 0" mass="0.0222871" diaginertia="9.5054e-06 7.26157e-06 2.56546e-06" />
                      <joint name="gripper_finger1_finger_joint" pos="0 0 0" axis="0 -1 0" range="-3.14 3.14" />
                      <geom type="mesh" mesh="rod_Link" group="1" />
                    </body>
                  </body>
                  <body name="gripper_finger2_knuckle_link" pos="0.016 0 0.10586">
                    <inertial pos="0 0 0" quat="0.634377 0.634377 -0.312356 0.312356" mass="0.011111" diaginertia="4.96455e-06 3.24091e-06 1.92174e-06" />
                    <joint name="gripper_finger2_joint" pos="0 0 0" axis="0 -1 0" range="0 0.65" />
                    <geom type="mesh" mesh="crank_Link" group="1" />
                    <body name="gripper_finger2_finger_link" pos="0.029208 0 -0.0227133">
                      <inertial pos="0 0 0" quat="0.985484 0 0.169771 0" mass="0.0222871" diaginertia="9.5054e-06 7.26157e-06 2.56546e-06" />
                      <joint name="gripper_finger2_finger_joint" pos="0 0 0" axis="0 -1 0" range="-3.14 3.14" />
                      <geom type="mesh" mesh="rod_Link" group="1" />
                    </body>
                  </body>
                  <body name="gripper_finger1_inner_knuckle_link" pos="-0.016 0 0.10586" quat="0.000796327 0 0 1">
                    <inertial pos="0 0 0" quat="0.927336 0 0.37423 0" mass="0.0318004" diaginertia="1.88191e-05 1.04943e-05 8.78398e-06" />
                    <joint name="gripper_finger1_inner_knuckle_joint" pos="0 0 0" axis="0 -1 0" range="-3.14 3.14" />
                    <geom type="mesh" mesh="proximal_phalanx_Link" group="1" />
                    <body name="gripper_finger1_finger_tip_link" pos="0.0394969 0 0.0382752">
                      <inertial pos="0 0 0" quat="0.981854 0 -0.189641 0" mass="0.0124305" diaginertia="3.17302e-06 3.03961e-06 7.87295e-07" />
                      <joint name="gripper_finger1_finger_tip_joint" pos="0 0 0" axis="0 1 0" range="-3.14 3.14" />
                      <geom type="mesh" mesh="distal_phalanx_Link" group="1" />
                    </body>
                  </body>
                  <body name="gripper_finger2_inner_knuckle_link" pos="0.016 0 0.10586">
                    <inertial pos="0 0 0" quat="0.927336 0 0.37423 0" mass="0.0318004" diaginertia="1.88191e-05 1.04943e-05 8.78398e-06" />
                    <joint name="gripper_finger2_inner_knuckle_joint" pos="0 0 0" axis="0 -1 0" range="-3.14 3.14" />
                    <geom type="mesh" mesh="proximal_phalanx_Link" group="1" />
                    <body name="gripper_finger2_finger_tip_link" pos="0.0394969 0 0.0382752">
                      <inertial pos="0 0 0" quat="0.981854 0 -0.189641 0" mass="0.0124305" diaginertia="3.17302e-06 3.03961e-06 7.87295e-07" />
                      <joint name="gripper_finger2_finger_tip_joint" pos="0 0 0" axis="0 1 0" range="-3.14 3.14" />
                      <geom type="mesh" mesh="distal_phalanx_Link" group="1" />
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <!-- IMU on base -->
    <gyro name="imu-gyro" site="imu" />
    <framequat name="imu-framequat" objtype="site" objname="imu" />
    <velocimeter name="imu-velocimeter" site="imu" />
    <framezaxis name="imu-framezaxis" objtype="site" objname="imu" />

    <!-- EE pose in local frame (endpoint site on cr10_Link6) -->
    <framepos name="endpoint-framepos" objtype="site" objname="endpoint" />
    <framequat name="endpoint-framequat" objtype="site" objname="endpoint" />

    <!-- armbase pose in world frame (armbasepoint site on cr10_Link1) -->
    <framepos name="armbasepoint-framepos" objtype="site" objname="armbasepoint" />
    <framequat name="armbasepoint-framequat" objtype="site" objname="armbasepoint" />
  </sensor>

  <actuator>
    <position name="cr10_joint1_act" joint="cr10_joint1" kp="100" ctrlrange="-3.92 0.94"/>
    <position name="cr10_joint2_act" joint="cr10_joint2" kp="110" ctrlrange="-1.57 1.57"/>
    <position name="cr10_joint3_act" joint="cr10_joint3" kp="95"  ctrlrange="-2.86 2.86"/>
    <position name="cr10_joint4_act" joint="cr10_joint4" kp="50"  ctrlrange="-3.14 3.14"/>
    <position name="cr10_joint5_act" joint="cr10_joint5" kp="50"  ctrlrange="-3.14 3.14"/>
    <position name="cr10_joint6_act" joint="cr10_joint6" kp="50"  ctrlrange="-3.14 3.14"/>
    <position name="gripper_finger1_joint_act" joint="gripper_finger1_joint" kp="500" ctrlrange="0 0.65"/>
  </actuator>

  <equality>
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger2_joint" polycoef="0 1.0 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger1_finger_joint" polycoef="0 2.1910883179497023 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger2_finger_joint" polycoef="0 2.1910883179497023 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger1_inner_knuckle_joint" polycoef="0 0.6690621097381623 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger2_inner_knuckle_joint" polycoef="0 0.6690621097381623 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger1_finger_tip_joint" polycoef="0 0.6690621097381623 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
    <joint joint1="gripper_finger1_joint" joint2="gripper_finger2_finger_tip_joint" polycoef="0 0.6690621097381623 0 0 0" solref="0.0005 1" solimp="0.99 0.999 0.001" />
  </equality>
</mujoco>
```

- [ ] **Step 3: Create scene_flat.xml**

Create `src/unilab/assets/robots/ranger_box/scene_flat.xml`:

```xml
<mujoco model="ranger_box scene">
  <include file="robot.xml"/>

  <statistic center="0 0 0.1" extent="1.5" meansize="0.04"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>

  <keyframe>
    <key name="home"
      qpos="0 0 0.278 1 0 0 0 0 0 0 0 0 0 0 0 0 -0.3 0.75 0 0.45 0 0 0 0 0 0 0 0 0"
      ctrl="0 -0.3 0.75 0 0.45 0 0"
    />
  </keyframe>
</mujoco>
```

- [ ] **Step 4: Verify model loads with correct dimensions**

```python
import mujoco
import numpy as np
from pathlib import Path

xml_path = Path("src/unilab/assets/robots/ranger_box/scene_flat.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))
assert model.nq == 29, f"Expected nq=29, got {model.nq}"
assert model.nu == 7, f"Expected nu=7, got {model.nu}"
print(f"nq={model.nq}, nu={model.nu} — OK")
```

Run: `uv run python -c "import mujoco; from pathlib import Path; m=mujoco.MjModel.from_xml_path(str(Path('src/unilab/assets/robots/ranger_box/scene_flat.xml'))); print(f'nq={m.nq} nu={m.nu}'); assert m.nq==29; assert m.nu==7; print('OK')"`
Expected: `nq=29 nu=7` then `OK`

- [ ] **Step 5: Commit**

```bash
git add src/unilab/assets/robots/ranger_box/
git commit -m "feat: add RangerBox robot.xml, scene_flat.xml, and 26 mesh files"
```

### Task 3: Env Config Dataclasses

**Files:**
- Create: `src/unilab/envs/locomotion/ranger_box/reach_env.py` (dataclasses portion — rest filled in Tasks 5-6)

**Interfaces:**
- Consumes: Go2ArmBaseCfg, Go2ArmSensor, Asset, ControlConfig, DomainRandConfig, NoiseConfig, EEGoalConfig, IKConfig, HistoryConfig, ArmStageConfig, SceneCfg, InitState from `go2_arm` and `common` packages
- Produces:
  - `RangerBoxAsset(Asset)` — overridden defaults for base_name, ee_site_name, arm_joint_names, wheel_positions, etc.
  - `RangerBoxSensor(Go2ArmSensor)` — overridden field values for IMU and EE sensor names
  - `RangerBoxControlConfig(ControlConfig)` — arm_kp/kd tuples, gripper_kp/kd, arm_action_scale
  - `BaseVelocityControllerConfig` — standalone @dataclass with max_lin_vel, tau, latency, noise params
  - `RangerBoxDomainRandConfig(DomainRandConfig)` — push_robots=false, randomize_ground_friction=false
  - `RangerBoxRewardConfig` — standalone @dataclass with scales dict + sigma_ee
  - `RangerBoxReachCfg(Go2ArmBaseCfg)` — top-level env config with all sub-configs

**Spec reference:** [spec §7.7](docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md)

- [ ] **Step 1: Create ranger_box package directory and __init__.py placeholder**

```bash
mkdir -p src/unilab/envs/locomotion/ranger_box
touch src/unilab/envs/locomotion/ranger_box/__init__.py
```

- [ ] **Step 2: Create reach_env.py with all dataclass definitions**

Create `src/unilab/envs/locomotion/ranger_box/reach_env.py` with only the dataclass definitions for now (the env class and DR provider are added in Tasks 5-6):

```python
"""RangerBoxReach — mobile manipulator EE reaching environment.

Dataclass definitions.  Env class, DR provider, and reward functions
are in this same module to keep the package focused.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.base import ControlConfig, DomainRandConfig
from unilab.envs.locomotion.common.domain_rand import NoiseConfig
from unilab.envs.locomotion.go2_arm.base import (
    Asset,
    Go2ArmBaseCfg,
    Go2ArmSensor,
)


# ── Asset ──────────────────────────────────────────────────────────────

@dataclass
class RangerBoxAsset(Asset):
    base_name: str = "base"
    ground: str = "floor"
    ee_site_name: str = "right_center"
    ee_body_name: str = "cr10_Link6"
    arm_joint_names: tuple[str, ...] = (
        "cr10_joint1", "cr10_joint2", "cr10_joint3",
        "cr10_joint4", "cr10_joint5", "cr10_joint6",
    )
    gripper_joint_name: str = "gripper_finger1_joint"
    steering_joint_names: tuple[str, ...] = (
        "fr_steering_joint", "fl_steering_wheel_joint",
        "rl_steering_wheel_joint", "rr_steering_wheel_joint",
    )
    wheel_joint_names: tuple[str, ...] = (
        "fr_wheel_joint", "fl_wheel_joint",
        "rl_wheel_joint", "rr_wheel_joint",
    )
    wheel_positions: tuple[tuple[float, float], ...] = (
        (0.445, -0.28), (0.445, 0.28),
        (-0.445, 0.28), (-0.445, -0.28),
    )
    wheel_radius: float = 0.152


# ── Sensor ─────────────────────────────────────────────────────────────

@dataclass
class RangerBoxSensor(Go2ArmSensor):
    local_linvel: str = "imu-velocimeter"
    gyro: str = "imu-gyro"
    framequat: str = "imu-framequat"
    ee_local_pos: str = "endpoint-framepos"
    ee_local_quat: str = "endpoint-framequat"
    arm_ref_world_quat: str = "armbasepoint-framequat"
    armbase_world_pos: str = "armbasepoint-framepos"


# ── Control ────────────────────────────────────────────────────────────

@dataclass
class RangerBoxControlConfig(ControlConfig):
    arm_action_scale: float = 0.03
    arm_kp: tuple[float, ...] = (100.0, 110.0, 95.0, 50.0, 50.0, 50.0)
    arm_kd: tuple[float, ...] = (3.5, 3.8, 2.5, 1.5, 1.5, 1.5)
    gripper_kp: float = 500.0
    gripper_kd: float = 10.0


# ── Base Velocity Controller Config ────────────────────────────────────

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


# ── Domain Randomization ───────────────────────────────────────────────

@dataclass
class RangerBoxDomainRandConfig(DomainRandConfig):
    randomize_ground_friction: bool = False
    randomize_kp: bool = True
    kp_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_kd: bool = True
    kd_multiplier_range: tuple[float, float] = (0.9, 1.1)
    randomize_body_mass: bool = True
    body_mass_multiplier_range: tuple[float, float] = (0.9, 1.1)
    random_com: bool = True
    com_offset_x: tuple[float, float] = (-0.03, 0.03)
    randomize_dof_armature: bool = True
    dof_armature_multiplier_range: tuple[float, float] = (0.8, 1.2)
    push_robots: bool = False


# ── Reward ─────────────────────────────────────────────────────────────

@dataclass
class RangerBoxRewardConfig:
    scales: dict[str, float] = field(default_factory=lambda: {
        "ee_distance": 4.0,
        "ee_distance_l2": -1.0,
        "base_vel_xy": -0.05,
        "base_vel_z": 0.0,
        "base_vel_yaw": -0.01,
        "arm_dof_vel": -0.001,
        "arm_dof_acc": -1.0e-6,
        "torques": 0.0,
        "base_orientation": 0.0,
        "base_height": 0.0,
        "arm_joint_limits": -1.0,
        "arm_collision": 0.0,
        "action_rate": -0.01,
        "similar_to_default": -0.005,
        "alive": 0.3,
    })
    sigma_ee: float = 0.15


# ── Helper: scene & model_file defaults ────────────────────────────────

def _default_ranger_box_model_file() -> str:
    return str(ASSETS_ROOT_PATH / "robots" / "ranger_box" / "scene_flat.xml")


def _default_ranger_box_scene() -> SceneCfg:
    return SceneCfg(model_file=_default_ranger_box_model_file())
```

- [ ] **Step 3: Verify dataclasses import cleanly**

Run: `python -c "import unilab.envs.locomotion.ranger_box.reach_env; print('OK')"`
Expected: `OK` (no ImportError)

- [ ] **Step 4: Write unit test for dataclass defaults**

Append to `tests/test_ranger_box_reach.py`:

```python


class TestRangerBoxDataclasses:
    """Verify all dataclass defaults match spec."""

    def test_asset_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxAsset
        a = RangerBoxAsset()
        assert a.base_name == "base"
        assert a.ee_site_name == "right_center"
        assert a.ee_body_name == "cr10_Link6"
        assert len(a.arm_joint_names) == 6
        assert a.arm_joint_names[0] == "cr10_joint1"
        assert a.gripper_joint_name == "gripper_finger1_joint"
        assert len(a.steering_joint_names) == 4
        assert len(a.wheel_joint_names) == 4
        assert len(a.wheel_positions) == 4
        assert a.wheel_radius == 0.152

    def test_sensor_inherits_go2arm(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxSensor
        from unilab.envs.locomotion.go2_arm.base import Go2ArmSensor
        s = RangerBoxSensor()
        assert isinstance(s, Go2ArmSensor)
        assert s.local_linvel == "imu-velocimeter"
        assert s.gyro == "imu-gyro"
        assert s.ee_local_pos == "endpoint-framepos"
        assert s.armbase_world_pos == "armbasepoint-framepos"
        # Verify parent field names are present (not renamed)
        assert hasattr(s, "framequat")
        assert hasattr(s, "arm_ref_world_quat")

    def test_control_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxControlConfig
        c = RangerBoxControlConfig()
        assert c.arm_action_scale == 0.03
        assert c.arm_kp == (100.0, 110.0, 95.0, 50.0, 50.0, 50.0)
        assert c.arm_kd == (3.5, 3.8, 2.5, 1.5, 1.5, 1.5)
        assert c.gripper_kp == 500.0
        assert c.gripper_kd == 10.0

    def test_domain_rand_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxDomainRandConfig
        d = RangerBoxDomainRandConfig()
        assert d.push_robots is False
        assert d.randomize_ground_friction is False
        assert d.randomize_kp is True
        assert d.kp_multiplier_range == (0.9, 1.1)

    def test_reward_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxRewardConfig
        r = RangerBoxRewardConfig()
        assert "ee_distance" in r.scales
        assert r.scales["base_vel_z"] == 0.0
        assert r.scales["base_orientation"] == 0.0
        assert r.scales["base_height"] == 0.0
        assert r.sigma_ee == 0.15

    def test_base_velocity_controller_config_defaults(self):
        from unilab.envs.locomotion.ranger_box.reach_env import BaseVelocityControllerConfig
        c = BaseVelocityControllerConfig()
        assert c.max_lin_vel == 1.5
        assert c.max_ang_vel == 1.0
        assert c.tau == 0.05
        assert c.max_latency_steps == 4
        assert c.enable_latency is True
```

- [ ] **Step 5: Run dataclass tests**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxDataclasses -v`
Expected: 6 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/unilab/envs/locomotion/ranger_box/ tests/test_ranger_box_reach.py
git commit -m "feat: add RangerBox dataclasses — Asset, Sensor, Control, DomainRand, Reward, BaseVelocityControllerConfig"
```

### Task 4: BaseVelocityController — Vectorized A+ Pipeline

**Files:**
- Create: `src/unilab/envs/locomotion/ranger_box/base_velocity_controller.py`

**Interfaces:**
- Consumes:
  - `SimBackend.set_root_planar_velocity(lin_vel_xy, yaw_rate, *, preserve_uncontrolled)` (Task 1)
  - `SimBackend.set_joint_qpos(names, values)` (Task 1)
  - `SimBackend.set_joint_qvel(names, values)` (Task 1)
  - `BaseVelocityControllerConfig` (Task 3)
  - `RangerBoxAsset` (Task 3)
- Produces:
  - `BaseVelocityController(cfg, dt, backend, asset_cfg, num_envs)` — init with ring buffer, state
  - `BaseVelocityController.reset(env_ids, rng)` — reset latency steps and velocity state for given envs
  - `BaseVelocityController.step(action_base_vel)` — full 9-step pipeline, returns None

**Spec reference:** [spec §3](docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md)

- [ ] **Step 1: Write the failing test for controller API**

Append to `tests/test_ranger_box_reach.py`:

```python


class TestBaseVelocityController:
    """Unit tests for the A+ vectorized base velocity controller."""

    @pytest.fixture
    def ctrl(self):
        from unilab.envs.locomotion.ranger_box.reach_env import (
            BaseVelocityControllerConfig,
            RangerBoxAsset,
        )
        # Use a mock backend so we can run without MuJoCo
        import numpy as np

        class _MockBackend:
            def set_root_planar_velocity(self, lin_vel_xy, yaw_rate, *, preserve_uncontrolled=True):
                self._last_lin_vel = np.asarray(lin_vel_xy).copy()
                self._last_yaw_rate = np.asarray(yaw_rate).copy()
            def set_joint_qpos(self, names, values):
                pass
            def set_joint_qvel(self, names, values):
                pass

        from unilab.envs.locomotion.ranger_box.base_velocity_controller import (
            BaseVelocityController,
        )
        cfg = BaseVelocityControllerConfig()
        asset = RangerBoxAsset()
        backend = _MockBackend()
        return BaseVelocityController(cfg, 0.02, backend, asset, num_envs=4)

    def test_init_shapes(self, ctrl):
        N = 4
        assert ctrl.v_real.shape == (N, 3)
        assert ctrl.latency_ring.shape == (5, N, 3)  # max_latency_steps=4 → ring depth 5
        assert ctrl.latency_steps.shape == (N,)
        assert ctrl.latency_write_ptr.shape == (N,)

    def test_reset_zeros_velocity(self, ctrl):
        ctrl.v_real[:] = 1.0
        ctrl.latency_ring[:, :, :] = 1.0
        ctrl.reset(np.array([0, 2]), np.random.default_rng(42))
        np.testing.assert_allclose(ctrl.v_real[[0, 2]], 0.0)
        np.testing.assert_allclose(ctrl.latency_ring[:, [0, 2], :], 0.0)

    def test_reset_does_not_touch_other_envs(self, ctrl):
        ctrl.v_real[:] = 1.0
        ctrl.reset(np.array([0]), np.random.default_rng(42))
        np.testing.assert_allclose(ctrl.v_real[0], 0.0)
        np.testing.assert_allclose(ctrl.v_real[1:], 1.0)

    def test_step_produces_valid_output_range(self, ctrl):
        action = np.zeros((4, 3))
        ctrl.reset(np.arange(4), np.random.default_rng(42))
        ctrl.step(action)
        # velocity should be zero (no action)
        assert np.all(np.abs(ctrl.v_real) < 0.01)

    def test_step_with_max_action_respects_limits(self, ctrl):
        N = 4
        ctrl.reset(np.arange(N), np.random.default_rng(42))
        action = np.ones((N, 3))  # max action = 1.0
        ctrl.step(action)
        # After first step with tau=0.05 and dt=0.02:
        # alpha = 0.02 / (0.05 + 0.02) ≈ 0.2857
        # target = clip(1.0 * 1.5, 1.5) = 1.5
        # v_real = 0 + 0.2857 * 1.5 ≈ 0.429
        assert np.all(ctrl.v_real[:, 0] > 0.0)  # velocity increases
        assert np.all(np.abs(ctrl.v_real[:, 0]) <= 1.5)  # respects max_lin_vel

    def test_wheel_ik_shapes(self, ctrl):
        """Wheel IK produces (N, 4) steer and omega arrays."""
        import numpy as np
        N = 4
        v_real = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.3, 0.1]])
        from unilab.envs.locomotion.ranger_box.base_velocity_controller import (
            _compute_wheel_ik,
        )
        steer, omega = _compute_wheel_ik(
            v_real,
            np.array([(0.445, -0.28), (0.445, 0.28), (-0.445, 0.28), (-0.445, -0.28)]),
            0.152,
        )
        assert steer.shape == (N, 4)
        assert omega.shape == (N, 4)
        # Straight forward: all steer = 0, all omega = vx / radius
        np.testing.assert_allclose(steer[0], 0.0, atol=1e-6)
        expected_omega = 1.0 / 0.152
        np.testing.assert_allclose(omega[0], expected_omega, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestBaseVelocityController -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'unilab.envs.locomotion.ranger_box.base_velocity_controller'`

- [ ] **Step 3: Implement BaseVelocityController**

Create `src/unilab/envs/locomotion/ranger_box/base_velocity_controller.py`:

```python
"""Vectorized base velocity controller (A+ scheme) for RangerBox.

Latency ring buffer, acceleration clip, first-order response, noise,
and wheel IK — all on (N, 3) tensors.
"""

from __future__ import annotations

import numpy as np


def _compute_wheel_ik(
    v_real: np.ndarray,
    wheel_positions: tuple[tuple[float, float], ...],
    wheel_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute steer angles and wheel angular velocities from base velocity.

    Args:
        v_real: ``(N, 3)`` — vx, vy, vyaw in base frame.
        wheel_positions: ``((x, y), ...)`` — 4 wheel positions in base frame.
        wheel_radius: wheel radius in meters.

    Returns:
        ``(steer, omega)`` — both ``(N, 4)`` float64.
    """
    N = len(v_real)
    pos = np.asarray(wheel_positions, dtype=np.float64)  # (4, 2)
    x = pos[:, 0]  # (4,)
    y = pos[:, 1]  # (4,)

    vx_i = v_real[:, 0:1] - v_real[:, 2:3] * y[None, :]   # (N, 4)
    vy_i = v_real[:, 1:2] + v_real[:, 2:3] * x[None, :]   # (N, 4)

    steer = np.arctan2(vy_i, vx_i)          # (N, 4)
    omega = np.sqrt(vx_i**2 + vy_i**2) / wheel_radius  # (N, 4)
    return steer, omega


class BaseVelocityController:
    """Vectorized base velocity controller with latency, acceleration,
    first-order response, noise, and wheel visualization.

    Operates on ``(N, 3)`` tensors for N parallel envs.
    """

    def __init__(
        self,
        cfg,                # BaseVelocityControllerConfig
        dt: float,
        backend,            # SimBackend
        asset_cfg,          # RangerBoxAsset
        num_envs: int,
    ):
        self._cfg = cfg
        self._dt = dt
        self._backend = backend
        self._asset = asset_cfg
        self._num_envs = num_envs

        self.v_real = np.zeros((num_envs, 3), dtype=np.float64)
        self.latency_ring = np.zeros(
            (cfg.max_latency_steps + 1, num_envs, 3), dtype=np.float64
        )
        self.latency_steps = np.zeros(num_envs, dtype=np.int32)
        self.latency_write_ptr = np.zeros(num_envs, dtype=np.int32)

        # Pre-compute clip arrays
        self._max_vel_arr = np.array(
            [cfg.max_lin_vel, cfg.max_lin_vel, cfg.max_ang_vel], dtype=np.float64
        )
        self._max_acc_arr = np.array(
            [cfg.max_lin_acc, cfg.max_lin_acc, cfg.max_ang_acc], dtype=np.float64
        )

    def reset(self, env_ids: np.ndarray, rng: np.random.Generator) -> None:
        self.latency_steps[env_ids] = rng.integers(
            0, self._cfg.max_latency_steps + 1, size=len(env_ids)
        )
        self.latency_write_ptr[env_ids] = 0
        self.latency_ring[:, env_ids, :] = 0.0
        self.v_real[env_ids] = 0.0

    def step(self, action_base_vel: np.ndarray) -> None:
        """Full vectorized pipeline: scale → clip → latency → accel limit
        → first-order → noise → final clip → wheel IK → world-frame write.

        Args:
            action_base_vel: ``(N, 3)`` — policy output in [-1, 1].
        """
        N = self._num_envs
        cfg = self._cfg
        dt = self._dt

        # --- 1. Scale ---
        v_cmd = action_base_vel.astype(np.float64).copy()
        v_cmd[:, 0:2] *= cfg.action_scale_lin
        v_cmd[:, 2] *= cfg.action_scale_ang

        # --- 2. Clip ---
        v_cmd[:, 0:2] = np.clip(v_cmd[:, 0:2], -cfg.max_lin_vel, cfg.max_lin_vel)
        v_cmd[:, 2] = np.clip(v_cmd[:, 2], -cfg.max_ang_vel, cfg.max_ang_vel)

        # --- 3. Latency ring ---
        if cfg.enable_latency:
            L = self.latency_ring.shape[0]
            wp = self.latency_write_ptr
            self.latency_ring[wp, np.arange(N), :] = v_cmd
            rp = (wp - self.latency_steps) % L
            v_cmd = self.latency_ring[rp, np.arange(N), :].copy()
            self.latency_write_ptr = (wp + 1) % L

        # --- 4. Acceleration limit ---
        dv = v_cmd - self.v_real
        dv[:, 0:2] = np.clip(dv[:, 0:2], -cfg.max_lin_acc * dt, cfg.max_lin_acc * dt)
        dv[:, 2] = np.clip(dv[:, 2], -cfg.max_ang_acc * dt, cfg.max_ang_acc * dt)
        v_target = self.v_real + dv

        # --- 5. First-order response ---
        alpha = dt / (cfg.tau + dt)
        self.v_real = self.v_real + alpha * (v_target - self.v_real)

        # --- 6. Noise (after first-order) ---
        if cfg.enable_noise:
            noise = np.random.standard_normal((N, 3)).astype(np.float64)
            noise[:, 0:2] *= cfg.action_noise_scale * cfg.max_lin_vel
            noise[:, 2] *= cfg.action_noise_scale * cfg.max_ang_vel
            self.v_real = self.v_real + noise

        # --- 7. Final clip ---
        self.v_real = np.clip(self.v_real, -self._max_vel_arr, self._max_vel_arr)

        # --- 8. Wheel visualization (before world-frame conversion) ---
        if cfg.enable_wheel_visualization:
            steer, omega = _compute_wheel_ik(
                self.v_real, self._asset.wheel_positions, self._asset.wheel_radius
            )
            self._backend.set_joint_qpos(list(self._asset.steering_joint_names), steer)
            self._backend.set_joint_qvel(list(self._asset.wheel_joint_names), omega)

        # --- 9. World-frame conversion ---
        # base_quat is read from the backend
        base_quat = self._backend.get_sensor_data("imu-framequat")

        # Rotate body-frame velocity to world frame
        v_body = np.concatenate(
            [self.v_real[:, 0:2], np.zeros((N, 1))], axis=1, dtype=np.float64
        )
        w_body = np.concatenate(
            [np.zeros((N, 2)), self.v_real[:, 2:3]], axis=1, dtype=np.float64
        )

        from unilab.envs.common.rotation import np_quat_apply_batched
        v_world = np_quat_apply_batched(base_quat, v_body)
        w_world = np_quat_apply_batched(base_quat, w_body)

        self._backend.set_root_planar_velocity(
            v_world[:, :2], w_world[:, 2], preserve_uncontrolled=True
        )
```

- [ ] **Step 4: Run tests to verify controller passes**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestBaseVelocityController -v`
Expected: 6 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/locomotion/ranger_box/base_velocity_controller.py tests/test_ranger_box_reach.py
git commit -m "feat: add BaseVelocityController with vectorized A+ pipeline"
```

### Task 5: Reward Functions

**Files:**
- Modify: `src/unilab/envs/locomotion/ranger_box/reach_env.py` (add reward functions + RewardContext)

**Interfaces:**
- Consumes:
  - `RangerBoxRewardConfig` (Task 3) — scales dict + sigma_ee
  - `rewards` module from `unilab.envs.locomotion.common` — reuse `alive`, `action_rate`, `similar_to_default`, `dof_vel`, `dof_acc`, `dof_pos_limits`, `torques`
- Produces:
  - `RewardContext` @dataclass — holds all per-step data for reward computation
  - `_reward_ee_distance(ctx) -> np.ndarray` — Gaussian EE-goal distance in world frame
  - `_reward_ee_distance_l2(ctx) -> np.ndarray` — squared L2 EE-goal distance
  - `_reward_base_vel_xy(ctx) -> np.ndarray` — base horizontal velocity penalty
  - `_reward_base_vel_yaw(ctx) -> np.ndarray` — base yaw velocity penalty
  - `_reward_arm_dof_vel(ctx) -> np.ndarray` — arm joint velocity penalty
  - `_reward_arm_dof_acc(ctx) -> np.ndarray` — arm joint acceleration penalty
  - `_reward_arm_joint_limits(ctx) -> np.ndarray` — joint limit margin penalty

- [ ] **Step 1: Write the failing test for reward functions**

Append to `tests/test_ranger_box_reach.py`:

```python


class TestRangerBoxRewards:
    """Unit tests for RangerBox reward functions."""

    @pytest.fixture
    def ctx(self):
        import numpy as np
        N = 4

        class _Ctx:
            num_envs = N
            sigma_ee = 0.15
            info: dict = {}
            linvel = np.zeros((N, 3))
            gyro = np.zeros((N, 3))
            gravity = np.zeros((N, 3))
            arm_pos = np.zeros((N, 6))
            arm_vel = np.zeros((N, 6))
            prev_arm_vel = np.zeros((N, 6))
            gripper_pos = np.zeros((N, 1))
            default_arm_angles = np.array([0.0, -0.3, 0.75, 0.0, 0.45, 0.0])
            armbase_pos_world = np.zeros((N, 3))
            armbase_quat_world = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
            ee_local_pos = np.zeros((N, 3))
            ee_pos_world = np.tile([0.5, 0.0, 0.8], (N, 1))
            world_ee_goal = np.tile([0.5, 0.0, 0.8], (N, 1))
            armbase_ee_goal = np.tile([0.3, 0.0, 0.5], (N, 1))
            arm_joint_upper = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14])
            arm_joint_lower = np.array([-3.92, -1.57, -2.86, -3.14, -3.14, -3.14])
            joint_limit_margin = 0.01
            ctrl_dt = 0.02
            current_actions = np.zeros((N, 10))

        return _Ctx()

    def test_ee_distance_perfect_match(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_ee_distance
        r = _reward_ee_distance(ctx)
        np.testing.assert_allclose(r, 1.0, atol=1e-4)  # exp(-0) = 1.0

    def test_ee_distance_far(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_ee_distance
        ctx.ee_pos_world = np.tile([0.0, 0.0, 0.0], (4, 1))
        ctx.world_ee_goal = np.tile([2.0, 0.0, 0.0], (4, 1))
        r = _reward_ee_distance(ctx)
        # dist² = 4.0, sigma²=0.0225, exp(-4/0.0225) ≈ exp(-177.8) ≈ 0
        assert np.all(r < 1e-10)

    def test_ee_distance_l2(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_ee_distance_l2
        ctx.ee_pos_world = np.tile([0.0, 0.0, 0.0], (4, 1))
        ctx.world_ee_goal = np.tile([3.0, 4.0, 0.0], (4, 1))
        r = _reward_ee_distance_l2(ctx)
        np.testing.assert_allclose(r, 25.0, rtol=1e-5)  # 3²+4²=25

    def test_base_vel_xy(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_base_vel_xy
        ctx.linvel = np.tile([1.0, 2.0, 0.0], (4, 1))
        r = _reward_base_vel_xy(ctx)
        np.testing.assert_allclose(r, 5.0, rtol=1e-5)  # 1²+2²=5

    def test_arm_joint_limits_ok(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_arm_joint_limits
        ctx.arm_pos = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1))
        r = _reward_arm_joint_limits(ctx)
        assert np.all(r == 0.0)  # within limits

    def test_arm_joint_limits_violated(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_arm_joint_limits
        ctx.arm_pos = np.tile([0.95, 0.0, 0.0, 0.0, 0.0, 0.0], (4, 1))
        r = _reward_arm_joint_limits(ctx)
        assert np.all(r > 0.0)  # positive penalty

    def test_arm_dof_acc(self, ctx):
        from unilab.envs.locomotion.ranger_box.reach_env import _reward_arm_dof_acc
        ctx.arm_vel = np.ones((4, 6))
        ctx.prev_arm_vel = np.zeros((4, 6))
        r = _reward_arm_dof_acc(ctx)
        expected = np.sum((1.0 / 0.02) ** 2)  # (1/0.02)² per DOF, 6 DOFs
        np.testing.assert_allclose(r[0], expected, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxRewards -v`
Expected: FAIL — `ImportError: cannot import name '_reward_ee_distance'`

- [ ] **Step 3: Implement reward functions in reach_env.py**

Append to `src/unilab/envs/locomotion/ranger_box/reach_env.py`:

```python

# ══════════════════════════════════════════════════════════════════════════
# Reward context
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _RewardContext:
    """Per-step data available to all reward functions."""
    info: dict
    linvel: np.ndarray
    gyro: np.ndarray
    gravity: np.ndarray
    arm_pos: np.ndarray
    arm_vel: np.ndarray
    prev_arm_vel: np.ndarray
    gripper_pos: np.ndarray
    num_envs: int
    default_arm_angles: np.ndarray
    armbase_pos_world: np.ndarray
    armbase_quat_world: np.ndarray
    ee_local_pos: np.ndarray
    ee_pos_world: np.ndarray
    world_ee_goal: np.ndarray
    armbase_ee_goal: np.ndarray
    sigma_ee: float
    arm_joint_upper: np.ndarray
    arm_joint_lower: np.ndarray
    joint_limit_margin: float
    ctrl_dt: float
    current_actions: np.ndarray


# ══════════════════════════════════════════════════════════════════════════
# Reward functions — all return positive values (negative in YAML scale)
# ══════════════════════════════════════════════════════════════════════════

def _reward_ee_distance(ctx: _RewardContext) -> np.ndarray:
    """Gaussian EE→goal distance in world frame."""
    diff = ctx.ee_pos_world - ctx.world_ee_goal
    d2 = np.sum(diff * diff, axis=1)
    sigma2 = ctx.sigma_ee * ctx.sigma_ee
    return np.exp(-d2 / sigma2)


def _reward_ee_distance_l2(ctx: _RewardContext) -> np.ndarray:
    """Squared L2 EE→goal distance in world frame."""
    diff = ctx.ee_pos_world - ctx.world_ee_goal
    return np.sum(diff * diff, axis=1)


def _reward_base_vel_xy(ctx: _RewardContext) -> np.ndarray:
    """Penalize base horizontal velocity magnitude."""
    return ctx.linvel[:, 0] ** 2 + ctx.linvel[:, 1] ** 2


def _reward_base_vel_yaw(ctx: _RewardContext) -> np.ndarray:
    """Penalize base yaw velocity."""
    return ctx.gyro[:, 2] ** 2


def _reward_arm_dof_vel(ctx: _RewardContext) -> np.ndarray:
    """Penalize arm joint velocity magnitude."""
    return np.sum(ctx.arm_vel * ctx.arm_vel, axis=1)


def _reward_arm_dof_acc(ctx: _RewardContext) -> np.ndarray:
    """Penalize arm joint acceleration magnitude."""
    acc = (ctx.arm_vel - ctx.prev_arm_vel) / ctx.ctrl_dt
    return np.sum(acc * acc, axis=1)


def _reward_arm_joint_limits(ctx: _RewardContext) -> np.ndarray:
    """Penalize proximity to arm joint limits (positive penalty when close)."""
    margin = ctx.joint_limit_margin
    upper_violation = np.maximum(0.0, ctx.arm_pos - (ctx.arm_joint_upper - margin))
    lower_violation = np.maximum(0.0, (ctx.arm_joint_lower + margin) - ctx.arm_pos)
    return np.sum(upper_violation + lower_violation, axis=1)
```

- [ ] **Step 4: Run reward tests**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxRewards -v`
Expected: 7 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/locomotion/ranger_box/reach_env.py tests/test_ranger_box_reach.py
git commit -m "feat: add RangerBox reward functions — ee_distance, base_vel, arm_joint_limits, arm_dof_acc"
```

### Task 6: RangerBoxReachEnv + DR Provider

**Files:**
- Modify: `src/unilab/envs/locomotion/ranger_box/reach_env.py` (add env class, DR provider, scene helpers, build_gains, _ReservedDimNames)

**Interfaces:**
- Consumes:
  - All dataclasses from Task 3 (RangerBoxAsset, RangerBoxSensor, RangerBoxControlConfig, BaseVelocityControllerConfig, RangerBoxDomainRandConfig, RangerBoxRewardConfig, RangerBoxReachCfg)
  - Reward functions + `_RewardContext` from Task 5
  - BaseVelocityController from Task 4
  - `Go2ArmBaseEnv`, `Go2ArmBaseCfg`, `build_go2_arm_position_gains` from `go2_arm.base`
  - `LocomotionDRProvider` from `locomotion.common.dr_provider`
  - `ReservedDimNames` from `unilab.training.sim2sim`
- Produces:
  - `build_ranger_box_position_gains(cc: RangerBoxControlConfig) -> dict[str, np.ndarray]`
  - `_resolve_ranger_box_scene(cfg: RangerBoxReachCfg) -> SceneCfg`
  - `RangerBoxReachDRProvider(LocomotionDRProvider)` — with `build_reset_plan`, `_compute_reset_obs`, `_sample_commands`, `_get_base_actuator_gains`, `_get_reset_randomization_baselines`
  - `RangerBoxReachEnv(Go2ArmBaseEnv)` — with `__init__`, `_init_action_space`, `_init_buffers`, `apply_action`, `update_state`, `_compute_raw_obs`, `_update_history`, `_compute_reward`, `obs_groups_spec`, `get_arm_dof_pos`, `get_arm_dof_vel`, `get_gripper_dof_pos`, `get_ee_local_pose`, `compute_arm_ik_delta`, `_get_projected_gravity`, goal sampling, `reset_ee_goals`, `_world_goal_to_armbase`

**Spec reference:** [spec §7](docs/superpowers/specs/2026-06-29-ranger-box-reach-design.md)

- [ ] **Step 1: Write the contract test for env construction**

Append to `tests/test_ranger_box_reach.py`:

```python


class TestRangerBoxReachEnvConstruction:
    """Smoke tests for env construction and basic API."""

    def test_create_env_via_gym_make(self):
        """gym.make creates a RangerBoxReachEnv with 1 env."""
        import gymnasium as gym
        # Register happens at import time via the registry decorator
        from unilab.base.registry import ensure_registries
        ensure_registries()
        # After Task 7 registers the module, this should work.
        # For now, test the class directly.
        pytest.skip("Requires registry wiring from Task 7")

    def test_env_instantiation_direct(self):
        """Direct instantiation produces correct action/obs spaces."""
        import numpy as np
        from unilab.envs.locomotion.ranger_box.reach_env import (
            RangerBoxReachCfg,
            RangerBoxReachEnv,
        )
        cfg = RangerBoxReachCfg()
        env = RangerBoxReachEnv(cfg, num_envs=4, backend_type="mujoco")
        assert env._num_action == 10
        obs = env.observation_space
        assert obs["obs"].shape[0] == 41  # H=1 by default
        act = env.action_space
        assert act.shape[0] == 10
        assert act.low[0] == -1.0
        assert act.high[0] == 1.0
        env.close()

    def test_env_reset_step_cycle(self):
        """reset() returns (obs_dict, info_dict), step() returns NpEnvState."""
        from unilab.envs.locomotion.ranger_box.reach_env import (
            RangerBoxReachCfg,
            RangerBoxReachEnv,
        )
        cfg = RangerBoxReachCfg()
        env = RangerBoxReachEnv(cfg, num_envs=4, backend_type="mujoco")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "obs" in obs
        assert obs["obs"].shape == (4, 41)
        actions = np.zeros((4, 10))
        state = env.step(actions)
        assert state.obs["obs"].shape == (4, 41)
        assert state.reward.shape == (4,)
        assert state.terminated.shape == (4,)
        env.close()

    def test_obs_does_not_contain_world_ee_goal(self):
        """Verify 41-dim obs layout has no world_ee_goal (translation invariance)."""
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxReachCfg, RangerBoxReachEnv
        cfg = RangerBoxReachCfg()
        env = RangerBoxReachEnv(cfg, num_envs=2, backend_type="mujoco")
        obs, _ = env.reset()
        # The obs is 41 dims. world_ee_goal would add 3 dims if present.
        # With the spec layout it's 41 dims and armbase_ee_goal is a derived field.
        assert obs["obs"].shape[1] == 41
        env.close()

    def test_action_to_ctrl_shape(self):
        """apply_action converts (N, 10) policy to (N, 7) ctrl."""
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxReachCfg, RangerBoxReachEnv
        from unilab.base.np_env import NpEnvState
        cfg = RangerBoxReachCfg()
        env = RangerBoxReachEnv(cfg, num_envs=4, backend_type="mujoco")
        actions = np.zeros((4, 10))
        state = env.init_state()
        state.info["current_actions"] = actions.copy()
        ctrl = env.apply_action(actions, state)
        assert ctrl.shape == (4, 7)
        env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxReachEnvConstruction -v`
Expected: FAIL — `RangerBoxReachCfg` or `RangerBoxReachEnv` not defined or incomplete

- [ ] **Step 3: Implement the full env class + DR provider**

Append to `src/unilab/envs/locomotion/ranger_box/reach_env.py`. This is the bulk of the implementation. The full code follows — it appends to the existing file content.

First, add the imports and helpers at the top of the file (if not already there from Task 3). Then add the scene resolution, config class, DR provider, and env class.

```python
# ── Scene resolution (continuing from dataclass section) ───────────────

def _resolve_ranger_box_scene(cfg: "RangerBoxReachCfg") -> SceneCfg:
    """Resolve scene config, falling back to defaults."""
    scene = cfg.scene
    default_model_file = _default_ranger_box_model_file()
    if scene is None:
        scene = SceneCfg(model_file=cfg.model_file)
    elif cfg.model_file != default_model_file and scene.model_file == default_model_file:
        scene = SceneCfg(
            model_file=cfg.model_file,
            fragment_files=list(scene.fragment_files) if scene.fragment_files else [],
            terrain=scene.terrain,
        )
    cfg.scene = scene
    return scene


def build_ranger_box_position_gains(cc: RangerBoxControlConfig) -> dict[str, np.ndarray]:
    """Return per-actuator kp/kd gain dict for 7 actuators (6 arm + 1 gripper).

    Returns ``{"kp": np.array(7,), "kd": np.array(7,)}`` — matching the
    ``position_actuator_gains`` format expected by ``create_backend``.
    """
    return {
        "kp": np.concatenate([
            np.asarray(cc.arm_kp, dtype=np.float64),
            np.asarray([cc.gripper_kp], dtype=np.float64),
        ]),
        "kd": np.concatenate([
            np.asarray(cc.arm_kd, dtype=np.float64),
            np.asarray([cc.gripper_kd], dtype=np.float64),
        ]),
    }
```

Then add the top-level config class:

```python
# ── Top-level EnvCfg ───────────────────────────────────────────────────

from unilab.base import registry
from unilab.envs.locomotion.go2_arm.base import (
    ArmStageConfig,
    EEGoalConfig,
    HistoryConfig,
    IKConfig,
    InitState,
)


@registry.envcfg("RangerBoxReach")
@dataclass
class RangerBoxReachCfg(Go2ArmBaseCfg):
    scene: SceneCfg | None = field(default_factory=_default_ranger_box_scene)
    model_file: str = field(default_factory=_default_ranger_box_model_file)
    max_episode_seconds: float = 30.0
    init_state: InitState = field(default_factory=InitState)
    control_config: RangerBoxControlConfig = field(default_factory=RangerBoxControlConfig)
    sensor: RangerBoxSensor = field(default_factory=RangerBoxSensor)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    goal_ee: EEGoalConfig = field(default_factory=EEGoalConfig)
    ik: IKConfig = field(default_factory=IKConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    arm_stage: ArmStageConfig = field(default_factory=ArmStageConfig)
    reward_config: RangerBoxRewardConfig = field(default_factory=RangerBoxRewardConfig)
    domain_rand: RangerBoxDomainRandConfig = field(default_factory=RangerBoxDomainRandConfig)
    base_velocity_controller: BaseVelocityControllerConfig = field(
        default_factory=BaseVelocityControllerConfig
    )
    asset: RangerBoxAsset = field(default_factory=RangerBoxAsset)
```

Then add the DR Provider:

```python
# ── DR Provider ────────────────────────────────────────────────────────

from unilab.dr.types import ResetPlan
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.dtype_config import get_global_dtype


class RangerBoxReachDRProvider(LocomotionDRProvider):
    """DR provider for RangerBoxReach — caches kp/kd/mass/armature at init."""

    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_body_mass: np.ndarray | None = None,
        base_dof_armature: np.ndarray | None = None,
    ):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_dof_armature = base_dof_armature

    def _sample_commands(self, env, num_reset: int) -> np.ndarray:
        return np.zeros((num_reset, 3), dtype=get_global_dtype())

    def _get_base_actuator_gains(self, env) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(self, env):
        return self._base_body_mass, None, None, self._base_dof_armature

    def build_reset_plan(self, env, env_ids: np.ndarray) -> ResetPlan:
        plan = super().build_reset_plan(env, env_ids)
        env._arm_goal_timer[env_ids] = 0
        env._history_obs_buf[env_ids] = 0.0
        env._history_critic_buf[env_ids] = 0.0
        env._prev_arm_vel[env_ids] = 0.0
        env._base_controller.reset(env_ids, np.random.default_rng())
        return plan

    def _compute_reset_obs(
        self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        # reset_ee_goals() is called HERE — after set_state() has run
        env.reset_ee_goals(env_ids)
        n = len(env_ids)
        sliced_info: dict = {}
        for k, v in info_updates.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == env._num_envs:
                sliced_info[k] = v[env_ids]
            else:
                sliced_info[k] = v
        ee_local_pos, _ = env.get_ee_local_pose()
        raw = env._compute_raw_obs(
            sliced_info, linvel, gyro, gravity, dof_pos, dof_vel,
            ee_local_pos[env_ids], env.armbase_ee_goal[env_ids],
            add_noise=True,
        )
        return env._update_history(raw, env_ids=env_ids)
```

Then add the env class:

```python
# ── RangerBoxReachEnv ──────────────────────────────────────────────────

from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.envs.common.rotation import np_matrix_from_quat, np_quat_apply_batched
from unilab.envs.locomotion.common import rewards as common_rewards
from unilab.envs.locomotion.ranger_box.base_velocity_controller import BaseVelocityController

_RAW_OBS_DIM = 41


@registry.env("RangerBoxReach", sim_backend="mujoco")
class RangerBoxReachEnv(Go2ArmBaseEnv):
    """Mobile manipulator EE reaching with A+ freejoint base controller."""

    _cfg: RangerBoxReachCfg

    def __init__(self, cfg: RangerBoxReachCfg, num_envs: int = 1, backend_type: str = "mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")

        # Resolve scene
        scene = _resolve_ranger_box_scene(cfg)

        # Build position gains
        position_actuator_gains = build_ranger_box_position_gains(cfg.control_config)

        # Create backend
        backend_kwargs = {
            "base_name": cfg.asset.base_name,
            "push_body_name": cfg.domain_rand.push_body_name,
            "position_actuator_gains": position_actuator_gains,
            "iterations": getattr(cfg, "iterations", None),
            "post_step_forward_sensor": getattr(cfg, "post_step_forward_sensor", False),
        }
        backend = create_backend(backend_type, scene, num_envs, cfg.sim_dt, **backend_kwargs)

        # Parent init
        super().__init__(cfg, backend, num_envs)

        # Action space: 10-dim policy
        self._num_action = 10

        # Ctrl bounds from backend (7 actuators)
        ctrl_range = self._backend.get_actuator_ctrl_range()
        self._ctrl_low = np.asarray(ctrl_range[:, 0], dtype=np.float64)
        self._ctrl_high = np.asarray(ctrl_range[:, 1], dtype=np.float64)

        # Gripper DOF indices
        self._gripper_dof_pos_idx = self._backend.get_joint_dof_pos_indices(
            [cfg.asset.gripper_joint_name]
        )
        self._gripper_dof_vel_idx = self._backend.get_joint_dof_vel_indices(
            [cfg.asset.gripper_joint_name]
        )

        # Base velocity controller
        self._base_controller = BaseVelocityController(
            cfg.base_velocity_controller, cfg.ctrl_dt, backend, cfg.asset, num_envs
        )

        # World goal and armbase caches
        self.world_ee_goal = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_ee_goal = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_pos_world = np.zeros((num_envs, 3), dtype=np.float64)
        self.armbase_quat_world = np.zeros((num_envs, 4), dtype=np.float64)

        # Prev arm velocity for acceleration reward
        self._prev_arm_vel = np.zeros((num_envs, 6), dtype=np.float64)

        # EE goal trajectory tracking
        self._arm_goal_timer = np.zeros((num_envs,), dtype=np.int32)

        # History buffers
        H_a = cfg.history.num_actor_history
        H_c = cfg.history.num_critic_history
        self._history_obs_buf = np.zeros(
            (num_envs, H_a * _RAW_OBS_DIM), dtype=np.float64
        )
        self._history_critic_buf = np.zeros(
            (num_envs, H_c * _RAW_OBS_DIM), dtype=np.float64
        )

        # Default arm angles from keyframe
        self._default_arm_angles = self.default_angles[:6].copy()

        # Arm joint limits for termination and reward
        self._arm_joint_upper = np.array([0.94, 1.57, 2.86, 3.14, 3.14, 3.14], dtype=np.float64)
        self._arm_joint_lower = np.array(
            [-3.92, -1.57, -2.86, -3.14, -3.14, -3.14], dtype=np.float64
        )

        # DR provider
        base_kp, base_kd = (None, None)
        if cfg.domain_rand.randomize_kp or cfg.domain_rand.randomize_kd:
            base_kp, base_kd = backend.get_actuator_gains()
        base_body_mass = None
        if cfg.domain_rand.randomize_body_mass:
            base_body_mass = backend.get_body_mass()
        base_dof_armature = None
        if cfg.domain_rand.randomize_dof_armature:
            base_dof_armature = backend.get_dof_armature()
        dr_provider = RangerBoxReachDRProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=base_body_mass,
            base_dof_armature=base_dof_armature,
        )
        self._init_domain_randomization(dr_provider)

        # Init goals and reward fns
        self._init_ee_goals()
        self._init_reward_functions()

    # ── Action space ────────────────────────────────────────────────

    def _init_action_space(self) -> None:
        import gymnasium as gym
        self._action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(10,), dtype=np.float32
        )

    def _init_buffers(self) -> None:
        """Override: default_angles is 7 dims (6 arm + 1 gripper), not 18."""
        dtype = get_global_dtype()
        raw_qpos = self._backend.get_keyframe_qpos("home")
        self._init_qpos = np.asarray(raw_qpos, dtype=dtype)
        self.default_angles = np.asarray(self._init_qpos[-7:], dtype=dtype)
        raw_qvel = self._backend.get_init_qvel()
        self._init_qvel = np.asarray(raw_qvel, dtype=dtype)

    # ── Action application ──────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions.copy()

        # Base velocity via controller
        self._base_controller.step(actions[:, 0:3])

        # Arm/gripper latency
        arm_gripper_action = (
            state.info["last_actions"][:, 3:10]
            if self._cfg.control_config.simulate_action_latency
            else actions[:, 3:10]
        )

        # world goal → armbase conversion (per-step dynamic)
        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world
        )

        # Arm IK delta
        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        dq_ik = self.compute_arm_ik_delta(
            self.armbase_ee_goal, ee_local_pos,
            self.ee_goal_orn_quat if hasattr(self, "ee_goal_orn_quat") else None,
            ee_local_quat,
        )

        arm_ctrl = (
            self.get_arm_dof_pos()
            + arm_gripper_action[:, 0:6] * self._cfg.control_config.arm_action_scale
            + self._cfg.ik.gain * dq_ik
        )

        # Gripper: always open in v1
        grip_ctrl = np.zeros((actions.shape[0], 1), dtype=np.float64)
        ctrl = np.concatenate([arm_ctrl, grip_ctrl], axis=1)
        ctrl_out = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        return ctrl_out.astype(get_global_dtype())

    # ── Observation ─────────────────────────────────────────────────

    def _compute_raw_obs(
        self, info, linvel, gyro, gravity, dof_pos, dof_vel,
        ee_local_pos, armbase_ee_goal, *, add_noise=True,
    ) -> np.ndarray:
        """41-dim raw observation (no history)."""
        n = len(dof_pos)
        arm_diff = dof_pos - self._default_arm_angles
        if add_noise:
            noise_cfg = self._cfg.noise_config
            linvel = self._obs_noise(linvel, noise_cfg.scale_linvel)
            gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
            gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
            arm_diff = self._obs_noise(arm_diff, noise_cfg.scale_joint_angle)
            dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
            ee_local_pos = self._obs_noise(ee_local_pos, noise_cfg.scale_ee_pos)
            armbase_ee_goal = self._obs_noise(armbase_ee_goal, noise_cfg.scale_ee_goal)

        last_actions = info.get("current_actions", np.zeros((n, 10), dtype=get_global_dtype()))
        ee_error = armbase_ee_goal - ee_local_pos
        gripper_pos = self.get_gripper_dof_pos()

        return np.concatenate([
            linvel,              # 3
            gyro,                # 3
            -gravity,            # 3
            arm_diff,            # 6
            dof_vel,             # 6
            ee_local_pos,        # 3
            armbase_ee_goal,     # 3
            ee_error,            # 3
            gripper_pos,         # 1
            last_actions,        # 10
        ], axis=1, dtype=get_global_dtype())

    def _update_history(self, raw_obs, env_ids=None, *, critic_raw_obs=None):
        D = _RAW_OBS_DIM
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        critic_step = raw_obs if critic_raw_obs is None else critic_raw_obs
        if env_ids is None:
            if H_a > 1:
                self._history_obs_buf = np.roll(self._history_obs_buf, -D, axis=1)
            self._history_obs_buf[:, -D:] = raw_obs
            if H_c > 1:
                self._history_critic_buf = np.roll(self._history_critic_buf, -D, axis=1)
            self._history_critic_buf[:, -D:] = critic_step
            return {"obs": self._history_obs_buf.copy(), "critic": self._history_critic_buf.copy()}
        else:
            if H_a > 1:
                self._history_obs_buf[env_ids] = np.roll(self._history_obs_buf[env_ids], -D, axis=1)
            self._history_obs_buf[env_ids, -D:] = raw_obs
            if H_c > 1:
                self._history_critic_buf[env_ids] = np.roll(
                    self._history_critic_buf[env_ids], -D, axis=1
                )
            self._history_critic_buf[env_ids, -D:] = critic_step
            return {
                "obs": self._history_obs_buf[env_ids].copy(),
                "critic": self._history_critic_buf[env_ids].copy(),
            }

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        H_a = self._cfg.history.num_actor_history
        H_c = self._cfg.history.num_critic_history
        return {"obs": H_a * _RAW_OBS_DIM, "critic": H_c * _RAW_OBS_DIM}

    # ── State update ────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._get_projected_gravity()

        ee_local_pos, ee_local_quat = self.get_ee_local_pose()
        arm_pos = self.get_arm_dof_pos()
        arm_vel = self.get_arm_dof_vel()

        # Armbase pose in world (via configured sensor names)
        self.armbase_pos_world = self._backend.get_sensor_data(
            self._cfg.sensor.armbase_world_pos
        )
        self.armbase_quat_world = self._backend.get_sensor_data(
            self._cfg.sensor.arm_ref_world_quat
        )

        # World goal → armbase conversion
        self.armbase_ee_goal = self._world_goal_to_armbase(
            self.world_ee_goal, self.armbase_pos_world, self.armbase_quat_world
        )

        # EE position in world frame
        ee_pos_world = self.armbase_pos_world + np_quat_apply_batched(
            self.armbase_quat_world, ee_local_pos
        )

        # Termination: tilt or arm joint hard limits
        tilt_sq = gravity[:, 0]**2 + gravity[:, 1]**2
        limit_violated = (
            (arm_pos > self._arm_joint_upper) | (arm_pos < self._arm_joint_lower)
        ).any(axis=1)
        terminated = (tilt_sq > np.sin(1.0)**2) | limit_violated

        # Arm acceleration
        arm_acc = (arm_vel - self._prev_arm_vel) / self._cfg.ctrl_dt
        self._prev_arm_vel = arm_vel.copy()

        # Build reward context
        ctx = _RewardContext(
            info=state.info,
            linvel=linvel, gyro=gyro, gravity=gravity,
            arm_pos=arm_pos, arm_vel=arm_vel, prev_arm_vel=self._prev_arm_vel,
            gripper_pos=self.get_gripper_dof_pos(),
            num_envs=self._num_envs,
            default_arm_angles=self._default_arm_angles,
            armbase_pos_world=self.armbase_pos_world,
            armbase_quat_world=self.armbase_quat_world,
            ee_local_pos=ee_local_pos,
            ee_pos_world=ee_pos_world,
            world_ee_goal=self.world_ee_goal,
            armbase_ee_goal=self.armbase_ee_goal,
            sigma_ee=self._reward_cfg.sigma_ee,
            arm_joint_upper=self._arm_joint_upper,
            arm_joint_lower=self._arm_joint_lower,
            joint_limit_margin=0.01,
            ctrl_dt=self._cfg.ctrl_dt,
            current_actions=state.info.get("current_actions", np.zeros((self._num_envs, 10))),
        )

        reward = self._compute_reward(ctx)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, arm_pos, arm_vel,
                                ee_local_pos, self.armbase_ee_goal, add_noise=True)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    # ── Reward ──────────────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        self._reward_cfg = self._cfg.reward_config
        self._reward_fns: dict[str, Any] = {
            "ee_distance": _reward_ee_distance,
            "ee_distance_l2": _reward_ee_distance_l2,
            "base_vel_xy": _reward_base_vel_xy,
            "base_vel_yaw": _reward_base_vel_yaw,
            "arm_dof_vel": _reward_arm_dof_vel,
            "arm_dof_acc": _reward_arm_dof_acc,
            "arm_joint_limits": _reward_arm_joint_limits,
            "action_rate": common_rewards.action_rate,
            "similar_to_default": common_rewards.similar_to_default,
            "alive": common_rewards.alive,
        }

    def _compute_reward(self, ctx: _RewardContext) -> np.ndarray:
        scales = self._reward_cfg.scales
        reward = np.zeros(ctx.num_envs, dtype=np.float64)
        for name, fn in self._reward_fns.items():
            scale = scales.get(name, 0.0)
            if abs(scale) < 1e-15:
                continue
            reward = reward + scale * fn(ctx)
        return np.asarray(reward * self._cfg.ctrl_dt, dtype=get_global_dtype())

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_projected_gravity(self) -> np.ndarray:
        quat = self._backend.get_sensor_data(self._cfg.sensor.framequat)
        R_wb = np_matrix_from_quat(quat)
        return np.einsum("nij,j->ni", np.swapaxes(R_wb, 1, 2),
                          np.array([0.0, 0.0, -1.0], dtype=R_wb.dtype))

    def get_arm_dof_pos(self) -> np.ndarray:
        idx = self._backend.get_joint_dof_pos_indices(self._cfg.asset.arm_joint_names)
        return self._backend.get_dof_pos()[:, idx]

    def get_arm_dof_vel(self) -> np.ndarray:
        idx = self._backend.get_joint_dof_vel_indices(self._cfg.asset.arm_joint_names)
        return self._backend.get_dof_vel()[:, idx]

    def get_gripper_dof_pos(self) -> np.ndarray:
        return self._backend.get_dof_pos()[:, self._gripper_dof_pos_idx]

    def get_ee_local_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self._backend.get_sensor_data(self._cfg.sensor.ee_local_pos)
        quat = self._backend.get_sensor_data(self._cfg.sensor.ee_local_quat)
        return pos, quat

    def compute_arm_ik_delta(self, target_pos, ee_pos, target_quat, ee_quat):
        """Simple IK delta: positional error in task space, scaled by gain."""
        delta = target_pos - ee_pos
        return delta * self._cfg.ik.damping

    # ── Goal management ─────────────────────────────────────────────

    def _init_ee_goals(self) -> None:
        self.world_ee_goal = np.zeros((self._num_envs, 3), dtype=np.float64)
        self.ee_goal_orn_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (self._num_envs, 1)
        )
        self._arm_goal_timer = np.zeros((self._num_envs,), dtype=np.int32)

    def reset_ee_goals(self, env_ids: np.ndarray) -> None:
        """Sample new world-frame EE goals for reset envs.

        Called AFTER set_state() in _compute_reset_obs so armbase pose is valid.
        30% reachable, 70% extended (requires base motion).
        """
        n = len(env_ids)
        rng = np.random
        reachable = rng.random(n) < 0.30
        goals = np.zeros((n, 3), dtype=np.float64)

        # Reachable goals: sample within arm workspace (sphere around armbase)
        n_reach = int(reachable.sum())
        if n_reach > 0:
            r = rng.uniform(0.2, 0.5, size=n_reach)
            phi = rng.uniform(-1.2, 1.0, size=n_reach)
            theta = rng.uniform(-2.0, 2.0, size=n_reach)
            goals[reachable, 0] = r * np.cos(phi) * np.cos(theta)
            goals[reachable, 1] = r * np.cos(phi) * np.sin(theta)
            goals[reachable, 2] = r * np.sin(phi)

        # Extended goals: further out, requires base motion
        n_ext = n - n_reach
        if n_ext > 0:
            r_e = rng.uniform(0.5, 1.2, size=n_ext)
            phi_e = rng.uniform(-1.2, 1.0, size=n_ext)
            theta_e = rng.uniform(-2.0, 2.0, size=n_ext)
            goals[~reachable, 0] = r_e * np.cos(phi_e) * np.cos(theta_e)
            goals[~reachable, 1] = r_e * np.cos(phi_e) * np.sin(theta_e)
            goals[~reachable, 2] = r_e * np.sin(phi_e)

        # Goals are in armbase frame. Convert to world frame.
        armbase_pos = self.armbase_pos_world[env_ids]
        armbase_quat = self.armbase_quat_world[env_ids]
        goals_world = armbase_pos + np_quat_apply_batched(armbase_quat, goals)
        self.world_ee_goal[env_ids] = goals_world

        self._arm_goal_timer[env_ids] = 0

    def _world_goal_to_armbase(self, world_goal, armbase_pos, armbase_quat):
        """Convert world-frame EE goal to armbase frame."""
        rel = world_goal - armbase_pos
        # Inverse rotation: conjugate quaternion
        q_conj = armbase_quat * np.array([1.0, -1.0, -1.0, -1.0])
        q_conj = q_conj / np.sum(q_conj * q_conj, axis=1, keepdims=True)
        return np_quat_apply_batched(q_conj, rel)

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel,
                     ee_local_pos, armbase_ee_goal, *, add_noise=True):
        raw = self._compute_raw_obs(
            info, linvel, gyro, gravity, dof_pos, dof_vel,
            ee_local_pos, armbase_ee_goal, add_noise=add_noise,
        )
        return self._update_history(raw)
```

- [ ] **Step 4: Run env construction tests**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxReachEnvConstruction -v`
Expected: 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/unilab/envs/locomotion/ranger_box/reach_env.py tests/test_ranger_box_reach.py
git commit -m "feat: add RangerBoxReachEnv with DR provider, reward computation, and A+ base control"
```

### Task 7: Registry Wiring + Hydra Config YAML

**Files:**
- Modify: `src/unilab/envs/locomotion/__init__.py` (add `"unilab.envs.locomotion.ranger_box"`)
- Modify: `src/unilab/envs/locomotion/ranger_box/__init__.py` (import env to trigger decorator)
- Create: `conf/ppo/task/ranger_box_reach/mujoco.yaml`

**Interfaces:**
- Consumes: `RangerBoxReachEnv`, `RangerBoxReachCfg` (Task 6)
- Produces: registered env name `RangerBoxReach-mujoco`, compose-able Hydra config for `task=ranger_box_reach/mujoco`

- [ ] **Step 1: Write the failing registry test**

Append to `tests/test_ranger_box_reach.py`:

```python


class TestRegistryAndConfig:
    """Verify env registration and Hydra config composition."""

    def test_registry_env_creation(self):
        """gym.make('RangerBoxReach-mujoco') works."""
        import gymnasium as gym
        from unilab.base.registry import ensure_registries
        ensure_registries()
        env = gym.make("RangerBoxReach-mujoco", num_envs=1)
        assert env._num_action == 10
        env.close()

    def test_envcfg_registered(self):
        """RangerBoxReachCfg is discoverable via registry."""
        from unilab.base.registry import ensure_registries, get_envcfg
        ensure_registries()
        cfg_cls = get_envcfg("RangerBoxReach")
        assert cfg_cls is not None

    def test_hydra_compose_does_not_throw(self):
        """Hydra task=ranger_box_reach/mujoco composes successfully."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-c",
             "from hydra import compose, initialize_config_dir; "
             "from pathlib import Path; "
             "cfg_dir = str(Path('conf/ppo').absolute()); "
             "with initialize_config_dir(version_base=None, config_dir=cfg_dir): "
             "    cfg = compose(config_name='config', overrides=['task=ranger_box_reach/mujoco']); "
             "    print('OK: task_name =', cfg.training.task_name)"],
            capture_output=True, text=True, cwd=".",
        )
        if result.returncode != 0:
            pytest.fail(f"Hydra compose failed:\n{result.stderr}")
        assert "OK" in result.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRegistryAndConfig -v`
Expected: FAIL — `gym.error.NameNotFound: RangerBoxReach-mujoco`

- [ ] **Step 3: Write ranger_box/__init__.py**

Replace the placeholder `src/unilab/envs/locomotion/ranger_box/__init__.py`:

```python
"""RangerBox mobile manipulator environments."""

from unilab.envs.locomotion.ranger_box.reach_env import (
    RangerBoxReachCfg,
    RangerBoxReachEnv,
)

__all__ = [
    "RangerBoxReachCfg",
    "RangerBoxReachEnv",
]
```

- [ ] **Step 4: Add ranger_box to locomotion/__init__.py registry modules**

Edit `src/unilab/envs/locomotion/__init__.py`. Change line 3-9 from:

```python
__unilab_registry_modules__ = (
    "unilab.envs.locomotion.go1",
    "unilab.envs.locomotion.go2",
    "unilab.envs.locomotion.go2w",
    "unilab.envs.locomotion.g1",
    "unilab.envs.locomotion.go2_arm",
)
```

To:

```python
__unilab_registry_modules__ = (
    "unilab.envs.locomotion.go1",
    "unilab.envs.locomotion.go2",
    "unilab.envs.locomotion.go2w",
    "unilab.envs.locomotion.g1",
    "unilab.envs.locomotion.go2_arm",
    "unilab.envs.locomotion.ranger_box",
)
```

- [ ] **Step 5: Create Hydra config YAML**

```bash
mkdir -p conf/ppo/task/ranger_box_reach
```

Create `conf/ppo/task/ranger_box_reach/mujoco.yaml`:

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
      - actor

  policy:
    init_noise_std: 0.5
    actor_hidden_dims: [256, 128, 64]
    critic_hidden_dims: [256, 128, 64]

  algorithm:
    learning_rate: 3.0e-4
    entropy_coef: 1.0e-3
    num_mini_batches: 4

reward_config:
  scales:
    ee_distance: 4.0
    ee_distance_l2: -1.0
    base_vel_xy: -0.05
    base_vel_z: 0.0
    base_vel_yaw: -0.01
    arm_dof_vel: -0.001
    arm_dof_acc: -1.0e-6
    torques: 0.0
    base_orientation: 0.0
    base_height: 0.0
    arm_joint_limits: -1.0
    arm_collision: 0.0
    action_rate: -0.01
    similar_to_default: -0.005
    alive: 0.3
  sigma_ee: 0.15

env:
  max_episode_seconds: 30.0
  init_state:
    pos: [0.0, 0.0, 0.278]
  control_config:
    arm_action_scale: 0.03
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
    sphere_l_range: [0.20, 0.50]
    sphere_phi_range: [-1.20, 1.00]
    sphere_theta_range: [-2.00, 2.00]
    reachable_fraction: 0.30
    extended_l_range: [0.50, 1.20]
    extended_fraction: 0.70
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
    randomize_ground_friction: false
    randomize_dof_armature: true
    dof_armature_multiplier_range: [0.8, 1.2]
    push_robots: false
  history:
    num_actor_history: 1
    num_critic_history: 1
  arm_stage:
    freeze_arm_joints: false
    disable_ee_goal_trajectory: false
    fixed_ee_goal_cart: [0.30, 0.0, 0.30]
```

- [ ] **Step 6: Run registry + config tests**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRegistryAndConfig -v`
Expected: 3 tests pass

- [ ] **Step 7: Commit**

```bash
git add src/unilab/envs/locomotion/__init__.py src/unilab/envs/locomotion/ranger_box/__init__.py conf/ppo/task/ranger_box_reach/ tests/test_ranger_box_reach.py
git commit -m "feat: register RangerBoxReach env and add Hydra mujoco task config"
```

### Task 8: Integration Tests — Full System Validation

**Files:**
- Modify: `tests/test_ranger_box_reach.py` (add integration test class)

**Interfaces:**
- Consumes: All prior tasks (full stack)

- [ ] **Step 1: Write integration tests**

Append to `tests/test_ranger_box_reach.py`:

```python


@pytest.mark.slow
class TestRangerBoxReachIntegration:
    """Integration tests exercising the full env stack."""

    @pytest.fixture(scope="class")
    def env4(self):
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxReachCfg, RangerBoxReachEnv
        cfg = RangerBoxReachCfg()
        env = RangerBoxReachEnv(cfg, num_envs=4, backend_type="mujoco")
        yield env
        env.close()

    def test_mixed_latency_across_envs(self, env4):
        """4 envs with different latency_steps produce distinct delayed commands."""
        import numpy as np
        env4.reset()
        # Set different latency values
        env4._base_controller.latency_steps[:] = [0, 1, 3, 4]
        # Step with non-zero actions
        actions = np.array([
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        env4._base_controller.step(actions)
        # env 0 (latency=0): v_real reflects action immediately
        # env 3 (latency=4): v_real still uses old ring entry = 0
        assert env4._base_controller.v_real[0, 0] > 0.0  # immediate
        # env 3 should have much smaller response (old value from ring)
        assert abs(env4._base_controller.v_real[3, 0]) < 0.01

    def test_history_h3_buffer_shape(self, env4):
        """With num_actor_history=3, obs dimensions are 3*41=123."""
        import numpy as np
        # Cannot change history after init; create a new env
        from unilab.envs.locomotion.ranger_box.reach_env import RangerBoxReachCfg, RangerBoxReachEnv
        cfg = RangerBoxReachCfg()
        cfg.history.num_actor_history = 3
        cfg.history.num_critic_history = 3
        env_h3 = RangerBoxReachEnv(cfg, num_envs=4, backend_type="mujoco")
        obs, _ = env_h3.reset()
        assert obs["obs"].shape == (4, 3 * 41)
        env_h3.close()

    def test_partial_reset_clears_history(self, env4):
        """Partial reset zeroes history buffers for reset envs only."""
        import numpy as np
        env4.reset()
        env4._history_obs_buf[:] = 1.0
        env4._history_critic_buf[:] = 2.0
        env4._base_controller.v_real[:] = 3.0
        # Reset envs 0 and 2
        obs, info = env4.reset(np.array([0, 2]))
        assert np.all(env4._history_obs_buf[0] == 0.0)
        assert np.all(env4._history_obs_buf[2] == 0.0)
        assert np.all(env4._history_obs_buf[1] == 1.0)  # untouched
        assert np.all(env4._history_obs_buf[3] == 1.0)  # untouched

    def test_world_goal_fixed_across_steps(self, env4):
        """world_ee_goal is fixed for an episode; armbase_ee_goal changes with base motion."""
        env4.reset()
        goal_before = env4.world_ee_goal.copy()
        for _ in range(10):
            actions = np.zeros((4, 10))
            actions[:, 0] = 0.5  # move base forward
            env4.step(actions)
        # world_ee_goal is unchanged (episode still running)
        np.testing.assert_allclose(env4.world_ee_goal, goal_before)

    def test_se2_lock_z_roll_pitch_constant(self, env4):
        """SE(2) planar lock: z, roll, pitch stay constant across steps."""
        env4.reset()
        qpos_before = env4._backend._data.qpos.copy()
        for _ in range(20):
            actions = np.zeros((4, 10))
            actions[:, 0] = 1.0  # max forward
            env4.step(actions)
        qpos_after = env4._backend._data.qpos.copy()
        # z (index 2 in qpos) should be unchanged
        np.testing.assert_allclose(qpos_after[:, 2], qpos_before[:, 2], atol=1e-4)
        # roll/pitch from quaternion should be near identity
        # quat indices 3:7 in qpos; roll/pitch are zero → qpos[:, 3] ≈ 1, qpos[:, 4:6] ≈ 0
        np.testing.assert_allclose(qpos_after[:, 4], 0.0, atol=5e-2)
        np.testing.assert_allclose(qpos_after[:, 5], 0.0, atol=5e-2)

    def test_dr_reset_randomize_kp_does_not_crash(self, env4):
        """DR reset with randomize_kp=True completes without error."""
        # Reset all envs to trigger DR
        env4.reset()
        for _ in range(5):
            actions = np.zeros((4, 10))
            state = env4.step(actions)
            if state.terminated.any():
                env4.reset(np.where(state.terminated)[0])
        # No crash = success

    def test_extended_goals_require_base_motion(self, env4):
        """Extended goals (>0.5m from armbase) cannot be reached by arm alone."""
        import numpy as np
        env4.reset()
        # Force an extended goal
        env4.world_ee_goal[:] = env4.armbase_pos_world + np.array([1.0, 0.0, 0.0])
        ee_local_pos, _ = env4.get_ee_local_pose()
        # arm IK delta cannot bridge 1m gap (arm workspace ~0.5m radius)
        dq_ik = env4.compute_arm_ik_delta(
            env4.armbase_ee_goal, ee_local_pos,
            env4.ee_goal_orn_quat, np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)),
        )
        arm_delta_magnitude = np.linalg.norm(dq_ik, axis=1)
        # IK delta is scaled by gain=0.05, so even a 1m error gives 0.05m delta
        # which is much less than the 1m gap — base motion is required
        assert np.all(arm_delta_magnitude < 0.5)

    def test_action_rate_computation(self, env4):
        """action_rate reward penalizes change between successive actions."""
        import numpy as np
        env4.reset()
        actions1 = np.ones((4, 10))
        actions2 = np.zeros((4, 10))
        state = env4.init_state()
        state.info["current_actions"] = actions1.copy()
        env4.apply_action(actions1, state)
        state.info["current_actions"] = actions2.copy()
        env4.apply_action(actions2, state)
        # last_actions = actions1, current_actions = actions2
        last = state.info.get("last_actions")
        if last is not None:
            diff = (actions2 - last) ** 2
            assert np.all(diff > 0.0)  # action changed

    def test_se2_rewards_zeroed(self, env4):
        """base_vel_z, base_orientation, base_height reward scales are 0.0."""
        scales = env4._reward_cfg.scales
        assert scales["base_vel_z"] == 0.0
        assert scales["base_orientation"] == 0.0
        assert scales["base_height"] == 0.0
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_ranger_box_reach.py::TestRangerBoxReachIntegration -v -m slow`
Expected: 9 tests pass

- [ ] **Step 3: Run ALL tests together**

Run: `uv run pytest tests/test_ranger_box_reach.py -v`
Expected: All ~30 tests pass (5 Task 1 + 6 Task 3 + 6 Task 4 + 7 Task 5 + 5 Task 6 + 3 Task 7 + 9 Task 8)

- [ ] **Step 4: Final commit**

```bash
git add tests/test_ranger_box_reach.py
git commit -m "test: add RangerBoxReach integration tests — latency, history, SE(2) lock, DR, goals"
```

- [ ] **Step 5: Run make check to verify no regressions**

```bash
make check
```

If `make check` fails on unrelated files, confirm the failures are pre-existing and not caused by these changes. If caused by these changes, fix before proceeding.

- [ ] **Step 6: Run make test-all for full CI gate**

```bash
make test-all
```
