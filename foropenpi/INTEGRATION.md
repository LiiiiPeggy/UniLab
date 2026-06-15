# 合并到 robosuite / robocasa（OpenPI 训练）

## 1. robosuite — `manipulators/__init__.py`

在 `robosuite/robosuite/models/robots/manipulators/__init__.py` 末尾增加：

```python
from .rangerboxcr10lidar_robot import RangerboxCR10Lidar
```

（导入后 `RobotModelMeta` 会自动注册 `REGISTERED_ROBOTS["RangerboxCR10Lidar"]`。）

## 2. robosuite — `models/grippers/__init__.py`

增加 import：

```python
from .embedded_ag95_gripper import EmbeddedAG95Gripper
```

在 `GRIPPER_MAPPING` 中增加：

```python
    "EmbeddedAG95": EmbeddedAG95Gripper,
```

## 3. robosuite — `robots/__init__.py`

在 `ROBOT_CLASS_MAPPING` 中增加：

```python
    "RangerboxCR10Lidar": WheeledRobot,
```

## 4. robocasa — 厨房环境（非场景资产，仅为加载 Ranger）

从本仓库 **Robocasa 根目录** 对照合并以下文件中的 `RangerboxCR10Lidar` 相关块（若 OpenPI 使用的 robocasa 较旧）：

| 文件 | 内容 |
|------|------|
| `robocasa/environments/kitchen/kitchen.py` | `JOINT_VELOCITY_LEGACY`、`body_part_ordering` |
| `robocasa/utils/env_utils.py` | `ROBOT_BASE_HEIGHT_OFFSET`、spawn 站位 |
| `robocasa/utils/camera_utils.py` | `eye_in_hand` → `robot0_cr10_Link6` |

## 5. OpenPI 评测 / 训练入口

`examples/robocasa/main.py` 默认 `PandaOmron`。使用 Ranger 时在 `gym.make` 传入：

```python
env = gym.make(
    f"robocasa/{env_name}",
    split=split,
    seed=seed,
    robots="RangerboxCR10Lidar",
)
```

并在 `src/openpi/training/config.py` 中配置与 **state/action 维度、相机名** 一致的数据集。

## 6. 验证

```bash
export PYTHONPATH="$(pwd)/robosuite:$(pwd)/robocasa"
pytest robosuite/tests/test_robots/test_rangerbox_kitchen_load.py -v
```

（测试文件未包含在 `foropenpi/`；请在本仓库 `ranger` 分支运行，或从上游复制测试。）

## 7. 重新生成 MJCF（可选）

STL 网格仍在原仓库 `rangerboxcr10lidar_description/meshes/`：

```bash
python foropenpi/tools/convert_ranger_urdf_to_mjcf.py \
  --urdf foropenpi/rangerboxcr10lidar_description/urdf/rangercr10lidar.urdf \
  --mesh-root /path/to/rangerboxcr10lidar_description/meshes \
  --out-xml robosuite/robosuite/models/assets/robots/rangerboxcr10lidar/robot.xml \
  --out-mesh-dir robosuite/robosuite/models/assets/robots/rangerboxcr10lidar/meshes
```
