# foropenpi — RangerboxCR10Lidar 机器人模型包

本目录仅包含 **Ranger 整机模型与 robosuite 接入文件**，不含 RoboCasa 厨房场景资产（layout/style/mesh 等）。

用于迁移到 [robocasa-benchmark/openpi](https://github.com/robocasa-benchmark/openpi) 等工作区，与已安装的 `robosuite` / `robocasa` 合并。

## 目录结构

```text
foropenpi/
├── README.md
├── INTEGRATION.md              # 合并到上游 robosuite/robocasa 的步骤
├── tools/
│   └── convert_ranger_urdf_to_mjcf.py
├── rangerboxcr10lidar_description/urdf/
│   └── rangercr10lidar.urdf    # 源 URDF（重生成 MJCF 用；STL 仍在原仓库 meshes/）
└── robosuite/robosuite/
    ├── models/assets/robots/rangerboxcr10lidar/
    │   ├── robot.xml           # MuJoCo MJCF
    │   └── meshes/*.obj
    ├── models/robots/manipulators/rangerboxcr10lidar_robot.py
    ├── models/grippers/embedded_ag95_gripper.py
    ├── controllers/config/robots/default_rangerboxcr10lidar.json
    └── integration/            # 需手工合并进上游 __init__.py 的片段
```

## 快速安装（复制到 OpenPI 工作区）

假设 OpenPI 旁已有 `robosuite`、`robocasa` 克隆：

```bash
OPENPI_WS=/path/to/your/openpi-workspace
ROBOCASA_ROOT=/path/to/Robocasa   # 本仓库根目录（STL 网格在 description 包）

# 1. 覆盖/合并 robosuite 模型与配置
cp -r foropenpi/robosuite/robosuite/models/assets/robots/rangerboxcr10lidar \
  "$OPENPI_WS/robosuite/robosuite/models/assets/robots/"
cp foropenpi/robosuite/robosuite/models/robots/manipulators/rangerboxcr10lidar_robot.py \
  "$OPENPI_WS/robosuite/robosuite/models/robots/manipulators/"
cp foropenpi/robosuite/robosuite/models/grippers/embedded_ag95_gripper.py \
  "$OPENPI_WS/robosuite/robosuite/models/grippers/"
cp foropenpi/robosuite/robosuite/controllers/config/robots/default_rangerboxcr10lidar.json \
  "$OPENPI_WS/robosuite/robosuite/controllers/config/robots/"

# 2. 按 INTEGRATION.md 合并 __init__.py 与 robocasa 补丁

export PYTHONPATH="$OPENPI_WS/robosuite:$OPENPI_WS/robocasa"
pip install -e "$OPENPI_WS/robosuite" -e "$OPENPI_WS/robocasa"
```

## OpenPI 训练时注意

1. **场景数据**：厨房资产仍须单独 `python -m robocasa.scripts.download_kitchen_assets`（不在本包内）。
2. **指定机器人**：`gym.make("robocasa/<Task>", robots="RangerboxCR10Lidar", ...)`
3. **注册名**：`RangerboxCR10Lidar`（大小写敏感）

详见 `INTEGRATION.md`。
