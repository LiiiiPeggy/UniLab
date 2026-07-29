# agilex_ws/src 工作空间子项目总览

本工作空间是一个 ROS 1 (catkin) 工作空间，集成了移动机器人底盘驱动、机械臂控制、升降机构、夹爪、激光雷达、深度相机、SLAM 建图与导航、RViz 插件等功能模块，用于构建一套完整的移动操作机器人（Loco-Manipulation）系统。

---

## 当前设备配置

| 角色 | 设备型号 | 对应子项目 | 关键启动命令 |
|------|----------|-----------|-------------|
| **移动底盘** | AgileX Ranger（全尺寸） | [ranger_ros](#1-ranger_ros----ranger-底盘驱动) | `roslaunch ranger_bringup ranger.launch` |
| **机械臂** | Dobot CR10 | [TCP-IP-ROS-6AXis](#3-tcp-ip-ros-6axis----dobot-六轴机械臂-sdk) | `export DOBOT_TYPE=cr10` 后 `roslaunch dobot_v4_bringup bringup_v4.launch` |
| **夹爪** | DH-Robotics AG95 | [gripper](#4-gripper----dh-robotics-夹爪驱动) | `roslaunch dh_gripper_driver dh_gripper.launch GripperModel:=AG95_MB` |
| **激光雷达** | RoboSense RS-Helios-16P | [rslidar_sdk](#7-rslidar_sdk----robosense-激光雷达驱动) | `roslaunch rslidar_sdk start.launch` |
| **深度相机** | Intel RealSense D435 | [realsense-ros](#8-realsense-ros----intel-realsense-深度相机驱动) | `roslaunch realsense2_camera rs_camera.launch` |

### 快速启动流程（Ranger + CR10 + AG95）

```bash
# 1. 使能 CAN 口（每次开机执行一次）
rosrun ranger_bringup bringup_can2usb.bash

# 2. 启动 Ranger 底盘
roslaunch ranger_bringup ranger.launch
# 遥控器第二根杆需要在最上：指令控制模式

# 启动
rosrun teleop_twist_keyboard teleop_twist_keyboard.py cmd_vel:=/cmd_vel
```
使用按键 `i` 前进、`,` 后退、`j` 左转、`l` 右转、`k` 停止。
**命令行直接发送（可用）**

```bash
# 前进（线速度 0.3 m/s）
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.3
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"

# 左转（角速度 0.5 rad/s）
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.5"

# 停止
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"
```


# 3. 启动 CR10 机械臂（新终端）
export DOBOT_TYPE=cr10
roslaunch dobot_v4_bringup bringup_v4.launch robotIp:=192.168.5.1

# 4. 启动 AG95 夹爪（新终端）
roslaunch dh_gripper_driver dh_gripper.launch GripperModel:=AG95_MB Connectport:=/dev/ttyUSB0

# 5. 启动 MoveIt 运动规划（新终端）
export DOBOT_TYPE=cr10
roslaunch dobot_moveit moveit.launch

# 6. 启动激光雷达 + 导航（可选）
roslaunch rslidar_sdk start.launch
roslaunch ranger_bringup navigation_4wd.launch
```

---

## 目录

1. [ranger_ros -- Ranger 底盘驱动](#1-ranger_ros----ranger-底盘驱动)
2. [ugv_sdk -- Ranger 底盘通信 SDK](#2-ugv_sdk----ranger-底盘通信-sdk)
3. [TCP-IP-ROS-6AXis -- Dobot 六轴机械臂 SDK](#3-tcp-ip-ros-6axis----dobot-六轴机械臂-sdk)
4. [gripper -- DH-Robotics 夹爪驱动](#4-gripper----dh-robotics-夹爪驱动)
5. [lifting_ctrl -- 电动升降柱控制](#5-lifting_ctrl----电动升降柱控制)
6. [bt_task_msgs -- 自定义消息/服务定义](#6-bt_task_msgs----自定义消息服务定义)
7. [rslidar_sdk -- RoboSense 激光雷达驱动](#7-rslidar_sdk----robosense-激光雷达驱动)
8. [realsense-ros -- Intel RealSense 深度相机驱动](#8-realsense-ros----intel-realsense-深度相机驱动)
9. [SLAM -- 建图与激光里程计](#9-slam----建图与激光里程计)
10. [plugin -- RViz 多目标导航插件](#10-plugin----rviz-多目标导航插件)
11. [rangerboxcr10lidar_description -- 整机描述包](#11-rangerboxcr10lidar_description----整机描述包)

---

## 1. ranger_ros -- Ranger 底盘驱动

**路径:** `ranger_ros/`
**来源:** https://github.com/agilexrobotics/ranger_ros.git
**协议:** BSD 3-Clause

为 AgileX Ranger 移动机器人底盘提供 ROS 驱动，通过 CAN 总线与底盘通信。**当前项目使用的底盘为 Ranger 全尺寸版**（轮距 0.56m，轴距 0.90m，最大速度 2.7 m/s）。

### 包含的 ROS 包

| 包名 | 说明1 |
|------|------|
| `ranger_base` | 核心驱动节点（C++），发布里程计、系统状态、执行器状态、电池状态 |
| `ranger_msgs` | 自定义消息定义（SystemState、MotionState、ActuatorState 等） |
| `ranger_bringup` | 启动文件、地图、导航参数、CAN 配置脚本 |
| `sensor_msgs` | 本地打包的 sensor_msgs 包（确保版本兼容） |

### 实车适配要点（相对原始仓库）

`ranger_bringup` 在原始底盘驱动基础上补齐了完整的建图和导航栈：

- **雷达启动**：`launch/open_lidar.launch` 将 `rslidar_sdk`、`pointcloud_to_laserscan` 和 `rf2o_laser_odometry` 串联，把 RoboSense 3D 点云转换为 2D `/scan`，再输出 `/odom`。
- **雷达-底盘标定**：`open_lidar.launch` 中包含静态 TF `rslidar -> dummy_base_link`，位移 `-0.54 0 0.059972`，用于对齐雷达与底盘坐标。
- **GMapping 建图**：`launch/gmapping.launch`，设置 `odom_frame: odom`、`base_frame: dummy_base_link`、`map_frame: map`，参数调为室内建图优化（`maxUrange: 10.0`、`particles: 100`、`linearUpdate: 0.3`、`delta: 0.05`）。
- **导航栈**：`launch/navigation_4wd.launch`，集成 `map_server`、`amcl`、`move_base` 和 RViz，默认加载 `maps/map11.yaml`。
- **速度平滑**：`launch/volecity_smoother.launch` + `param/velocity_smoother.yaml`，限制线速度/角速度为 `0.5`，减小实车速度突变。
- **DWA 局部规划器**：`param/4wd/*` 导航参数，底盘 footprint 约 `1.10m x 0.90m`（`[[-0.55,-0.45], [-0.55,0.45], [0.55,0.45], [0.55,-0.45]]`）。
- **AMCL 定位**：`base_frame_id: rslidar`、`global_frame_id: world`，粒子数 `in_particles: 300`、`max_particles: 1500`。
- **地图与 RViz 配置**：`maps/*.pgm`、`maps/*.yaml`、`rviz/mapping.rviz`、`rviz/navigation.rviz`、`rviz/slam.rviz`。

### 运动模式（自动切换）

- **DUAL_ACKERMAN** -- 标准阿克曼转向（默认）
- **PARALLEL** -- 全轮平行转向（横向移动）
- **SPINNING** -- 原地旋转

节点根据 `/cmd_vel` 中 `linear.x`、`linear.y`、`angular.z` 的值自动选择运动模式。

### ROS 接口

**发布的话题:**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/system_state` | `ranger_msgs/SystemState` | 车辆状态、控制模式、错误码、电池电压 |
| `/motion_state` | `ranger_msgs/MotionState` | 当前运动模式 |
| `/actuator_state` | `ranger_msgs/ActuatorStateArray` | 8 个执行器状态 |
| `/odom` | `nav_msgs/Odometry` | 航迹推算里程计 |
| `/battery_state` | `sensor_msgs/BatteryState` | BMS 电池状态 |

**订阅的话题:**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/cmd_vel` | `geometry_msgs/Twist` | 速度控制命令 |

### 使用方法

```bash
# 首次设置 CAN 适配器
sudo modprobe gs_usb
rosrun ranger_bringup setup_can2usb.bash

# 后续启动（每次开机后执行一次）
rosrun ranger_bringup bringup_can2usb.bash

# 启动底盘驱动
roslaunch ranger_bringup ranger.launch

# 启动完整导航栈（含地图、AMCL、move_base）
roslaunch ranger_bringup navigation_4wd.launch

# 启动 SLAM 建图
roslaunch ranger_bringup gmapping.launch
```

### 依赖

- `ugv_sdk`（本工作空间内）
- `libasio-dev`、`libboost-all-dev`、`eigen3`

---

## 2. ugv_sdk -- Ranger 底盘通信 SDK

**路径:** `ugv_sdk/`
**来源:** https://github.com/westonrobot/ugv_sdk.git
**协议:** BSD

纯 C++ SDK 库（非 ROS 节点），通过 CAN 总线与 Ranger 底盘通信，是 `ranger_ros` 的底层依赖。本工作空间对其源码无定制修改，原样使用。

### 主要 API

`RangerRobot` 类（命名空间 `westonrobot`）提供 CAN 连接、运动命令和状态读取接口。

### CAN 适配器设置脚本

- `scripts/setup_can2usb.bash` -- 首次设置（加载 gs_usb 模块、启动 can0）
- `scripts/bringup_can2usb.bash` -- 启动 can0（每次开机执行）

---

## 3. TCP-IP-ROS-6AXis -- Dobot 六轴机械臂 SDK

**路径:** `TCP-IP-ROS-6AXis/`
**来源:** https://github.com/Dobot-Arm/TCP-IP-ROS-6AXis.git
**协议:** MIT

Dobot（越疆科技）六轴协作机械臂的官方 ROS SDK，通过 TCP/IP 协议与控制器通信。**当前项目使用的机械臂为 CR10**，适配主要通过运行时参数完成，未修改仓库源码。

### 通信架构

```
用户代码 / MoveIt / RViz
        | (ROS 话题、服务、ActionLib)
        v
dobot_v4_bringup
        | (TCP Socket)
        v
Dobot 控制器 (192.168.5.1)
Dobot 控制器 (192.168.8.188)
  端口 29999: Dashboard 命令（请求/响应，单客户端）
  端口 30004: 实时反馈（1440 字节二进制流，8ms 周期）
```

### 包含的 ROS 包

| 包名 | 说明 |
|------|------|
| `dobot_v4_bringup` | V4 控制器通信节点（90+ 个 ROS 服务，支持 FollowJointTrajectoryAction） |
| `dobot_moveit` | 统一 MoveIt 启动器（根据 DOBOT_TYPE 分发到 `cr10_moveit`） |
| `dobot_description` | URDF 模型（含 CR10） |

### 发布的话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/joint_states` | `sensor_msgs/JointState` | 6 个关节位置（10Hz） |

### 使用方法

```bash
# 仿真测试（无需实物）【可行】
export DOBOT_TYPE=cr10
roslaunch dobot_description display.launch        # RViz 关节滑块
roslaunch dobot_moveit demo.launch                # MoveIt 仿真
roslaunch dobot_gazebo gazebo.launch              # Gazebo 仿真

# 真实 CR10 控制
#  平板app里面需要设置模式“TCP/IP二次开发”
export DOBOT_TYPE=cr10
roslaunch dobot_v4_bringup bringup_v4.launch robotIp:=192.168.8.188

# MoveIt 运动规划（另一个终端）
export DOBOT_TYPE=cr10
roslaunch dobot_moveit moveit.launch
```

### 关键服务（部分）
**方式一：ROS Service 命令行**

```bash
# 使能机械臂（必须先执行）
rosservice call /dobot_v4_bringup/srv/EnableRobot

# 关节运动（移动到指定关节角度，单位：度）
rosservice call /dobot_v4_bringup/srv/JointMovJ "{j1: 0, j2: 0, j3: 0, j4: 0, j5: 0, j6: 0}"

# 笛卡尔运动（移动到指定位姿）
rosservice call /dobot_v4_bringup/srv/MovJ "{x: 200, y: 0, z: 200, rx: 180, ry: 0, rz: 0}"

# 直线运动
rosservice call /dobot_v4_bringup/srv/MovL "{x: 200, y: 0, z: 300, rx: 180, ry: 0, rz: 0}"

# 急停
rosservice call /dobot_v4_bringup/srv/EmergencyStop

# 清除错误
rosservice call /dobot_v4_bringup/srv/ClearError

# 下使能
rosservice call /dobot_v4_bringup/srv/DisableRobot
```

- `EnableRobot` / `DisableRobot` -- 使能/禁用
- `MovJ` / `MovL` / `JointMovJ` -- 关节/笛卡尔运动
- `ServoJ` / `ServoP` -- 实时伺服
- `EmergencyStop` / `ClearError` -- 急停/清除错误
- `DO` / `DI` / `AO` / `AI` -- 数字/模拟 IO

---

## 4. gripper -- DH-Robotics 夹爪驱动

**路径:** `gripper/`
**作者:** 深圳大寰机器人科技有限公司

DH-Robotics 系列电动夹爪的 ROS 驱动。**当前项目使用的夹爪为 AG95（Modbus RTU 协议）**，不在原始 OriGIT 项目中，为本工作空间新增的设备驱动包。

### 包含的 ROS 包

| 包名 | 说明 |
|------|------|
| `dh_gripper_msgs` | 自定义消息（GripperCtrl、GripperState） |
| `dh_gripper_driver` | 驱动节点、测试节点、关节状态发布节点 |

### ROS 接口

**订阅:**

| 话题 | 类型 | 说明 |
|------|------|------|
| `/gripper/ctrl` | `GripperCtrl` | 控制夹爪（位置 0-1000、力、速度、初始化） |

**发布:**

| 话题 | 类型 | 频率 | 说明 |
|------|------|------|------|
| `/gripper/states` | `GripperState` | 50Hz | 夹爪状态 |
| `/gripper/joint_states` | `sensor_msgs/JointState` | 50Hz | 关节状态（`gripper_finger1_joint`，0~0.637 rad） |

### 使用方法 [可行]

```bash
# 默认 AG95 Modbus 夹爪
roslaunch dh_gripper_driver dh_gripper.launch GripperModel:=AG95_MB Connectport:=/dev/ttyUSB0

# 启用测试节点（自动开合循环）
roslaunch dh_gripper_driver dh_gripper.launch test_run:=true
```

### 编程控制示例

```python
pub = rospy.Publisher('/gripper/ctrl', GripperCtrl, queue_size=50)
msg = GripperCtrl()
msg.initialize = False
msg.position = 500   # 0-1000
msg.force = 100      # 百分比
msg.speed = 100      # 百分比
pub.publish(msg)
```

---

## 5. lifting_ctrl -- 电动升降柱控制 (可用)

**路径:** `lifting_ctrl/`
**语言:** Python

通过 RS-485 串口（Modbus RTU 协议）控制电动升降柱（线性执行器），不在原始 OriGIT 项目中，为本工作空间新增的控制包。

### 支持的电机型号

| 型号 | 节点 | 编码器类型 |
|------|------|------------|
| 830ABS | `lifting_ctrl_service_node.py` | 绝对值编码器 |
| 850pro | `lifting_ctrl_service_node_850pro.py` | 增量编码器 |

### ROS 接口

**服务:**

| 服务名 | 类型 | 说明 |
|--------|------|------|
| `LiftingMotorService` | `LiftMotorSrv` | 直接电机控制 |
| `LiftingMotorService`（接口层） | `LiftInterfaceSrv` | 多电机路由/复用 |

**话题:**

| 话题 | 类型 | 说明 |
|------|------|------|
| `LiftMotorStatePub` | `LiftMotorMsg` | 电机状态（高度、速度、电流、限位等） |

### 服务模式（mode 参数）

| mode | 功能 | val 参数 |
|------|------|----------|
| 0 | 位置控制 | 目标高度（mm） |
| 1 | 初始化/归零 | 无 |
| -2 | 急停 | 无 |
| -3 | 清除错误 | 无 |
| -4 | 速度控制 | 目标速度（mm/s） |
| -5 | 设置最大速度 | 转速（RPM） |

### 配置文件

每个电机有独立的 JSON 配置文件，位于 `scripts/config/` 目录：

- `1_lifting_motor_config.json` -- 电机 ID 1（行程方向: `upLimitVal: 0`、`downLimitVal: -610`，最大转速 1500 rpm，堵转电流阈值 16 A）
- `2_lifting_motor_config.json` -- 电机 ID 2（端口 `/dev/lifter_2`，波特率 38400，行程 0 ~ 230 mm，最大转速 500 rpm，堵转电流阈值 5 A）
- `4_lifting_motor_config.json` -- 电机 ID 4（端口 `/dev/lifter_4`，波特率 38400，行程 0 ~ 230 mm）

### 使用方法

```bash
# 单电机启动
roslaunch lifting_ctrl start_motor.launch                    # 830ABS
roslaunch lifting_ctrl start_850pro_motor.launch             # 850pro

# 指定电机 ID 和命名空间
roslaunch lifting_ctrl start_motor.launch motor_id_:=2 lifter_ns:=lifter_2

# 发送控制命令
rosservice call /lifter_1/LiftingMotorService "val: 300
mode: 0"     # 移动到 300mm
rosservice call /lifter_1/LiftingMotorService "val: 0
mode: -2"    # 急停
```

### 依赖

- `pyserial`（`pip3 install pyserial`）
- `bt_task_msgs`（本工作空间内，提供消息/服务定义）

---

## 6. bt_task_msgs -- 自定义消息/服务定义

**路径:** `bt_task_msgs/`
**协议:** BSD

纯消息/服务定义包（无可执行代码），为升降机构控制提供 ROS 消息和服务类型。不在原始 OriGIT 项目中，为本工作空间新增。

### 定义的消息

| 消息 | 字段数 | 说明 |
|------|--------|------|
| `LiftMotorMsg` | 28 | 升降电机完整状态（ID、模式、电压、电流、速度、位置、高度、限位、错误码等） |

### 定义的服务

| 服务 | 说明 |
|------|------|
| `LiftMotorSrv` | 底层电机控制（val + mode -> state） |
| `LiftInterfaceSrv` | 上层接口（val + mode + id -> message + success + code） |

---

## 7. rslidar_sdk -- RoboSense 激光雷达驱动

**路径:** `rslidar_sdk/`
**来源:** https://github.com/RoboSense-LiDAR/rslidar_sdk.git
**协议:** BSD 3-Clause

RoboSense 激光雷达官方 ROS 驱动。**当前项目使用的雷达为 RS-Helios-16P**，已针对该型号和底盘安装环境做了配置适配。

### 实车适配要点（相对原始仓库）

- `config/config.yaml` 中雷达型号从默认 `RSM1` 改为 `RSHELIOS_16P`。
- 最小有效距离从 `0.2` 调整为 `0.45`，过滤车体/近距离噪声点。
- `rs_driver` 子模块已实体化为源码目录，方便 catkin 工作空间直接编译。

### 配置文件

`config/config.yaml` 关键配置：

```yaml
common:
  msg_source: 1            # 1=在线雷达
  send_point_cloud_ros: true

lidar:
  - driver:
      lidar_type: RSHELIOS_16P
      min_distance: 0.45
      msop_port: 6699
      difop_port: 7788
    ros:
      ros_frame_id: rslidar
      ros_send_point_cloud_topic: /rslidar_points
```

### 发布的话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/rslidar_points` | `sensor_msgs/PointCloud2` | 解码后的点云 |

### 使用方法

```bash
roslaunch rslidar_sdk start.launch
```

### 依赖

- `yaml-cpp`（>= 0.5.2）
- `libpcap`（>= 1.7.4）

---

## 8. realsense-ros -- Intel RealSense 深度相机驱动

**路径:** `realsense-ros/`
**版本:** 2.3.2
**协议:** Apache 2.0

Intel RealSense 深度相机的官方 ROS 封装。当前工程仅对编译依赖做了轻量补充（在 CMakeLists.txt 中增加 OpenCV 查找和链接），未针对具体相机型号做 launch 参数改动。

### 发布的话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB 彩色图像 |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | 深度图（16位） |
| `/camera/infra1/image_rect_raw` | `sensor_msgs/Image` | 左红外 |
| `/camera/infra2/image_rect_raw` | `sensor_msgs/Image` | 右红外 |
| `/camera/depth/color/points` | `sensor_msgs/PointCloud2` | 彩色点云 |

### 使用方法

```bash
# 基本启动
roslaunch realsense2_camera rs_camera.launch

# 启用点云
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud

# 深度对齐到彩色
roslaunch realsense2_camera rs_camera.launch align_depth:=true

#rviz可视化彩色点云
roslaunch realsense2_camera demo_pointcloud.launch 
```

### 依赖

- `librealsense2`（>= 2.50.0）
- `cv_bridge`、`image_transport`

---

## 9. SLAM -- 建图与激光里程计

**路径:** `SLAM/`

激光 SLAM 相关算法包的集合，配合 Ranger + RoboSense 的 2D 建图导航链路使用。

### 包含的 ROS 包

| 包名 | 来源 | 说明 |
|------|------|------|
| `openslam_gmapping` | ros-perception | GMapping 核心算法库 |
| `slam_gmapping` | ros-perception | GMapping ROS 封装节点 |
| `pointcloud_to_laserscan` | ros-perception | RoboSense 3D 点云转 2D `/scan` |
| `laser_scan_matcher` | ccny-ros-pkg | 基于 CSM 的增量激光扫描匹配 |
| `rf2o_laser_odometry` | MAPIRlab | 基于 Range Flow 的激光里程计 |

### RF2O 激光里程计（实车适配）

相对原始仓库的关键改动：

- 输入话题从 `/laser_scan` 改为 `/scan`，接 `pointcloud_to_laserscan` 输出。
- 输出里程计从 `/odom_rf2o` 改为 `/odom`，直接供 GMapping、AMCL 和 move_base 使用。
- `publish_tf` 从 `false` 改为 `true`，由 RF2O 发布 odom TF。
- `base_frame_id` 去掉前导 `/`，改为 `base_link`。
- `init_pose_from_topic` 清空，移除仿真依赖。
- 运行频率从 `6.0` 提高到 `10`。
- 源码中增加 `std::isfinite(dcenter)` 保护，避免 NaN/Inf 进入计算。

### GMapping 关键参数（ranger_bringup/gmapping.launch）

| 参数 | 值 | 说明 |
|------|-----|------|
| `~base_frame` | `dummy_base_link` | 对应雷达标定后的基座坐标系 |
| `~odom_frame` | `odom` | 里程计坐标系 |
| `~map_frame` | `map` | 地图坐标系 |
| `~particles` | 100 | 粒子数 |
| `~delta` | 0.05 | 地图分辨率（米/像素） |
| `~linearUpdate` | 0.3 | 触发更新的最小线性位移（米） |
| `~maxUrange` | 10.0 | 最大有效测距（米） |

---

## 10. plugin -- RViz 多目标导航插件

**路径:** `plugin/rviz_navi_multi_goals_pub_plugin/`
**来源:** https://github.com/autolaborcenter/rviz_navi_multi_goals_pub_plugin.git
**包名:** `navi_multi_goals_pub_rviz_plugin`

RViz 面板插件，在原始多目标导航功能基础上扩展了建图、地图管理和导航启动的一体化控制。

### 实车适配要点（相对原始仓库）

- 新增建图页：地图列表 + `record` 按钮，用于启停建图并保存地图。
- 点击地图后可启动 `map_server`、`rf2o_laser_odometry`、导航 launch，并切换到对应地图。
- 保存地图时调用 `map_saver`，并停止 `slam_gmapping` 和 `rf2o_laser_odometry_node`。
- 建图流程中按需终止 `move_base`、`amcl` 等节点，避免建图和导航同时运行。
- 原始多目标点逻辑保留：订阅临时目标点、按队列发布到 `/move_base_simple/goal`、监听 `/move_base/status`。

### 功能

1. **建图 Tab** -- 启停 SLAM 建图、保存地图、选择已有地图
2. **导航 Tab** -- 多目标点队列、循环巡航、暂停/取消

### 使用方法

1. 启动导航栈
2. 在 RViz 中加载插件：`Panels -> Add New Panel -> MultiNaviGoalsPanel`
3. 添加 Marker 显示：`Display -> add -> Marker`
4. 重配置 2D Nav Goal 工具：`Tool Property` 中将 topic 改为 `/move_base_simple/goal_temp`
5. 在地图上点击设置目标点，点击 "Start navigation" 开始巡航

---

## 11. rangerboxcr10lidar_description -- 整机描述包

**路径:** `rangerboxcr10lidar_description/`

不在原始 OriGIT 项目中，为本工作空间新增的整机 URDF/mesh 集成包，将所有设备合并到统一的 TF 树中。

### 主要内容

- `urdf/rangercr10lidar.urdf`（`<robot name="rangercr10lidar">`）：整合 Ranger 底盘、CR10 机械臂、AG95 夹爪、RoboSense 雷达、RealSense D435 和上装箱体。
- `cr10_mount_joint`：将 `cr10_base_link` 固定到底盘 `base_link`。
- AG95 固定到 `cr10_Link6`，单个 `gripper_finger1_joint` 通过 mimic joint 驱动夹爪连杆联动。
- 雷达通过 `lidar_joint0` 固定在 `base_link` 上。
- RealSense D435 通过 `d435_joint` 固定到 `lidar_link0` 上方。
- `config/joint_names_rangerboxcr10lidar_description.yaml`：整合底盘转向/轮子关节和 CR10 六轴关节。

### 启动

```bash
roslaunch rangerboxcr10lidar_description display.launch
```

---

## 整体架构示意

```
+------------------+     +------------------+     +------------------+
|   ranger_ros     |     |  TCP-IP-ROS-6AXis|     |   realsense-ros  |
| (Ranger 底盘驱动) |     | (Dobot CR10 SDK) |     | (RealSense D435) |
+--------+---------+     +--------+---------+     +--------+---------+
         |                         |                         | USB
         | CAN Bus                 | TCP/IP                  v
         v                         v                +--------+---------+
+--------+---------+     +--------+---------+       |  RealSense 硬件  |
|     ugv_sdk      |     |  Dobot 控制器    |       +------------------+
| (底盘通信SDK库)    |     | (192.168.5.1)   |
+------------------+     +------------------+

+------------------+     +------------------+     +------------------+
|    gripper       |     |  lifting_ctrl    |     |    bt_task_msgs  |
| (AG95 夹爪驱动)   |     | (升降柱控制)      |     | (消息/服务定义)   |
+--------+---------+     +--------+---------+     +------------------+
         |                         | RS-485/Modbus
         | Serial/Modbus           v
         v                +--------+---------+
+--------+---------+      |  电动升降柱电机   |
| DH-Robotics AG95 |      +------------------+
+------------------+

+------------------+     +------------------+     +------------------+
|    rslidar_sdk   |     |      SLAM        |     |     plugin       |
| (RS-Helios-16P)  |     | (gmapping/RF2O)  |     | (RViz多目标插件)  |
+--------+---------+     +--------+---------+     +------------------+
         |                         |
         | UDP                     | /scan + TF
         v                         v
+--------+---------+     +--------+---------+
| RS-Helios-16P    |     |   move_base     |
+------------------+     | (Navigation)    |
                         +------------------+
```

---

## 依赖关系

```
ranger_ros
  +-- ugv_sdk
  +-- ranger_msgs

lifting_ctrl
  +-- bt_task_msgs

gripper
  +-- dh_gripper_msgs（内部包）

TCP-IP-ROS-6AXis
  （无外部工作空间依赖）

rslidar_sdk
  （无外部工作空间依赖）

realsense-ros
  +-- librealsense2（系统库）

SLAM
  （无外部工作空间依赖）

plugin
  （无外部工作空间依赖，依赖 rviz、move_base 等系统包）
```

---

## 注意事项

1. **CAN 适配器**：使用 Ranger 底盘前，每次开机需执行 `bringup_can2usb.bash` 启动 CAN 接口
2. **环境变量**：使用 Dobot 机械臂前需设置 `export DOBOT_TYPE=cr10`
3. **串口权限**：使用夹爪和升降柱时，确保当前用户有串口访问权限（通常需加入 `dialout` 组）
4. **编译顺序**：建议先编译 `bt_task_msgs`，再编译 `lifting_ctrl`（因为后者依赖前者的消息定义）
5. **雷达标定**：`rslidar_sdk` 的 `config/config.yaml` 已配置为 RS-Helios-16P，修改前确认实际雷达型号
6. **导航 launch 注意**：`navigation_4wd.launch` 末尾包含 `ranger_mini_v2.launch`，当前底盘为 Ranger 全尺寸，真机部署前建议确认 launch 文件是否匹配
7. **插件路径**：RViz 插件源码中仍有部分历史包名（如 `agilexpro`），若一键建图/导航功能异常需检查路径是否与当前 `ranger_bringup` 一致
