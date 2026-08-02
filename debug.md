当前效果（Stage-5 完整任务策略实测）
用最新 checkpoint（2026-08-01_23-13-08，训练 199 迭代）跑 25.6 万步实测：

指标	值	含义
EE-目标距离	mean 0.38m / 中位 0.24m / 10% 分位 0.07m	有一半时间 EE 离目标不到 24cm，最好时 7cm
到达成功率	<10cm 占 18.6%、<15cm 占 32.1%	近 1/3 时间 EE 真正够到目标
base 位移	mean 0.52m / max 0.96m	base 在主动导航，把远处的扩展目标移进臂展
终止	0	无异常终止，episode 全跑满 10s
具体行为
策略学会了**"base 导航到目标附近 → arm 够取"**的协同：

30% 可达目标（0.2-0.5m）：arm 直接够取，EE 距离可到 7cm
70% 扩展目标（0.5-1.2m）：base 会移动（平均 0.5m，最远 0.96m）去靠近
全程 base 稳定（倾斜 <0.1°）、无 NaN、无越限终止
诚实评估
这是一个可工作的基线，但未完全收敛：

reward +7.85、best +32.6 —— 正分但还能更高
p90=0.87m —— 10% 时间（多是不顺手的扩展目标方向）还没够到
只训了 200 迭代（~5 分钟）；加长训练（max_iterations: 200 → 2000+）会显著提升到达率


底盘移动方法：不是 agx，是 A+ kinematic 方案
不是用 agx/ 里的东西。 那个 agx/ 目录（README 写着 # agilex_ws）是真实机器人的 ROS 1 工作空间——包含 AgileX 底盘驱动（ugv_sdk）、Dobot CR5 机械臂控制（TCP-IP-ROS-6AXis）、夹爪、激光雷达、SLAM 等，是实机软件栈，只是放在仓库里参考，MuJoCo 仿真完全不用它。

底盘移动用的是这套环境自带的 A+ kinematic base scheme（无轮地接触物理）：

1. 底盘是 freejoint（自由体），不是轮子驱动的
robot.xml:32：base body 上挂 <freejoint/>，所有 base 和轮子的 geom 都是 contype="0" conaffinity="0" —— 零碰撞，轮子物理上从不接触地面。

2. 速度直接写进 qvel
base_velocity_controller.py:140 apply_velocity() 调用 set_root_planar_velocity(v_world[:,:2], w_world[:,2])，而 backend.py 的实现是直接改 freejoint 的 qvel（vx/vy/yaw，同时把 vz/wx/wy 清零做 SE(2) 平面锁）。


策略动作 → BaseVelocityController（一阶滤波）→ set_root_planar_velocity → 写 freejoint qvel → MuJoCo 按速度积分
3. 轮子纯可视化
base_velocity_controller.py:150：用 _compute_wheel_ik 从底盘速度算转向角 + 轮转速，写进 set_joint_qpos/set_joint_qvel，只是让轮子看起来在转，不产生任何驱动力。

为什么这么设计
这是当初评审定的 A+ 方案（也写进了 CLAUDE.md）：移动底盘做成"运动学底座"——用速度指令直接控制位置，而不是用轮地摩擦 + 底盘动力学。好处是训练稳定（没有轮子打滑/陷地/动力学数值问题），坏处是仿真里没有真实轮地物理（加速/转向的动力学效应是简化的）。

所以：底盘运动学完全是我们自己实现的，跟 agx 的实机底盘驱动没有任何关系。 如果你是想让仿真行为逼近真实 AgileX 底盘，那是另一件事（需要在 A+ 速度模型里加真实底盘的运动学约束），可以后面讨论。