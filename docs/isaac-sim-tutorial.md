# Isaac Sim 仿真教程

> 🧪 了解 USD、PhysX、传感器和 GPU 并行仿真，再进入 Isaac Lab。

**预计阅读**：20 min  
**前置知识**：Python、基本机器人学和 RL transition  
**下一步**：[强化学习基础](reinforcement-learning.md) · [代码仓](codebases.md)

**本文路线**：安装 → URDF/USD → Python 生命周期 → 状态与传感器 → RL transition → Isaac Lab

Isaac Sim 适合复杂场景、相机/激光传感器和 GPU 物理。通常先将 URDF 导入 Isaac Sim，再在 USD 场景里继续编辑、加传感器和跑物理。Isaac Lab 是建在 Isaac Sim 之上的机器人学习框架，负责并行环境和训练；它不是另一个物理引擎。

官方入口：

- [Isaac Sim 文档](https://docs.isaacsim.omniverse.nvidia.com/)
- [快速安装](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/quick-install.html)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)

本文只讲最小流程：导入机器人、看懂场景对象、读写关节状态、推进仿真，再接 Isaac Lab。Isaac Sim 的版本、驱动和 Python API 变化很快，具体安装命令以你使用的版本文档为准。

## 1. 运行方式和安装

| 方式 | 用途 |
| --- | --- |
| GUI standalone | 创建场景、查看机器人和传感器 |
| headless | 服务器运行、批量评测和 CI |
| streaming | 远程运行、本地通过 WebRTC 查看 |
| Isaac Lab | 在 Isaac Sim 上做并行机器人学习 |

Isaac Sim 对 NVIDIA 驱动、GPU 显存和磁盘空间要求较高。下载与系统匹配的 standalone 包，按官方文档执行安装脚本。Windows 使用对应的 .bat 或 .exe；Linux 命令不能直接照搬到 Windows。本文没有在当前环境启动 Isaac Sim，不能保证示例在所有版本上无需调整。

## 2. URDF 导入和 USD 运行

URDF 适合交换机器人本体信息：链接、关节、惯性、视觉和碰撞几何。Isaac Sim 导入 URDF 后会生成 USD Prim 和关节结构，后续场景编辑、传感器、物理和保存都围绕 USD 进行。

导入后至少检查：

- 根 Prim 路径、关节名称和关节顺序；
- 长度/角度单位、质量和惯性；
- 网格材质、碰撞几何和关节限位；
- 驱动目标类型，以及传感器和控制器是否需要额外配置。

URDF 不是完整的 USD 场景文件，也不会自动包含你的房间、灯光、相机和任务逻辑。导入成功只说明资产进入 Stage，不代表机器人已经可以直接训练。

## 3. 先认识四个对象

- **Stage**：当前 USD 场景树，包含机器人、物体、灯光、相机和物理设置。
- **Prim**：Stage 中的一个 USD 节点，例如 /World/Robot 或相机。
- **Articulation**：由关节连接的一组刚体，通常对应一个机器人。
- **Physics step**：PhysX 推进一步的时间；渲染、控制器和策略可以使用不同频率。

先在 GUI 中打开空场景，创建地面和光源，再加入一个机器人资产。确认 Prim 路径、关节名称、单位（通常是米和弧度）及碰撞几何。

## 4. 最小 Python 生命周期

不同版本的模块路径可能变化，下面只展示结构：

~~~python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / 120.0,
    rendering_dt=1.0 / 60.0,
)
robot = Articulation(prim_path="/World/Robot", name="robot")
world.scene.add(robot)
world.reset()

for _ in range(1000):
    q = robot.get_joint_positions()
    robot.set_joint_position_targets(q)
    world.step(render=False)

simulation_app.close()
~~~

生命周期顺序是：启动应用，创建或加载 Stage，注册机器人和传感器，reset，循环读取观测并写入动作，调用 world.step，最后关闭应用。实际导入名以当前版本 API 文档为准。

## 5. 状态、动作和传感器

常见关节接口的语义如下（具体函数名随版本变化）：

~~~python
q = robot.get_joint_positions()       # [num_joints]
qd = robot.get_joint_velocities()     # [num_joints]
effort = robot.get_applied_joint_efforts()
robot.set_joint_position_targets(q_target)
robot.set_joint_velocity_targets(qd_target)
# 或 robot.set_joint_efforts(tau_target)
~~~

位置目标、速度目标和力矩目标不能混用。动作如果是末端位姿或 delta pose，还需要 IK 或轨迹控制器转换为关节目标。日志中同时保存策略输出和真正送入机器人的目标，因为限幅、滤波和控制器可能改变后者。

相机、深度、语义分割、激光、触觉和 IMU 都是独立传感器。记录分辨率、内外参、坐标系、频率、延迟和时间戳；图像、位姿和动作不一定在同一个 physics step 更新。

## 6. 时间步和 RL transition

- physics_dt：物理积分步长。
- rendering_dt：渲染或相机更新周期。
- control_dt：低层控制器接收新目标的周期。
- policy_dt：策略网络产生新动作的周期。
- decimation：一个策略动作保持的物理步数，通常为 policy_dt / physics_dt。

例如 physics_dt=1/120、policy_dt=1/30 时，一个动作保持 4 个物理步。改变 policy_dt 会同时改变控制带宽、样本数量和结果可比性。

RL 环境的一步通常记录：

~~~text
(obs, action, reward, next_obs, terminated, truncated, info)
~~~

obs 和 next_obs 是前后观测；action 是策略输出或控制器目标；reward 是即时奖励；terminated 表示任务自然结束；truncated 表示达到时间或步数上限；info 保存成功、碰撞和奖励分项等调试信息。

## 7. Isaac Lab 入门

需要 PPO、SAC 或大规模并行环境时，使用与 Isaac Sim 版本匹配的 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)：

1. 先运行一个官方示例，确认 GUI 和 headless 都能启动。
2. 阅读任务的 scene、action、observation、reward 和 termination 配置。
3. 用很小的 num_envs 和固定 seed 检查 reset、观测范围、奖励和显存。
4. 再接 RSL-RL、RL-Games、SKRL 或自定义 trainer。

Isaac Sim 负责 Stage、PhysX、渲染和传感器；Isaac Lab 负责机器人学习层。能在 Isaac Sim 脚本里控制机器人，不代表已经建立了 Isaac Lab RL 环境。

## 8. 建议的自行实验

每次只改一个变量：

1. 移动物体或改 Prim 属性，观察 Stage 和碰撞结果。
2. 改 physics_dt、policy_dt 或 rendering_dt，比较轨迹、图像数量和吞吐。
3. 逐个加入相机、深度或激光，检查时间戳和坐标变换。
4. 增大 num_envs，观察 GPU 显存、env steps/s 和 wall-clock。

这些实验跑通后，再自行扩展到操作任务、视觉策略、sim-to-real 或自主探索。

## 9. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 驱动或 CUDA 不匹配 | Isaac Sim 版本支持矩阵、NVIDIA 驱动和 GPU 型号 |
| 资产加载失败 | 资产服务器、代理、证书和本地缓存 |
| headless 立即退出 | 官方示例、SimulationApp 参数和启动环境 |
| 机器人不动 | Prim 路径、关节顺序、目标类型和控制器 |
| 显存爆满 | num_envs、相机分辨率、传感器数量和 reset |
| 结果无法复现 | 版本、GPU、seed、资产、随机化和控制频率 |
