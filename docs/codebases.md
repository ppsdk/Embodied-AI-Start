# 代码仓、数据与基准

> 🛠️ 按研究路线查找可运行代码、数据集、仿真环境和基准实现。

**预计阅读**：15 min<br>
**前置知识**：Git、Python 环境和基础命令行<br>
**下一步**：[学习路线](roadmap.md) · [MuJoCo 教程](mujoco-tutorial.md) · [Isaac Sim 教程](isaac-sim-tutorial.md)

**本文路线**：核心仓库 → 模型/策略 → RL/MBRL → WM/WAM → 仿真与真机

## 1. 核心仓库

| 项目         | 定位                                                    | 推荐入口                                                                                                                                        |
| ------------ | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| StarVLA      | 模块化 VLA 研究与工程平台                               | [GitHub](https://github.com/starVLA/starVLA) · [Docs/Project](https://starvla.github.io/) · [Paper](https://arxiv.org/abs/2604.05014)                  |
| RLinf        | 面向具身/智能体基础模型的可扩展 RL 后训练基础设施       | [GitHub](https://github.com/RLinf/RLinf) · [Docs](https://rlinf.readthedocs.io/) · [Paper](https://arxiv.org/abs/2509.15965)                           |
| FastWAM      | Fast-WAM 官方训练与评测代码                             | [GitHub](https://github.com/yuantianyuan01/FastWAM) · [Project](https://yuantianyuan01.github.io/FastWAM/) · [Paper](https://arxiv.org/abs/2603.16666) |
| OpenPI       | Physical Intelligence 的 π0 / π0.5 开源实现与模型入口 | [GitHub](https://github.com/Physical-Intelligence/openpi) · [π0.5 Paper](https://arxiv.org/abs/2504.16054)                                          |
| bimanual-vla | 双臂 VLA 真机部署入口与运行参考                         | [GitHub](https://github.com/SUNNYsyy2005/bimanual-vla)                                                                                             |
| XPolicyLab   | 策略适配、服务化部署与跨 benchmark 评测连接             | [GitHub](https://github.com/XPolicyLab/XPolicyLab) · [Website](https://xpolicylab.github.io/) · [教程](xpolicylab-tutorial.md) |
| Piper ROS    | AgileX Piper 机械臂的 ROS 2 Humble 驱动、URDF、MoveIt 2、Gazebo、MuJoCo 与 CAN 配置资源 | [GitHub humble](https://github.com/agilexrobotics/piper_ros/tree/humble) · [上游 README](https://github.com/agilexrobotics/piper_ros/blob/humble/README.MD) |

ROS 2 Humble 的基础机器人学工程建议组合使用以下官方仓库，而不是寻找一个把 TF、RViz 2、MoveIt 2 和控制器全部揉在一起的仓库：

| 组件 | 用途 | 推荐版本/入口 |
| --- | --- | --- |
| MoveIt 2 Tutorials | Panda 配置、RViz 2 quickstart、规划与执行示例 | [GitHub](https://github.com/moveit/moveit2_tutorials/tree/humble) · [RViz quickstart](https://github.com/moveit/moveit2_tutorials/tree/humble/doc/tutorials/quickstart_in_rviz) |
| ROS 2 geometry2 | TF2 Python 广播、监听、等待变换和 frame dump 示例 | [GitHub](https://github.com/ros2/geometry2/tree/humble/examples_tf2_py) |
| MoveIt Resources | Panda、Fanuc、PR2 等 URDF 和 MoveIt 配置资源 | [GitHub](https://github.com/moveit/moveit_resources/tree/ros2) |
| ros2_control_demos | `ros2_control` 和控制器的 RRBot 等示例；不是 MoveIt 教程本身 | [GitHub](https://github.com/ros-controls/ros2_control_demos/tree/humble) |
| MoveIt Calibration | MoveIt 维护的手眼标定工具入口，支持 eye-in-hand 与 eye-to-hand 的标定流程 | [GitHub](https://github.com/moveit/moveit_calibration) |
| hand-eye_calibration | ROS 2 Humble 社区实现，带 PyQt5 GUI、ArUco 和多种求解器；硬件支持以项目 README 为准 | [GitHub](https://github.com/hhanoo/hand-eye_calibration/tree/humble) |

StarVLA 与 RLinf 的直接组合示例：

- [RL on StarVLA Models](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html)
- [RL with ManiSkill Benchmark](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/maniskill.html)

## 2. 模型基础

这些仓库用于理解模型本身。

| 项目          | 原语                   | 简介                                                       | 链接                                                                                                 |
| ------------- | ---------------------- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Transformers  | Transformer backbone   | token、attention、mask、预训练模型和 `[B,L,D]` 张量流   | [GitHub](https://github.com/huggingface/transformers) · [Docs](https://huggingface.co/docs/transformers/) |
| Diffusers     | Diffusion / DiT 工具链 | scheduler、epsilon/x0/v 参数化和扩散模型实验              | [GitHub](https://github.com/huggingface/diffusers) · [Docs](https://huggingface.co/docs/diffusers/)       |
| Flow Matching | Flow matching          | probability path、velocity field 和 ODE 采样实现          | [GitHub](https://github.com/facebookresearch/flow_matching)                                             |
| DiT           | Diffusion Transformer  | Transformer 作为 latent diffusion backbone 的基础研究代码 | [GitHub](https://github.com/facebookresearch/DiT)                                                       |

## 3. VLA 与动作策略

| 项目             | 简介                                    | 链接                                                                                                                  |
| ---------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| OpenVLA          | 阅读一个完整开源 VLA 的训练、微调和部署 | [GitHub](https://github.com/openvla/openvla)                                                                             |
| OpenVLA-OFT      | OpenVLA 的效率优化微调与推理           | [GitHub](https://github.com/moojink/openvla-oft)                                                                         |
| OpenPI           | π0 / π0.5 的开源实现、配置与推理调用  | [GitHub](https://github.com/Physical-Intelligence/openpi) · [Docs](https://github.com/Physical-Intelligence/openpi#readme) |
| Octo             | 通用机器人策略与多数据集预训练/微调     | [GitHub](https://github.com/octo-models/octo)                                                                            |
| Isaac GR00T      | NVIDIA 的人形机器人基础模型、数据管线与训练入口 | [GitHub](https://github.com/NVIDIA/Isaac-GR00T) · [项目页](https://developer.nvidia.com/isaac/gr00t) |
| Diffusion Policy | diffusion 动作策略的官方实现            | [GitHub](https://github.com/real-stanford/diffusion_policy)                                                              |
| ACT              | 低成本双臂操作与 action chunking        | [GitHub](https://github.com/tonyzhaozh/act)                                                                              |
| LeRobot           | Hugging Face 的低成本机器人学习工具链，包含 SmolVLA、数据采集和策略训练入口 | [GitHub](https://github.com/huggingface/lerobot) · [Docs](https://huggingface.co/docs/lerobot/) |

## 4. RL 与 MBRL

| 项目              | 类型                                                               | 链接                                                                                                 |
| ----------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| RLinf             | VLA/基础模型的大规模 RL 后训练                                     | [GitHub](https://github.com/RLinf/RLinf)                                                                |
| AReaL             | 支持 GRPO、SAPO 等基础模型 RL 后训练算法                           | [GitHub](https://github.com/areal-project/AReaL) · [算法文档](https://github.com/areal-project/AReaL/blob/main/docs/zh/algorithms/grpo_series.md) |
| CleanRL           | 单文件 RL 实现，包含 PPO/SAC/DQN 等算法                            | [GitHub](https://github.com/vwxyzjn/cleanrl) · [Docs](https://docs.cleanrl.dev/)                          |
| CleanRL MuJoCo PPO | Gymnasium MuJoCo 连续动作 PPO 单文件示例；默认 `HalfCheetah-v4` | [代码](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) · [MuJoCo 依赖](https://github.com/vwxyzjn/cleanrl/blob/master/requirements/requirements-mujoco.txt) |
| RL Baselines3 Zoo | 基于 Stable-Baselines3 的训练、评测、调参和视频脚本；适合批量比较环境/算法 | [GitHub](https://github.com/DLR-RM/rl-baselines3-zoo) · [Docs](https://rl-baselines3-zoo.readthedocs.io/) |
| Stable-Baselines3 | 易用的经典 model-free RL 基线                                      | [GitHub](https://github.com/DLR-RM/stable-baselines3) · [Docs](https://stable-baselines3.readthedocs.io/) |
| DQN Zoo           | DQN 及其变体的参考实现                                             | [GitHub](https://github.com/google-deepmind/dqn_zoo)                                                     |
| d3rlpy            | Offline RL 算法与数据处理                                          | [GitHub](https://github.com/takuseno/d3rlpy) · [Docs](https://d3rlpy.readthedocs.io/)                     |
| Implicit Q-Learning | IQL 参考实现                                                       | [GitHub](https://github.com/ikostrikov/implicit_q_learning)                                              |
| TD-MPC2           | 潜空间动力学 + MPC 的 MBRL 基线                                    | [GitHub](https://github.com/nicklashansen/tdmpc2) · [Project](https://www.tdmpc2.com/)                    |
| DreamerV3         | 用 latent dynamics 做 imagined rollout 和 actor-critic 更新的 MBRL | [GitHub](https://github.com/danijar/dreamerv3)                                                          |
| Minari            | Offline RL 数据集 API 与数据目录                                   | [GitHub](https://github.com/Farama-Foundation/Minari) · [Docs](https://minari.farama.org/)                |

WAM 相关的代码和索引入口：

- [FastWAM](https://github.com/yuantianyuan01/FastWAM)：测试时未来想象成本与动作生成的基线。
- [Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM)：WAM 论文和项目索引。

近期论文中的新方法不一定已经有公开代码。要复现时，先从[论文清单](papers.md)的 arXiv 页面进入，再确认作者仓库、checkpoint、数据和评测脚本是否真的公开。

## 5. World Model：像素、latent 与 3D/4D

World Model 在这里是广义的环境表征、未来预测和场景生成路线；动作决策能力根据各仓库的任务和评测单独记录。

| 项目                  | 方向                              | 推荐用途                                     | 链接                                                                                                       |
| --------------------- | --------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| V-JEPA 2              | JEPA / latent predictive learning | 未来表征、物理理解与动作条件研究入口         | [GitHub](https://github.com/facebookresearch/vjepa2) · [Paper](https://arxiv.org/abs/2506.09985)                |
| Cosmos Predict2       | 视频世界模型 / physical AI 生成   | 视频未来预测、数据生成与世界模型连接方式探索     | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2) · [Project](https://www.nvidia.com/en-us/ai/cosmos/) |
| LPWM                  | 对象中心 latent world model       | 从视频发现 latent particles，学习随机、动作/语言/目标条件动态 | [GitHub](https://github.com/taldatech/lpwm) · [Paper](https://arxiv.org/abs/2603.04553) · [Project](https://taldatech.github.io/lpwm-web/) |
| GWM                   | 3D Gaussian world model           | 在 Gaussian primitives 上预测动作条件未来，可作视觉表征或 neural simulator | [GitHub](https://github.com/Gaussian-World-Model/gaussianwm) · [Paper](https://arxiv.org/abs/2508.17600) · [Project](https://gaussian-world-model.github.io/) |
| VGGT                  | 3D 几何与多视图场景表示           | 相机/深度/点云几何，为 3D WM 提供结构化表征  | [GitHub](https://github.com/facebookresearch/vggt)                                                            |
| 3D Gaussian Splatting | 3D 场景表示与新视角合成           | 动态/可渲染场景表示的基础组件，不等于完整 WM | [GitHub](https://github.com/graphdeco-inria/gaussian-splatting)                                               |
| OccWorld              | 3D occupancy world model          | 读取离散 occupancy token，预测未来场景和 ego trajectory；适合学习 4D occupancy 基线 | [GitHub](https://github.com/wzzheng/OccWorld) · [Paper](https://arxiv.org/abs/2311.16038) |
| SparseWorld            | 稀疏 4D occupancy world model      | 用稀疏、动态 query 预测未来 occupancy，关注效率和长时程场景生成 | [GitHub](https://github.com/MSunDYY/SparseWorld) · [Paper](https://arxiv.org/abs/2510.17482) |
| HY-World 2.0           | 多模态 3D world generation        | 文本/图像/视频生成可导航 3DGS，并提供交互式 3D 场景工具链 | [GitHub](https://github.com/Tencent-Hunyuan/HY-World-2.0) · [Paper](https://arxiv.org/abs/2604.14268) |
| PhysMani               | 物理约束的 3D Gaussian WM          | 复现动态操作中的 Gaussian velocity field 与策略融合 | [GitHub](https://github.com/vLAR-group/PhysMani) · [Paper](https://arxiv.org/abs/2607.01938) |
| IRIS                  | 离散 token 自回归 WM               | Atari100k 少样本环境建模、离散 latent rollout 与模型内 RL | [GitHub](https://github.com/eloialonso/iris) · [Paper](https://arxiv.org/abs/2209.00588) |
| DIAMOND               | 像素 diffusion WM                  | 在 Atari/视频环境中研究视觉细节、交互式神经游戏引擎和模型内 RL | [GitHub](https://github.com/eloialonso/diamond) · [Paper](https://arxiv.org/abs/2405.12399) · [Project](https://diamond-wm.github.io) |
| Dynalang              | 语言条件 latent WM                | 让环境规律描述参与未来表征预测和 imagined rollout | [GitHub](https://github.com/jlin816/dynalang) · [Paper](https://arxiv.org/abs/2308.01399) |
| GNS                   | 图网络物理模拟器                  | 粒子级流体、刚体和可变形物体动力学；作为物理 WM 前端 | [GitHub](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) · [Paper](https://arxiv.org/abs/2002.09405) |
| DriveDreamer         | 驾驶视频 diffusion WM              | 交通结构约束、动作条件驾驶视频预测和数据生成 | [GitHub](https://github.com/JeffWang987/DriveDreamer) · [Paper](https://arxiv.org/abs/2309.09777) |
| DreamDojo             | 人类视频预训练机器人 WM            | 用连续 latent action 迁移交互知识，再用少量机器人数据做动作校准 | [Project](https://dreamdojo-world.github.io/) · [Paper](https://arxiv.org/abs/2602.06949) |
| PlayWorld             | 自主探索机器人 WM                  | 用机器人 autonomous play 收集长尾接触/失败数据，训练操作视频模拟器 | [Project](https://robot-playworld.github.io/) · [Paper](https://arxiv.org/abs/2603.09030) |
| Causal-JEPA           | 因果/对象级 latent WM              | 对象级 masking 和关系预测，研究干预与规划所需的结构化表征 | [Paper](https://arxiv.org/abs/2602.11389) · [Code](https://github.com/galilai-group/cjepa) |
| WorldEcho/WorldSync   | 动作跟随与 WM 安全评测             | 测量 off-expert action 是否真的改变未来，并接入风险 head/shield | [Paper](https://arxiv.org/abs/2608.24885) |
| ReWorld               | 长时程交互式视频 WM                | 混合 attention + 位姿索引记忆库，学习长时程回访和实时交互 | [Paper](https://arxiv.org/abs/2608.23565) |
| PETS                  | 概率 dynamics + 采样式 MBRL         | ensemble dynamics、不确定性传播和 trajectory sampling | [Paper](https://arxiv.org/abs/1805.12114) |
| DayDreamer            | 真实机器人在线 WM/MBRL              | 在四足、机械臂和移动机器人上直接收集数据并做 Dreamer imagined rollout | [Project](https://danijar.com/project/daydreamer/) · [Paper](https://arxiv.org/abs/2206.14176) |
| SlotFormer            | 对象中心 slot dynamics              | 在无对象标签的 slot 表征上预测对象关系和未来状态 | [Project](https://slotformer.github.io/) · [Paper](https://arxiv.org/abs/2210.05861) |
| FOCUS                 | 对象中心探索 WM                    | 用对象级预测误差构造探索奖励，主动收集机器人-物体交互 | [Paper](https://arxiv.org/abs/2307.02427) |
| IRASim                | 细粒度动作条件视频 WM               | 逐帧 action conditioning，强调机械臂-物体接触和策略评测 | [Paper](https://arxiv.org/abs/2406.14540) |
| FlowDreamer           | RGB-D + scene-flow WM               | 显式预测 3D 运动流，再生成未来 RGB-D 帧 | [Paper](https://arxiv.org/abs/2505.10075) |
| WorldEval             | WM 策略评测环境                    | 用 latent action 生成策略 rollout，评估真实机器人策略和 checkpoint | [Project](https://worldeval.github.io/) · [Paper](https://arxiv.org/abs/2505.19017) |
| WorldGym              | WM policy evaluation               | 用动作条件视频模型做 Monte Carlo rollout，比较模型内外策略排名 | [Paper](https://arxiv.org/abs/2506.00613) |
| ViTacWorld            | 视觉-触觉 WM                       | 预测动作条件下的视觉和触觉未来，服务接触任务的数据扩增和评测 | [Project](https://vitacworld.github.io/) · [Paper](https://arxiv.org/abs/2607.22530) |

## 6. 仿真、环境与基准

| 项目         | 特点                                     | 推荐用途                           | 链接                                                                                                           |
| ------------ | ---------------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| MuJoCo       | 成熟、通用的接触动力学仿真               | 经典控制与轻量 RL 原型             | [GitHub](https://github.com/google-deepmind/mujoco) · [Docs](https://mujoco.readthedocs.io/) · [仿真教程](mujoco-tutorial.md) |
| Gymnasium    | 标准 RL 环境 API                         | 算法调用与小实验                   | [GitHub](https://github.com/Farama-Foundation/Gymnasium) · [Docs](https://gymnasium.farama.org/)                    |
| ManiSkill    | GPU 并行机器人操作、数据生成与评测       | 单机仿真操作、VLA/RL 评测          | [GitHub](https://github.com/mani-skill/ManiSkill) · [Project](https://maniskill.ai/)                                |
| Isaac Lab    | 基于 Isaac Sim 的 GPU 加速机器人学习框架 | 大规模 RL、sim-to-real、复杂传感器 | [GitHub](https://github.com/isaac-sim/IsaacLab) · [Docs](https://isaac-sim.github.io/IsaacLab/)                     |
| Isaac Sim    | 基于 USD/PhysX 的机器人仿真平台           | 高保真传感器、复杂场景与 GPU 仿真   | [GitHub](https://github.com/isaac-sim) · [Docs](https://docs.isaacsim.omniverse.nvidia.com/) · [仿真教程](isaac-sim-tutorial.md) |
| robosuite    | MuJoCo 机器人操作任务框架                | 操作控制器、数据生成与仿真对比     | [GitHub](https://github.com/ARISE-Initiative/robosuite) · [Docs](https://robosuite.ai/)                             |
| LIBERO       | 语言条件、知识迁移与 lifelong 操作       | VLA 标准化评测                     | [Project](https://libero-project.github.io/main.html) · [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| CALVIN       | 长时程语言条件操作                       | VLA 闭环与任务链评测               | [GitHub](https://github.com/mees/calvin) · [Paper](https://arxiv.org/abs/2112.03227)                                |
| Meta-World   | 多任务机械臂操作                         | 多任务/组合泛化与 RL success rate  | [Project](https://meta-world.github.io/) · [GitHub](https://github.com/Farama-Foundation/Metaworld)                 |
| RoboDojo     | 统一仿真与真实世界的通用机器人操作评测   | sim-to-real、泛化、记忆、精细操作和长时程评测 | [Website](https://robodojo-benchmark.com/) · [GitHub](https://github.com/robodojo-benchmark/RoboDojo) · [Docs](https://robodojo-benchmark.com/doc) |
| RoboTwin 2.0 | 数字孪生数据生成与双臂操作任务           | 操作泛化与大规模数据               | [GitHub](https://github.com/RoboTwin-Platform/RoboTwin)                                                           |
| RoboCasa     | 家庭厨房与长时程操作                     | WM/WAM、VLA 与 RL 的复杂接触任务   | [GitHub](https://github.com/robocasa/robocasa)                                                                    |
| DMControl    | 连续控制任务套件                         | MBRL 原型、动力学和规划对比        | [GitHub](https://github.com/google-deepmind/dm_control)                                                           |

## 7. 机器人数据（部分）

| 数据/生态         | 内容                                 | 链接                                                |
| ----------------- | ------------------------------------ | --------------------------------------------------- |
| Open X-Embodiment | 多机构、多本体机器人轨迹集合         | [Project](https://robotics-transformer-x.github.io/)   |
| DROID             | 大规模、场景多样的真实机器人操作数据 | [Project](https://droid-dataset.github.io/)            |
| BridgeData V2     | 通用机器人操作轨迹                   | [Project](https://rail-berkeley.github.io/bridgedata/) |

## 8. 按目标选技术栈

| 目标                  | 基础组合                                          | 原因                                                           |
| --------------------- | ------------------------------------------------- | -------------------------------------------------------------- |
| 第一次跑 VLA 闭环评测 | OpenVLA/OpenVLA-OFT + LIBERO                    | 有公开 checkpoint、任务定义与成功判定                          |
| 单机 GPU 做操作 RL    | ManiSkill + CleanRL/SB3                           | GPU 并行环境与算法基线组合直接                                 |
| 学 MBRL               | TD-MPC2 + DMControl/ManiSkill                     | 动力学模型、imagined rollout、价值和 MPC 路径清晰              |
| 学 WM 表征/视频/3D    | V-JEPA 2 + LPWM + Cosmos Predict2 + GWM + VGGT    | 对比全局 latent、对象中心 latent、视频生成和 3D Gaussian，再接动作条件验证 |
| 学开源 VLA            | OpenVLA/OpenVLA-OFT/OpenPI + LIBERO              | 代码可读，并有常见评测入口                                     |
| 做 VLA 的 RL 后训练   | StarVLA + RLinf + LIBERO/ManiSkill                | 已有直接集成示例                                               |
| 研究 WAM 推理效率     | FastWAM + LIBERO/RoboTwin                         | 对应论文的训练/测试设定                                        |
| 大规模 sim-to-real    | Isaac Lab + RL 框架                               | 高吞吐仿真与传感器/机器人生态                                  |
| 双臂真机部署          | bimanual-vla + 已完成 benchmark 的 VLA checkpoint | 沿用仓库的硬件连接、启动流程和安全检查                         |
