# Benchmark 指南：VLA、WM、MBRL 与 WAM

> 📊 根据研究问题选择任务、数据和评测协议，避免只比较一段视频。

**适合读者**：需要选 benchmark、设计评测或复现实验的读者  
**预计阅读**：15 min  
**前置知识**：至少了解 VLA、WM、RL/MBRL 的基本任务形式  
**下一步**：[论文清单](papers.md) · [代码仓](codebases.md) · [强化学习基础](reinforcement-learning.md)

**本文路线**：研究问题 → benchmark 入口 → 协议与指标 → 本仓库实践边界

Benchmark 不是一张总排行榜，而是一套固定的任务、数据和成功判定。选 benchmark 时先确认它能否回答问题，再看分数。WM 常看预测和几何质量，MBRL 常看回报、样本效率和规划成本，这些指标不能硬合成一个总分。

## 1. 按研究问题选择

| 研究问题               | 首选                                                         | 可补充                                                                                                    | 主要指标                                                                               |
| ---------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 语言条件操作与组合泛化 | [LIBERO](https://libero-project.github.io/main.html)            | [CALVIN](https://github.com/mees/calvin)                                                                     | task success、长时程完成率、seen/unseen 泛化                                           |
| 多任务 VLA 适配        | [LIBERO-100](https://github.com/Lifelong-Robot-Learning/LIBERO) | [Meta-World](https://github.com/Farama-Foundation/Metaworld)                                                 | 每任务 success、平均 success、任务间方差                                               |
| GPU 并行操作 RL        | [ManiSkill](https://github.com/haosulab/ManiSkill)              | [Isaac Lab](https://github.com/isaac-sim/IsaacLab)                                                           | 环境步数、样本效率、success、吞吐                                                      |
| 仿真器与实验路线选择   | [MuJoCo 仿真教程](mujoco-tutorial.md)                           | [Isaac Sim 仿真教程](isaac-sim-tutorial.md)                                                                  | 版本、GPU、physics/policy dt、并行环境数、资产配置                                     |
| 通用机器人操作综合评测 | [RoboDojo](https://robodojo-benchmark.com/)                     | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)、[RoboCasa](https://github.com/robocasa/robocasa) | sim/real 一致评测、泛化、记忆、精细操作、长时程与开放词汇指令                          |
| 双臂与数字孪生泛化     | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)   | [RoboCasa](https://github.com/robocasa/robocasa)                                                             | task success、跨场景/物体泛化、恢复率                                                  |
| WM：JEPA / video / 3D  | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)   | [RoboCasa](https://github.com/robocasa/robocasa)、[ManiSkill](https://maniskill.ai/)                            | latent/video/geometry prediction、动作条件一致性、长时程漂移                           |
| 4D occupancy WM        | [nuScenes](https://www.nuscenes.org/) / Occ3D 标注              | [ScanNet](https://github.com/ScanNet/ScanNet) 室内场景、OccWorld-ScanNet                                     | occupancy IoU/mIoU、未来帧一致性、ego/planning error；先固定坐标系、体素范围和预测时长 |
| MBRL                   | [DMControl](https://github.com/google-deepmind/dm_control)      | [ManiSkill](https://maniskill.ai/)                                                                           | return、环境步数、模型 rollout 误差、规划/推理成本                                     |
| Offline RL 数据协议    | [Minari](https://minari.farama.org/)                            | [D4RL](https://github.com/Farama-Foundation/D4RL)                                                            | normalized return、数据集质量、OOD 动作风险                                            |

### LIBERO 介绍

原始 LIBERO 论文的核心 suite 是 Spatial、Object、Goal 和 LIBERO-100；LIBERO-100 常见地拆成用于策略预训练的 LIBERO-90 与 lifelong-learning 评测的 LIBERO-10。工具或论文中的 `LIBERO-Long`、`libero_10` 等别名应以具体实现的任务列表为准。**LIBERO-Plus 是额外的鲁棒性扩展，不是原始 suite 的第五类。**

## 2. 常见 benchmark 入口

| Benchmark                                                                          | 覆盖范围                                                               | 选择时记录                                                                                 |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                           | MuJoCo 语言条件桌面操作与知识迁移                                      | suite、任务数、初始状态、相机、success 判定                                                |
| [CALVIN](https://github.com/mees/calvin)                                              | 语言条件长时程连续操作                                                 | ABC/XYZ split、chain 长度、每步与整链 success                                              |
| [ManiSkill](https://github.com/haosulab/ManiSkill)                                    | GPU 并行操作、数据生成、sim-to-real                                    | 版本、任务 ID、并行环境数、控制频率                                                        |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld)                          | 经典多任务机械臂操作                                                   | MT10/MT50/单任务、train/test task split                                                    |
| [RoboDojo](https://robodojo-benchmark.com/)                                           | 统一的仿真与真实世界通用机器人操作评测（42 个仿真任务、18 个真实任务） | sim/real split、任务子集、Isaac Sim 版本、XPolicyLab 接口、真实硬件、scene reset、评测次数 |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)                         | 双臂数字孪生任务与数据                                                 | 场景、物体组合、双臂 action interface                                                      |
| [RoboCasa](https://github.com/robocasa/robocasa)                                      | 家庭厨房与长时程操作                                                   | task suite、场景随机化、接触与终止条件                                                     |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab)                                    | GPU 仿真与大规模 RL                                                    | Isaac Sim 版本、GPU、并行度、sim-to-real 设置                                              |
| [DMControl](https://github.com/google-deepmind/dm_control)                            | 连续控制与 MBRL 原型                                                   | domain/task、episode length、frame skip、模型 rollout horizon                              |
| [Minari](https://minari.farama.org/) / [D4RL](https://github.com/Farama-Foundation/D4RL) | 离线轨迹与 offline RL                                                  | dataset 版本、return normalization、数据 split                                             |

## 3. 本仓库暂不覆盖

- locomotion、足式/腿式运动控制 benchmark；
- VLN、纯导航与地图构建 benchmark；
- 只报告视频生成质量、没有动作闭环或任务成功判定的榜单。

这些方向与具身智能相关，但不属于当前 VLA/WAM/WM/RL 操作主线。
