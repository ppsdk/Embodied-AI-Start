# Benchmark 指南：VLA、WM、MBRL 与 WAM

Benchmark 不是一张总排行榜，而是一套固定的任务、数据和成功判定。选 benchmark 时先看它能不能回答你的问题，再看分数。WM 常看预测和几何质量，MBRL 常看回报、样本效率和规划成本，这些指标不能硬合成一个总分。

## 1. 按研究问题选择

| 研究问题               | 首选                                                         | 可补充                                                                         | 主要指标                                                     |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| 语言条件操作与组合泛化 | [LIBERO](https://libero-project.github.io/main.html)            | [CALVIN](https://github.com/mees/calvin)                                          | task success、长时程完成率、seen/unseen 泛化                 |
| 多任务 VLA 适配        | [LIBERO-100](https://github.com/Lifelong-Robot-Learning/LIBERO) | [Meta-World](https://github.com/Farama-Foundation/Metaworld)                      | 每任务 success、平均 success、任务间方差                     |
| GPU 并行操作 RL        | [ManiSkill](https://github.com/haosulab/ManiSkill)              | [Isaac Lab](https://github.com/isaac-sim/IsaacLab)                                | 环境步数、样本效率、success、吞吐                            |
| 仿真器与实验路线选择   | [MuJoCo 仿真教程](mujoco-tutorial.md)                          | [Isaac Sim 仿真教程](isaac-sim-tutorial.md)                                     | 版本、GPU、physics/policy dt、并行环境数、资产配置           |
| 通用机器人操作综合评测 | [RoboDojo](https://robodojo-benchmark.com/)                     | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)、[RoboCasa](https://github.com/robocasa/robocasa) | sim/real 一致评测、泛化、记忆、精细操作、长时程与开放词汇指令 |
| 双臂与数字孪生泛化     | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)   | [RoboCasa](https://github.com/robocasa/robocasa)                                  | task success、跨场景/物体泛化、恢复率                        |
| WM：JEPA / video / 3D  | [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)   | [RoboCasa](https://github.com/robocasa/robocasa)、[ManiSkill](https://maniskill.ai/) | latent/video/geometry prediction、动作条件一致性、长时程漂移 |
| 4D occupancy WM        | [nuScenes](https://www.nuscenes.org/) / Occ3D 标注       | [ScanNet](https://github.com/ScanNet/ScanNet) 室内场景、OccWorld-ScanNet | occupancy IoU/mIoU、未来帧一致性、ego/planning error；先固定坐标系、体素范围和预测时长 |
| MBRL                   | [DMControl](https://github.com/google-deepmind/dm_control)      | [ManiSkill](https://maniskill.ai/)                                                | return、环境步数、模型 rollout 误差、规划/推理成本           |
| Offline RL 数据协议    | [Minari](https://minari.farama.org/)                            | [D4RL](https://github.com/Farama-Foundation/D4RL)                                 | normalized return、数据集质量、OOD 动作风险                  |

### LIBERO 介绍

原始 LIBERO 论文的核心 suite 是 Spatial、Object、Goal 和 LIBERO-100；LIBERO-100 常见地拆成用于策略预训练的 LIBERO-90 与 lifelong-learning 评测的 LIBERO-10。工具或论文中的 `LIBERO-Long`、`libero_10` 等别名应以具体实现的任务列表为准。**LIBERO-Plus 是额外的鲁棒性扩展，不是原始 suite 的第五类。**

## 2. 常见 benchmark 入口

| Benchmark                                                                          | 覆盖范围                            | 选择时记录                                                    |
| ---------------------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------- |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)                           | MuJoCo 语言条件桌面操作与知识迁移   | suite、任务数、初始状态、相机、success 判定                   |
| [CALVIN](https://github.com/mees/calvin)                                              | 语言条件长时程连续操作              | ABC/XYZ split、chain 长度、每步与整链 success                 |
| [ManiSkill](https://github.com/haosulab/ManiSkill)                                    | GPU 并行操作、数据生成、sim-to-real | 版本、任务 ID、并行环境数、控制频率                           |
| [Meta-World](https://github.com/Farama-Foundation/Metaworld)                          | 经典多任务机械臂操作                | MT10/MT50/单任务、train/test task split                       |
| [RoboDojo](https://robodojo-benchmark.com/)                                           | 统一的仿真与真实世界通用机器人操作评测（42 个仿真任务、18 个真实任务） | sim/real split、任务子集、Isaac Sim 版本、XPolicyLab 接口、真实硬件、scene reset、评测次数 |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin)                         | 双臂数字孪生任务与数据              | 场景、物体组合、双臂 action interface                         |
| [RoboCasa](https://github.com/robocasa/robocasa)                                      | 家庭厨房与长时程操作                | task suite、场景随机化、接触与终止条件                        |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab)                                    | GPU 仿真与大规模 RL                 | Isaac Sim 版本、GPU、并行度、sim-to-real 设置                 |
| [DMControl](https://github.com/google-deepmind/dm_control)                            | 连续控制与 MBRL 原型                | domain/task、episode length、frame skip、模型 rollout horizon |
| [Minari](https://minari.farama.org/) / [D4RL](https://github.com/Farama-Foundation/D4RL) | 离线轨迹与 offline RL               | dataset 版本、return normalization、数据 split                |

### WM 与 MBRL 的评测边界

- **WM**：根据路线报告 latent prediction、视频质量/时序一致性、3D 几何/新视角质量、动作条件敏感性，以及是否改善下游控制；目前没有一个覆盖 JEPA、视频和 3D/4D 的统一总榜。
- **4D occupancy WM**：除了 occupancy IoU/mIoU，还要报告未来预测时长、体素分辨率、是否给定未来 ego pose/trajectory，以及规划误差。不同坐标系、体素范围和占据类别定义下的分数不能直接横比。
- **MBRL**：固定环境、模型容量、真实环境步数和规划预算，报告 return、success、sample efficiency、rollout horizon、模型误差和规划延迟。
- 将 WM 解释为控制模型时，同时报告动作条件和决策收益证据，并分别列出视频/3D 质量与闭环成功率。

## 3. 统一记录模板

跨方法比较时，至少记录下面字段；缺失字段应写成 `未报告`，不要用默认值补齐。

```text
benchmark / version:
suite / task subset:
observation: views, resolution, history, proprioception
action: joint / end-effector / delta, chunk horizon, control rate
initialization: checkpoint, seed, initial-state protocol
data: demonstrations / videos / online rollouts, amount and source
metric: success / return / chain success, denominator and confidence interval
compute: GPU, training hours, environment steps, inference latency
model path: VLA / WM / MBRL / WAM / RL, explicit future rollout or not
wm target: latent / video / 3D-4D / state, action-conditioned or not
mbrl target: dynamics / reward / value, planner or actor-critic, rollout horizon
failure: contact, language grounding, long-horizon drift, recovery, safety
```

## 4. 读 leaderboard 的边界

[Embodied Meta-LLM Leaderboard](https://ppsdk.github.io/embodied-meta-leaderboard/) 适合做论文和模型的来源索引，尤其是 VLA/WAM 在统一页面上的协议追踪。使用它时保留原始任务子集、metric、数据类型、评测次数和更新时间；不同 benchmark 的 success rate 或 normalized return **不能直接求平均或排序**。如需汇总，应先按同一 benchmark、同一 protocol 分组，再报告均值、方差与覆盖范围。

## 5. 本仓库暂不覆盖

- locomotion、足式/腿式运动控制 benchmark；
- VLN、纯导航与地图构建 benchmark；
- 只报告视频生成质量、没有动作闭环或任务成功判定的榜单。

这些方向与具身智能相关，但不属于当前 VLA/WAM/WM/RL 操作主线。
