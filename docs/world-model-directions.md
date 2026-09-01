# WM ：从表示到交互闭环

WM 的表示可以先按四条主线理解：像素/视频直接预测未来画面，全局 latent/JEPA 预测可用于判断和规划的未来特征，对象中心 latent（如 LPWM、SlotFormer）预测粒子和交互，3D/4D（如 GWM、OccWorld）预测带空间坐标的 Gaussian、occupancy 或点云场景。除此之外，还要看时间模型和条件接口：离散自回归（IRIS）、像素 diffusion（DIAMOND）、语言条件（Dynalang）、显式 scene flow（FlowDreamer）、视觉-触觉预测（ViTacWorld）、层级逻辑-视觉预测（H-WM）、策略评测环境（WorldEval/WorldGym）以及真实机器人在线学习（DayDreamer）都属于 WM 的重要方向。具体的张量、数据字段、训练目标和闭环检查见[知识图谱](knowledge-map.md)和[模型基础](model-basics.md)。


## 一张表先定位

| 方向                | 核心机制                                                 | 代表论文                                                                              | 主要输入/输出                                                      | 闭环定位                                    |
| ------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| 真实机器人在线学习  | 直接在真实机器人上更新 latent dynamics，并在模型内想象   | [DayDreamer](https://arxiv.org/abs/2206.14176)                                           | 相机/状态/动作 -> latent state、reward、继续状态                   | 真实机器人 online MBRL 基线                 |
| 对象槽位动力学      | 先分解 slot，再预测对象属性和关系                        | [SlotFormer](https://arxiv.org/abs/2210.05861)、[FOCUS](https://arxiv.org/abs/2307.02427)   | 视频 ->`M` 个 object slot -> future slots                        | 可接规划；FOCUS 还把预测误差用于探索        |
| 细粒度动作-视频对齐 | 在视频生成 block 内注入逐帧动作条件                      | [IRASim](https://arxiv.org/abs/2406.14540)                                               | 历史 RGB + action trajectory -> future RGB video                   | 可做策略评测和测试时规划，需验证真实动力学  |
| 显式 3D 运动        | 先预测 scene flow，再生成未来 RGB-D                      | [FlowDreamer](https://arxiv.org/abs/2505.10075)                                          | RGB-D + action -> 3D flow + future RGB-D                           | 运动表征前置，仍需动作条件闭环              |
| 视觉-触觉 WM        | 同时预测视觉和触觉反馈                                   | [ViTacWorld](https://arxiv.org/abs/2607.22530)                                           | RGB、触觉、动作 -> future RGB 和 tactile                           | 适合接触丰富任务的数据扩增与评测            |
| 层级逻辑-视觉 WM    | 高层预测符号状态，低层预测视觉变化                       | [H-WM](https://arxiv.org/abs/2602.11291)                                                 | 视觉/语言/动作 -> logical subgoal + visual future                  | 用中间子目标降低长时程漂移                  |
| Robot-factored WM   | 控制器、运动学和 URDF 先渲染机器人几何，WM 学场景响应    | [Robot-Factored WM](https://arxiv.org/abs/2607.22535)                                    | action -> nominal robot trajectory/rendering -> scene future       | 解决 embodiment/action realization 混杂问题 |
| BEV/导航 WM         | 在鸟瞰图或地图 latent 中预测空间和视角演化               | [BEV Pretrained WM](https://arxiv.org/abs/2310.18847)                                    | 图像/位姿 -> BEV/map future、路线或视角                            | 更适合导航和主动感知，不直接等于操作 WM     |
| WM 策略评测环境     | 在 WM 中采样完整策略 rollout，比较模型内外排名           | [WorldEval](https://arxiv.org/abs/2505.19017)、[WorldGym](https://arxiv.org/abs/2506.00613) | 起始帧 + policy action -> imagined trajectory、reward              | 评测代理；模型内成功率不能替代真实测试      |
| 物理一致性诊断      | 检查模型是否真的按 action 改变未来，而不是只生成相似画面 | [WorldEcho](https://arxiv.org/abs/2608.24885)                                            | 同一初始状态 + expert/off-expert action -> future difference、risk | 属于验证和安全层，不是新的表示空间          |
| 概率与不确定性      | ensemble 或随机 dynamics 输出多条未来假设                | [PETS](https://arxiv.org/abs/1805.12114)                                                 | state/action -> distribution or `K` rollouts                     | 用于风险敏感规划；概率预测本身不是安全保证  |

## 这些方向怎么和已有四类表示交叉

- **表示轴**决定模型保留什么：RGB/视频、全局 latent、对象 slot、点云/Gaussian、occupancy、触觉或 BEV。
- **时间轴**决定怎么预测：自回归 token、diffusion、scene-flow、graph simulator、层级模型或 ensemble。
- **用途轴**决定是否进入决策：只做未来表征，还是做 imagined rollout、MPC、value/policy optimization，或者只做策略评测。
- 一篇论文可以同时落在多个格子。例如 FlowDreamer 是 RGB-D 表示 + scene-flow 时间模型；ViTacWorld 是视觉-触觉表示 + action-conditioned video；H-WM 是视觉表示 + 层级逻辑预测。

## 读论文时要记录的变量

至少记录以下字段，避免只看生成样例：

| 字段     | 需要写清楚的内容                                                                |
| -------- | ------------------------------------------------------------------------------- |
| 观测     | 单/多视角 RGB、深度、BEV、触觉、关节状态、语言；历史窗口长度和频率              |
| 动作     | 关节位置/速度/力矩、末端 `SE(3)`、action chunk、latent action，还是车辆控制量 |
| 预测目标 | 下一帧、未来视频、slot、scene flow、occupancy、reward、终止或风险               |
| 时间跨度 | 单步、未来 `H` 步、开放环 rollout 长度，以及是否 receding horizon             |
| 条件对齐 | action 与帧/触觉/位姿的时间戳、控制频率、延迟和坐标系                           |
| 决策接口 | 无 planner、MPC、搜索、value/policy optimization，还是仅 policy evaluation      |
| 证据     | 视频/几何指标、动作跟随、策略排名相关性、真实机器人成功率和安全测试             |

## 推荐阅读顺序

1. **先看 DayDreamer**：理解真实机器人 online WM 的数据闭环和代价。
2. **再看 SlotFormer + FOCUS**：理解对象级状态为何有助于交互建模和探索。
3. **接着看 IRASim + FlowDreamer**：比较动作-帧对齐和显式 3D motion 两种视频 WM 设计。
4. **然后看 ViTacWorld + Robot-Factored WM**：分别补上触觉和 embodiment factorization。
5. **最后看 H-WM、WorldEval/WorldGym、WorldEcho**：从层级规划、策略评测和物理一致性把 WM 接到部署流程。

## 边界提醒

- **场景重建/渲染前端**（例如静态 3DGS、VGGT）提供表示，不自动拥有时间动力学或 action interface。
- **latent action** 是模型内部变量，除非经过目标机器人数据校准，否则不能当作关节或末端控制量。
- **策略评测 WM** 可以筛选 checkpoint、估计相对排名或做风险预筛，但不能替代真实机器人安全测试。
- **MBRL** 只在模型真正服务 imagined rollout、MPC、value 或 policy optimization 时成立；只生成未来视频仍属于 WM。
