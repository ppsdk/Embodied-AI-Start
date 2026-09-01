# WM：从表示到交互闭环

World Model（WM，世界模型）要回答的不是“能不能生成一段像真的视频”，而是：**给定当前状态、机器人动作和任务条件，世界接下来会怎样变化**。最小形式为：

```text
p(x[t+1:t+H] | x[t], a[t:t+H-1], l)
```

其中 `x[t]` 是观测或状态，`a[t]` 是动作，`l` 是语言/目标，`H` 是预测步数。未来 `x` 可以是图像、latent、对象状态、运动场、3D/4D 场景、触觉或物理量。

## 1. 先把边界说清楚

一个面向机器人的 WM 至少满足三点：

1. **预测未来**：有时间上的状态转移，而不是只编码当前图像。
2. **以外部世界为对象**：预测物体、空间、接触、任务进展等，而不是只输出 reward 或语言解释。
3. **对干预敏感**：改变候选动作后，预测后果应发生相应变化。

因此，纯视频生成器、静态 3D 重建器、只输出 value 的评估器，不能自动称为可控机器人 WM。它们可以是 WM 的组件，是否构成完整 WM 要看有没有动作条件、未来状态和闭环证据。

WM 与 WAM 的关系也要分开：WM 是 **consequence prediction**；World Action Model（WAM）是把这种预测结构性接入动作生成后的 policy 范式。WAM 可以使用像素、latent、flow 或 3D 表示，WAM 不是第五种表示类型。

## 2. 四条表示主线

| 表示 | 预测对象 | 长处 | 常见风险 | 代表方向 |
| --- | --- | --- | --- | --- |
| 像素/视频 | RGB、RGB-D、video token 或 video latent | 直观、可视化强，容易利用视频预训练 | 看起来合理但动作不一致；接触和长程一致性差 | IRASim、DIAMOND、WorldGym、World4RL |
| 全局 latent / JEPA | 对控制有用的压缩状态 `z[t]` | 计算省，适合 MPC、搜索和策略辅助训练 | 可解释性弱，需验证 latent 是否保留物理变量 | Dreamer、V-JEPA 2、PSG-JEPA、Fast-WAM |
| 对象中心 / 粒子 | `M` 个对象 slot、粒子或部件状态 | 对象持久性、交互和组合泛化更清楚 | slot 身份交换、遮挡和关系漂移 | LPWM、SlotFormer、FOCUS |
| 3D/4D 几何 | 点云、occupancy、SDF、Gaussian、scene flow | 可查询自由空间、碰撞、可达性和多视角关系 | 深度/坐标代价高，动态更新和实时性困难 | GWM、PointWorld、TesserAct、OccWorld、3DFlowAction |

这四条线不是互斥架构。一个系统可以用视频 encoder 得到 object slots，再预测 3D flow；也可以用 latent dynamics 做规划、用视频 decoder 只做可视化检查。

### 2.1 其他重要表示

- **运动场/scene flow**：直接预测哪些区域向哪里移动，适合 pushing、rearrangement 和短程 servoing。
- **物理状态**：接触、力、摩擦、支撑、碰撞、形变和材料属性，适合插入、装配和接触丰富任务。
- **符号/层级状态**：物体是否已抓取、抽屉是否打开、子目标是否完成，适合长程任务和恢复。
- **记忆状态**：场景摘要、历史事件、可寻址 KV 或任务进度，解决返回旧场景和跨阶段执行。

## 3. 预测形式与系统用途

### 3.1 被动预测与动作条件预测

被动 WM 学习：

```text
p(x[t+1:t+H] | x[t])
```

它能学习视频动态，但不能据此保证规划可用。控制需要动作条件 WM：

```text
p(x[t+1:t+H] | x[t], a[t:t+H-1], l)
```

训练时应包含不同质量的动作，尤其是失败动作。只用 expert action 训练，模型可能学会“无论做什么都成功”。FACT 的做法是：失败轨迹不作为 imitation target，却保留其 action-consequence 作为失败后果监督。

### 3.2 WM 的六种用法

1. **策略表征学习**：把未来 latent、动作后果或 inverse dynamics 作为辅助目标。
2. **显式规划**：在 latent、视频或 3D 状态中 rollout 候选动作，用 MPC/MCTS/采样搜索选动作。
3. **learned simulator**：生成 imagined rollout，用于 policy 后训练或强化学习。
4. **evaluator / value / verifier**：预测进度、成功、风险或动作一致性，给候选轨迹打分。
5. **在线自适应**：用新机器人上的上下文估计动力学、校正动作或检索成功经验。
6. **合成数据引擎**：生成带动作、位姿、深度或触觉的训练片段，但必须验证可执行性。

同一模型可有多个角色，但报告时要写清楚：预测结果是否进入动作生成、只在训练使用还是部署时也使用，以及闭环控制是否真正调用它。

## 4. 数据字段：读论文时必须问清楚

一条可用于动作条件 WM 的样本至少应能写成：

```text
{obs[t-K+1:t], state[t-K+1:t], action[t:t+H-1],
 obs[t+1:t+H], reward[t:t+H-1], done, task, timestamps}
```

| 字段 | 需要说明 |
| --- | --- |
| 观测 | 单/多视角 RGB、深度、点云、触觉、力/力矩、关节状态和语言；历史窗口 `K`、分辨率、频率 |
| 动作 | 关节位置/速度/力矩、末端位姿、夹爪、action chunk 或 latent action；坐标系和控制周期 |
| 未来目标 | 下一帧、未来 `H` 帧、latent、slot、flow、occupancy、接触、reward、终止或风险 |
| 时间对齐 | 相机与控制器时间戳、执行延迟、帧堆叠、动作是指令值还是实际执行值 |
| 轨迹质量 | expert、成功、失败、恢复、碰撞、截断；失败原因和终止位置 |
| 任务条件 | 语言指令、目标图像、子目标、场景 ID、机器人本体 ID |
| 坐标与本体 | 相机/世界/末端坐标变换，URDF、关节限制、控制器和标定信息 |

关键区别是 **commanded action** 与 **executed action**：真实机器人有延迟、限幅和控制误差，最好记录实际状态变化，否则模型学到的可能只是命令而非后果。

## 5. 训练目标和闭环检查

常见目标包括：

- 像素重建或 diffusion/flow matching loss：保证未来外观。
- latent prediction / JEPA loss：保证未来表征可预测。
- inverse dynamics loss：从 `(x[t], x[t+1])` 估计动作，检查状态差异是否可控。
- object/flow/occupancy loss：保证几何和运动结构。
- reward/progress/termination loss：支持规划和失败识别。
- contact/force/physics consistency loss：抑制穿透、滑移和不可能接触。

最低限度的闭环检查是反事实干预：固定同一个 `x[t]`，输入 expert、轻微偏离和明显错误的 `a`，比较预测未来是否有方向正确的差异。只报告视频质量不能证明动作因果性。

## 6. 代表方向速查

| 方向 | 核心机制 | 代表工作 |
| --- | --- | --- |
| 真实机器人在线 WM | 在真实交互中更新 latent dynamics，并在模型内想象 | [DayDreamer](https://arxiv.org/abs/2206.14176) |
| 对象槽位动力学 | 先分解 slot，再预测对象属性和关系 | [SlotFormer](https://arxiv.org/abs/2210.05861)、[FOCUS](https://arxiv.org/abs/2307.02427) |
| 动作条件视频 | 在视频生成 block 中注入逐帧动作 | [IRASim](https://arxiv.org/abs/2406.14540)、[DIAMOND](https://arxiv.org/abs/2405.12399) |
| 语言条件 WM | 预测语言相关状态、子目标或未来视觉 | [Dynalang](https://arxiv.org/abs/2210.03822)、[H-WM](https://arxiv.org/abs/2602.11291) |
| 3D/4D WM | 预测点、occupancy、Gaussian、flow 或 4D 场景 | GWM、PointWorld、TesserAct、OccWorld |
| 触觉与物理 | 联合预测视觉、触觉、接触和力 | ViTacWorld、PIN-WM、PhysWorld |
| WM 做策略评测 | 用 WM rollout 比较策略或候选动作 | [WorldEval](https://arxiv.org/abs/2505.19017)、[WorldGym](https://arxiv.org/abs/2506.00613) |
| 失败感知 | 将坏动作作为 consequence/risk 监督 | FACT、WorldEcho |
| 结构化 latent | 用 JEPA、DINO 或粒子状态替代像素重建 | V-JEPA 2、PSG-JEPA、LPWM |

## 7. 近期工作：补齐“表示之外”的问题

### 7.1 Latent 是否真的可用于控制

V-JEPA 2、PSG-JEPA 和相关 JEPA-WM 说明，latent prediction 的关键不只是 loss 下降，还要验证：物理状态能否从 latent 中被识别；不同动作是否造成可分离的 latent 转移；规划得到的动作是否提升真实成功率。LPWM 的 Encoder 可以提供对象/粒子感知前端，但只有 Encoder 不能替代预测 dynamics、动作接口和 rollout。

### 7.2 Streaming、长上下文与记忆寻址

[MiniWorld](https://github.com/zhao-yian/MiniWorld) 的要点是 chunk-level causal + asynchronous diffusion：chunk 内可双向注意力，chunk 间保持因果依赖，并用 rolling KV cache 限制活跃计算。它说明 streaming 是执行形态，autoregressive 是依赖结构，不能混为同义词。

WorldTrace 将长期视频 WM 的记忆拆成 `recent window + summary slots`，并在 RoPE 下解决旧 token 不可寻址和平均 key 的 phase cancellation。其启发是：长期记忆至少要分别考虑 **选择什么、如何压缩、能否寻址**，而不是简单增加上下文长度。

### 7.3 失败后果与动作耦合

FACT 的训练划分很值得借鉴：成功轨迹可用于 imitation，失败动作仍用于学习“这样做会造成什么”。评估时应加入 action swap、off-expert action、失败类型覆盖和风险校准，而不仅是 expert rollout 的视频相似度。

### 7.4 WAM 的运行时问题

Fast-WAM、PILOT 和 WAM4D 关注训练期 world supervision 能否在推理时移除；BICPO-VLA 关注异步 action chunk 的 request-to-handoff gap：新 chunk 请求时旧 chunk 仍在执行，接管状态已经变化。评测应报告真实 control Hz、chunk stride、推理延迟、boundary jump 和成功率，而非只报 FLOPs。

### 7.5 外部编排与跨本体部署

Harness VLA/HarnessWAM 表明，局部 predictive policy 还需要场景 belief、task graph、进度监测、验证和 recovery。Qwen-RobotManip 等跨本体工作进一步强调 representation、motion、behavior alignment，以及目标机器人上的标定、IK/FK、控制器和安全适配。跨本体 claim 必须同时报告模型迁移能力和部署工程投入。

## 8. 如何评价一个 WM

建议把指标分成四层：

1. **表观层**：视频/latent 预测误差、多视角一致性、长期漂移。
2. **结构层**：对象跟踪、flow/occupancy、接触、碰撞、支撑和物理状态识别。
3. **因果层**：动作跟随、反事实区分、成功/失败后果覆盖、风险校准。
4. **控制层**：MPC 或 policy 的真实成功率、长程完成率、OOD 泛化、延迟、控制频率和安全事件。

World-model 内成功率不能替代真实机器人测试；生成质量提升也不能自动推出控制收益。对 learned simulator 和 evaluator，还应报告其与真实环境的策略排名相关性、候选数—延迟曲线和失效类型。

## 9. 一条实用的阅读与研究检查链

```text
状态是否保留任务相关、对象和物理变量？
        ↓
给定不同动作，预测后果是否发生可解释变化？
        ↓
成功和失败后果是否都覆盖并校准？
        ↓
future / transition / value 是否真正改变动作？
        ↓
在真实延迟、记忆和 chunk handoff 下是否稳定？
        ↓
能否发现失败、重规划并恢复？
        ↓
跨任务、布局、视角和 embodiment 的收益是否超过适配成本？
```

读一篇 WM 论文时，优先记录四件事：`输入字段`、`预测张量`、`动作如何进入模型`、`预测如何影响动作或评测`。如果只找到 encoder、视频样例或 value head，应准确称为结构化感知、生成模型或 evaluator，不要夸大为完整机器人 WM。

相关内容：[知识图谱](knowledge-map.md)、[模型基础](model-basics.md)、[论文索引](papers.md)、[代码库](codebases.md)。
