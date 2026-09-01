# 论文清单：VLA、WM、MBRL 与 WAM

本页的“近两年新增工作”按 2026-08-31 的公开 arXiv 和项目页面整理。预印本会继续变更，引用或复现前请打开原页面确认版本、作者、代码和实验协议。

## 阅读标记

- **S0｜必读**：建立共同语言，建议精读摘要、方法图、损失函数和实验结论。
- **S1｜主线**：按研究方向选择精读。
- **S2｜扩展**：遇到具体问题时再读。

## S0｜模型基础

这些论文先于具体 VLA/WM 论文阅读，用来建立“backbone、训练目标、采样器”三者的边界。配套的张量流见 [模型基础](model-basics.md)。

| 原语 | 论文 | 为什么读 | 代码入口 |
| --- | --- | --- | --- |
| Transformer | Attention Is All You Need | self-attention、位置编码、causal mask 与序列建模骨架 | [Transformers](https://github.com/huggingface/transformers) |
| Diffusion | Denoising Diffusion Probabilistic Models | 前向加噪、反向去噪、噪声预测目标与采样 | [Diffusers](https://github.com/huggingface/diffusers) |
| Flow Matching | Flow Matching for Generative Modeling | probability path、条件速度场和 ODE 生成 | [Flow Matching](https://github.com/facebookresearch/flow_matching) |
| DiT | Scalable Diffusion Models with Transformers | Transformer 作为扩散/视频生成 backbone 的实现范式 | [DiT](https://github.com/facebookresearch/DiT) |

## S0｜最短主线

| 顺序 | 论文 | 为什么读 | 入口 |
| --- | --- | --- | --- |
| 1 | Reinforcement Learning: An Introduction, 2nd ed. | MDP、价值函数、TD、策略梯度的统一底座 | [Book PDF](https://incompleteideas.net/book/RLbook2020.pdf) |
| 2 | Offline Reinforcement Learning: Tutorial, Review, and Perspectives | 理解固定数据集、分布偏移与保守学习 | [arXiv](https://arxiv.org/abs/2005.01643) |
| 3 | World Models | 理解 latent dynamics + controller 的经典 MBRL 前身 | [arXiv](https://arxiv.org/abs/1803.10122) · [Project](https://worldmodels.github.io/) |
| 4 | Diffusion Policy | 机器人动作生成、action chunk 与多峰行为的代表作 | [arXiv](https://arxiv.org/abs/2303.04137) · [Code](https://github.com/real-stanford/diffusion_policy) |
| 5 | π0: A Vision-Language-Action Flow Model for General Robot Control | 理解 VLM backbone、flow matching action expert 与通用策略 | [arXiv](https://arxiv.org/abs/2410.24164) · [Project](https://www.pi.website/blog/pi0) |
| 6 | π0.5: A Vision-Language-Action Model with Open-World Generalization | 异构协同训练、高层语义与开放世界长时程泛化 | [arXiv](https://arxiv.org/abs/2504.16054) · [Project](https://www.pi.website/blog/pi05) · [Code](https://github.com/Physical-Intelligence/openpi) |
| 7 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | 现代 JEPA 世界表征、物理理解与预测接口 | [arXiv](https://arxiv.org/abs/2506.09985) · [Code](https://github.com/facebookresearch/vjepa2) |
| 8 | World Action Models: The Next Frontier in Embodied AI | 用级联式/联合式框架建立 WAM 的系统分类 | [arXiv](https://arxiv.org/abs/2605.12090) · [Resources](https://github.com/OpenMOSS/Awesome-WAM) |
| 9 | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | 区分训练期视频建模收益和测试期显式想象成本 | [arXiv](https://arxiv.org/abs/2603.16666) · [Project](https://yuantianyuan01.github.io/FastWAM/) · [Code](https://github.com/yuantianyuan01/FastWAM) |
| 10 | StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing | 把 VLA 研究拆为可替换 backbone、action head、训练与部署模块 | [arXiv](https://arxiv.org/abs/2604.05014) · [Code](https://github.com/starVLA/starVLA) |
| 11 | RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation | 理解 VLA/基础模型 RL 后训练的系统问题 | [arXiv](https://arxiv.org/abs/2509.15965) · [Code](https://github.com/RLinf/RLinf) |

## S1｜VLA 与动作策略

| 论文 | 关键点 | 入口 |
| --- | --- | --- |
| OpenVLA: An Open-Source Vision-Language-Action Model | 开源 VLA 训练、微调与评测范式 | [arXiv](https://arxiv.org/abs/2406.09246) · [Code](https://github.com/openvla/openvla) |
| OpenVLA-OFT: An Open-Source Fine-Tuning Recipe for OpenVLA | OpenVLA 的高效微调与推理优化 | [Code](https://github.com/moojink/openvla-oft) |
| π0.5: A Vision-Language-Action Model with Open-World Generalization | 异构机器人数据协同、语义子任务与开放世界泛化 | [arXiv](https://arxiv.org/abs/2504.16054) · [Code](https://github.com/Physical-Intelligence/openpi) |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 跨本体数据混合与通用策略 | [arXiv](https://arxiv.org/abs/2310.08864) · [Project](https://robotics-transformer-x.github.io/) |
| Octo: An Open-Source Generalist Robot Policy | 开源通用策略、数据混合与轻量适配 | [arXiv](https://arxiv.org/abs/2405.12213) · [Code](https://github.com/octo-models/octo) |
| Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT) | 动作分块与 CVAE 在双臂长序列控制中的作用 | [arXiv](https://arxiv.org/abs/2304.13705) · [Code](https://github.com/tonyzhaozh/act) |
| Diffusion Policy | 以条件去噪建模多峰连续动作分布 | [arXiv](https://arxiv.org/abs/2303.04137) · [Code](https://github.com/real-stanford/diffusion_policy) |
| A Survey on Vision-Language-Action Models for Embodied AI | 从组件、训练目标、任务和架构补齐全景 | [arXiv](https://arxiv.org/abs/2405.14093) |

### 近两年新增工作

下面这些工作更接近当前工程问题：模型太大、推理太慢、历史信息没用好，或者还不会提前考虑接触和后续动作。2026 年条目目前主要是 arXiv 预印本，阅读时要自行核对版本、代码和实验协议。

| 论文 | 它在解决什么问题 | 入口 |
| --- | --- | --- |
| GR00T N1: An Open Foundation Model for Generalist Humanoid Robots（2025） | 用视觉语言模块理解任务，再用 diffusion transformer 生成连续动作；同时混合真实轨迹、人类视频和合成数据 | [arXiv](https://arxiv.org/abs/2503.14734) · [NVIDIA 项目页](https://developer.nvidia.com/isaac/gr00t) |
| Gemini Robotics: Bringing AI into the Physical World（2025） | 将通用多模态模型扩展为直接控制机器人的 VLA，并讨论跨本体适配和具身推理 | [arXiv](https://arxiv.org/abs/2503.20020) |
| SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics（2025） | 用更小的模型和异步推理降低训练、显存和部署成本，适合先在普通 GPU 上跑通 | [arXiv](https://arxiv.org/abs/2506.01844) · [LeRobot](https://github.com/huggingface/lerobot) |
| MolmoAct: Action Reasoning Models that can Reason in Space（2025） | 在低层动作前加入可编辑的空间轨迹/中层计划，让策略更容易解释和调整 | [arXiv](https://arxiv.org/abs/2508.07917) |
| TemporalFlow-VLA: Learning Physically Grounded Execution History for Long-Horizon Robot Manipulation（2026，预印本） | 不只堆历史图像，而是用机器人状态和几何信息训练“执行历史”表征，再服务长时程动作预测 | [arXiv](https://arxiv.org/abs/2608.26821) |
| FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference（2026，预印本） | 把不同噪声阶段的动作 chunk 放进流式缓冲区，边执行边解码，目标是降低 flow-matching VLA 的等待时间 | [arXiv](https://arxiv.org/abs/2608.27384) |
| PHR-VLA: Planning Horizon Reasoning for Vision-Language-Action Models（2026，预印本） | 训练时用未来局部动力学表征监督 VLA，让模型在接触和细粒度操作前多看一步 | [arXiv](https://arxiv.org/abs/2608.27609) |

## S1｜World Model：像素、latent 与 3D/4D

| 论文 | 关键点 | 入口 |
| --- | --- | --- |
| V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | 预测 latent representation 而不是逐像素重建，并连接物理理解与规划 | [arXiv](https://arxiv.org/abs/2506.09985) · [Code](https://github.com/facebookresearch/vjepa2) |
| Genie: Generative Interactive Environments | 从视频学习可交互的潜在环境与动作条件未来 | [arXiv](https://arxiv.org/abs/2402.15391) · [Project](https://sites.google.com/view/genie-2024/home) |
| Latent Particle World Models: Self-supervised Object-centric Stochastic Dynamics Modeling（2026，预印本） | 从视频自监督发现对象粒子，以粒子级 latent action 建模随机动态，并支持语言/目标图像条件 | [arXiv](https://arxiv.org/abs/2603.04553) · [Project](https://taldatech.github.io/lpwm-web/) · [Code](https://github.com/taldatech/lpwm) |
| GWM: Towards Scalable Gaussian World Models for Robotic Manipulation（2025） | 用 latent DiT + 3D VAE 预测动作作用下 Gaussian primitives 的传播，连接 3D 视频预测、模仿学习和 MBRL | [arXiv](https://arxiv.org/abs/2508.17600) · [Project](https://gaussian-world-model.github.io/) · [Code](https://github.com/Gaussian-World-Model/gaussianwm) |
| VGGT: Visual Geometry Grounded Transformer | 多视图几何、相机和 3D 场景表征，可作为 3D WM 的结构化前端 | [arXiv](https://arxiv.org/abs/2503.11651) · [Code](https://github.com/facebookresearch/vggt) |
| OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving（2023） | 用离散 occupancy token 和时空 Transformer 预测未来场景与 ego trajectory | [arXiv](https://arxiv.org/abs/2311.16038) · [Code](https://github.com/wzzheng/OccWorld) |
| DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model（2024） | 用时空 diffusion 生成可控的未来 occupancy，强调长时程和轨迹条件 | [arXiv](https://arxiv.org/abs/2410.10429) · [Project](https://gusongen.github.io/DOME) |
| PreWorld: Semi-Supervised Vision-Centric 3D Occupancy World Model（2025） | 用 2D 监督辅助 3D occupancy、4D forecasting 和 motion planning | [arXiv](https://arxiv.org/abs/2502.07309) |
| Delta-Triplane Transformers as Occupancy World Models（2025） | 预测紧凑 triplane 的增量，降低 4D occupancy 的生成成本 | [arXiv](https://arxiv.org/abs/2503.07338) |
| RoboOccWorld: Occupancy World Model for Robots（2025） | 面向室内机器人场景的 occupancy evolution 与相机位姿条件预测 | [arXiv](https://arxiv.org/abs/2505.05512) |
| FR3D: Future Dynamic 3D Reconstruction（2026，预印本） | 把相机自运动与环境变化解耦，预测持久 3D latent 的未来动态重建 | [arXiv](https://arxiv.org/abs/2606.18250) · [Project](https://fr3d-wm.github.io) |
| PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation（2026，预印本） | 学习无散 Gaussian velocity field，并把预测的 3D 动态接入操作策略 | [arXiv](https://arxiv.org/abs/2607.01938) · [Code](https://github.com/vLAR-group/PhysMani) |
| 3D-Belief: Embodied Belief Inference via Generative 3D World Modeling（2026，预印本） | 在 3D 中维护多假设 belief，更新未观测区域并评测导航/操作 | [arXiv](https://arxiv.org/abs/2605.11367) |
| IRIS: Transformers are Sample-Efficient World Models（2022） | 离散 VAE + 自回归 Transformer，在 Atari100k 中用少量交互学习可 rollout 的环境模型 | [arXiv](https://arxiv.org/abs/2209.00588) · [Code](https://github.com/eloialonso/iris) |
| DIAMOND: Diffusion for World Modeling（2024） | 用像素 diffusion 建模环境未来，研究视觉细节对模型内 RL 的影响 | [arXiv](https://arxiv.org/abs/2405.12399) · [Code](https://github.com/eloialonso/diamond) · [Project](https://diamond-wm.github.io) |
| Dynalang: Learning to Model the World with Language（2023） | 让描述环境规律的语言参与未来视觉/文本表征预测，并用 imagined rollout 学策略 | [arXiv](https://arxiv.org/abs/2308.01399) · [Code](https://github.com/jlin816/dynalang) |
| Learning to Simulate Complex Physics with Graph Networks（GNS，2020） | 用粒子图和消息传递学习流体、刚体和可变形物体动力学 | [arXiv](https://arxiv.org/abs/2002.09405) · [Code](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate) |
| GAIA-1: A Generative World Model for Autonomous Driving（2023） | 用视频、文本和动作 token 生成可控驾驶场景 | [arXiv](https://arxiv.org/abs/2309.17080) |
| DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving（2023） | 先学习交通结构约束，再用 diffusion 预测真实驾驶未来 | [arXiv](https://arxiv.org/abs/2309.09777) · [Code](https://github.com/JeffWang987/DriveDreamer) |
| Dream to Manipulate: Compositional World Models Empowering Robot Imitation Learning with Imagination（DreMa，2024） | 用 Gaussian Splatting + 物理模拟器构造可组合数字孪生，生成反事实机器人数据 | [arXiv](https://arxiv.org/abs/2412.14957) · [Project](https://dreamtomanipulate.github.io/) |
| Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics（2025） | 用双自回归机制学习部分可观测随机动力学，并在模型中做策略优化 | [arXiv](https://arxiv.org/abs/2501.10100) |
| DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos（2026，预印本） | 从大规模人类视频学习连续 latent action，再用少量机器人数据校准动作可控性 | [arXiv](https://arxiv.org/abs/2602.06949) · [Project](https://dreamdojo-world.github.io/) |
| PlayWorld: Learning Robot World Models from Autonomous Play（2026，预印本） | 用机器人自主探索收集长尾接触和失败数据，训练更可靠的操作视频 WM | [arXiv](https://arxiv.org/abs/2603.09030) · [Project](https://robot-playworld.github.io/) |
| World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy（2026，预印本） | 联合预测未来帧和 reward，并让更新后的策略轨迹回流继续校准 WM | [arXiv](https://arxiv.org/abs/2602.06508) |
| JEDI: Joint Embedding Diffusion World Model for Online MBRL（2026，预印本） | 在联合 embedding 中端到端学习 diffusion dynamics，兼顾 JEPA 表征和多模态 rollout | [arXiv](https://arxiv.org/abs/2605.13013) |
| Causal-JEPA: Learning World Models through Object-Level Latent Masking（2026，预印本） | 在对象 latent 上做遮挡预测，逼迫模型利用交互关系而非背景捷径 | [arXiv](https://arxiv.org/abs/2602.11389) · [Code](https://github.com/galilai-group/cjepa) |
| WorldEcho / WorldSync: Do Robotic World Models Really Follow Actions?（2026，预印本） | 用 off-expert action、视觉完整性和 $SE(3)$ 轨迹对齐检查动作跟随，并改进动作干预一致性 | [arXiv](https://arxiv.org/abs/2608.24885) |
| ReWorld: An Interactive World Model with Long-Horizon Memory（2026，预印本） | 用混合 attention 和位姿索引记忆库兼顾短期交互与长时程回访 | [arXiv](https://arxiv.org/abs/2608.23565) |
| Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models（PETS，2018） | 用概率 dynamics ensemble 和 trajectory sampling 将不确定性纳入模型预测控制 | [arXiv](https://arxiv.org/abs/1805.12114) |
| DayDreamer: World Models for Physical Robot Learning（2022） | 把 Dreamer 直接放到真实四足、机械臂和移动机器人上在线学习，说明 WM 不一定只在仿真器里训练 | [arXiv](https://arxiv.org/abs/2206.14176) · [Project](https://danijar.com/project/daydreamer/) |
| SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models（2022） | 在无对象标签的 slot 表征上建模对象关系和未来状态，可用于 VQA、目标条件规划和模型内决策 | [arXiv](https://arxiv.org/abs/2210.05861) · [Project](https://slotformer.github.io/) |
| FOCUS: Object-Centric World Models for Robotics Manipulation（2023） | 用对象中心 WM 产生探索奖励，主动寻找机器人-物体交互，而不是只拟合专家轨迹 | [arXiv](https://arxiv.org/abs/2307.02427) |
| IRASim: A Fine-Grained World Model for Robot Manipulation（2024） | 在视频扩散 Transformer 的每个 block 注入逐帧动作条件，专门改善机械臂与物体接触的时序对齐 | [arXiv](https://arxiv.org/abs/2406.14540) |
| FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation（2025） | 先预测显式 3D scene flow，再生成未来 RGB-D；把运动几何从渲染过程里单独拿出来建模 | [arXiv](https://arxiv.org/abs/2505.10075) |
| WorldEval: World Model as Real-World Robot Policies Evaluator（2025） | 用 latent action 驱动视频 WM，在线比较真实机器人策略和 checkpoint；重点是策略评测，不是替代真实执行 | [arXiv](https://arxiv.org/abs/2505.19017) · [Project](https://worldeval.github.io/) |
| WorldGym: World Model as An Environment for Policy Evaluation（2025） | 用动作条件视频 WM 做 Monte Carlo policy rollout，并检验模型内排名与真实成功率的相关性 | [arXiv](https://arxiv.org/abs/2506.00613) |
| H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model（2026，预印本） | 将高层逻辑状态预测和低层视觉状态预测分开，再用层级中间目标减少长时程误差累积 | [arXiv](https://arxiv.org/abs/2602.11291) |
| ViTacWorld: Scaling Visuo-Tactile World Models for Contact-Rich Robot Manipulation（2026，预印本） | 同时预测视觉和触觉未来，服务接触丰富任务的数据扩增与策略评测；提醒读者相机看不到的接触信息也要建模 | [arXiv](https://arxiv.org/abs/2607.22530) · [Project](https://vitacworld.github.io/) |
| Robot-Factored World Models via Robot Rendering（2026，预印本） | 将 action 先通过控制器和 URDF 渲染成机器人几何，再让 WM 学习物体响应，减少动作执行过程和未来状态泄漏 | [arXiv](https://arxiv.org/abs/2607.22535) |
| Bird's Eye View Based Pretrained World model for Visual Navigation（2023） | 在 BEV 表征上预训练导航世界模型，说明移动机器人 WM 可以预测地图/视角演化，而不只生成第一视角视频 | [arXiv](https://arxiv.org/abs/2310.18847) |
| A Comprehensive Survey on World Models for Embodied AI | 从表征、时间建模、空间建模和决策用途区分 WM 路线 | [arXiv](https://arxiv.org/abs/2510.16732) |

这里的 WM 是广义环境表征/预测/生成范式；JEPA、视频、物理模拟和 3D/4D 路线不要求自带 planner 或 actor-critic。Occupancy、驾驶、游戏和动态重建论文常来自不同领域，阅读时要核对动作接口、坐标系和任务类型。只有加入动作条件与决策收益证据，才应进一步主张其控制价值。

## S1｜WAM：世界与动作联合建模

WAM 关注的是“未来世界表征”和“机器人动作”是否在同一个闭环里。下面的工作分别代表三种近期路线：用人类视频做任务条件、异步地想象未来、或者直接把世界演化和动作写进一个因果模型。

| 论文 | 关键点 | 入口 |
| --- | --- | --- |
| Zero-WAM: In-Context World-Action Modeling from Human Videos for Open-Ended Task Generalization（2026，预印本） | 把人类视频当成新任务的上下文提示，不更新参数也能尝试未见任务 | [arXiv](https://arxiv.org/abs/2608.26103) |
| WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time（2026，预印本） | 冻结 WAM，只用无标注的人类视频更新轻量记忆，在测试时改变任务偏好 | [arXiv](https://arxiv.org/abs/2607.06988) |
| GlanceWAM: Sparse Test-Time Imagination for World-Action Models（2026，预印本） | 把未来想象放到控制关键路径之外，后台生成少量前瞻表征，避免同步视频生成拖慢动作 | [arXiv](https://arxiv.org/abs/2608.23927) |
| Riemann-1.0: An Embodied World Action Model for Physical AI（2026，预印本） | 用统一的因果序列同时建模多视角视觉、机器人状态和动作，把策略与可交互世界模拟放进一个模型 | [arXiv](https://arxiv.org/abs/2608.27033) |
| Making Latent Evolution Explicit: Operator-Structured Transitions for World Action Models（2026，预印本） | 用结构化算子描述 latent 的演化，试图让 WAM 的状态转移更容易分析和控制 | [arXiv](https://arxiv.org/abs/2608.27259) |

## S1｜Model-based RL

| 论文 | 关键点 | 入口 |
| --- | --- | --- |
| World Models | latent dynamics 与 controller 的经典 MBRL 前身 | [arXiv](https://arxiv.org/abs/1803.10122) · [Project](https://worldmodels.github.io/) |
| PlaNet: Learning Latent Dynamics for Planning from Pixels | 从像素学习潜空间动力学并做在线规划 | [arXiv](https://arxiv.org/abs/1811.04551) |
| Dream to Control: Learning Behaviors by Latent Imagination | 在潜空间 imagined rollout 上学习 actor-critic | [arXiv](https://arxiv.org/abs/1912.01603) |
| DreamerV3: Mastering Diverse Domains through World Models | 统一超参数、跨域扩展与稳定训练 | [arXiv](https://arxiv.org/abs/2301.04104) · [Code](https://github.com/danijar/dreamerv3) |
| MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model | 只学习决策所需的动态、奖励与价值 | [arXiv](https://arxiv.org/abs/1911.08265) |
| MBPO: When to Trust Your Model | 短模型 rollout 如何权衡模型偏差和样本效率 | [arXiv](https://arxiv.org/abs/1906.08253) |
| TD-MPC2: Scalable, Robust World Models for Continuous Control | 隐式潜空间动力学、价值学习与 MPC 的结合 | [arXiv](https://arxiv.org/abs/2310.16828) · [Project](https://www.tdmpc2.com/) · [Code](https://github.com/nicklashansen/tdmpc2) |

## S1｜Offline / Online RL

| 类别 | 论文 | 核心问题 | 入口 |
| --- | --- | --- | --- |
| Online, on-policy | Proximal Policy Optimization Algorithms (PPO) | 如何用 clipped surrogate 稳定策略更新 | [arXiv](https://arxiv.org/abs/1707.06347) |
| Post-training, group-relative | DeepSeekMath（GRPO） | 用组内相对奖励代替 Value Critic，并以 clip 与 KL 正则更新序列策略 | [arXiv](https://arxiv.org/abs/2402.03300) |
| Post-training, soft-gated | Soft Adaptive Policy Optimization（SAPO） | 用 sigmoid 软门控与正负非对称温度平滑控制 token-level 更新 | [arXiv](https://arxiv.org/abs/2511.20347) |
| Online, off-policy | Soft Actor-Critic (SAC) | 最大熵目标、样本复用与连续控制 | [arXiv](https://arxiv.org/abs/1801.01290) |
| Online, value-based | Human-level control through deep reinforcement learning (DQN) | 离散动作 Q-learning、target network 与 replay buffer | [Nature](https://www.nature.com/articles/nature14236) · [Code](https://github.com/google-deepmind/dqn_zoo) |
| Offline, value-based | Conservative Q-Learning (CQL) | 通过保守 Q 估计抑制 OOD 动作过估计 | [arXiv](https://arxiv.org/abs/2006.04779) |
| Offline, value-based | Implicit Q-Learning (IQL) | 不显式查询数据外动作的价值学习 | [arXiv](https://arxiv.org/abs/2110.06169) |
| Offline, sequence | Decision Transformer | 把控制重写为回报条件序列建模 | [arXiv](https://arxiv.org/abs/2106.01345) · [Code](https://github.com/kzl/decision-transformer) |
| Offline, model-based | MOPO | 用不确定性惩罚模型 rollout 的分布外区域 | [arXiv](https://arxiv.org/abs/2005.13239) |
| Offline, model-based | COMBO | 保守价值学习与模型生成数据结合 | [arXiv](https://arxiv.org/abs/2102.08363) |

### 机器人策略后训练的新方向

这些工作不是新的通用 RL 教科书算法，而是把 RL 接到已经训练好的机器人策略上。重点看它们如何处理失败数据、动作 chunk 和大模型推理延迟。

| 论文 | 关键点 | 入口 |
| --- | --- | --- |
| Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning（2026，预印本） | 冻结行为克隆策略，只训练一个小的 off-policy Q 函数，用成功和失败 rollout 做 Q 引导与自我改进 | [arXiv](https://arxiv.org/abs/2608.21204) |
| Learning to Act While Waiting: RL Finetuning of Generalist Robot Policies Under Inference Latency（2026，预印本） | 用已提交动作和推理中的中间观测补回近似 Markov 状态，让 RL 能在 VLA 推理等待时继续学习 | [arXiv](https://arxiv.org/abs/2608.23831) |
| PAC-ACT: Post-training Actor-Critic for Action Chunking Transformers（2026，预印本） | 在 action chunk 层面做 actor-critic，并保留行为先验，减少接触任务在线探索时的动作漂移 | [arXiv](https://arxiv.org/abs/2607.09590) |
| GRAFT: Grounded and Efficient Online Reinforcement Adaptation for Fine-Grained Robot Manipulation（2026，预印本） | 用任务相关的局部视觉 grounding 和缓存的视觉语言前缀，降低真实机器人在线适配成本 | [arXiv](https://arxiv.org/abs/2608.27079) |

## S2｜数据、基准与评测

| 论文/项目 | 用途 | 入口 |
| --- | --- | --- |
| D4RL: Datasets for Deep Data-Driven Reinforcement Learning | Offline RL 经典数据与评测协议 | [arXiv](https://arxiv.org/abs/2004.07219) · [Code](https://github.com/Farama-Foundation/D4RL) |
| LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning | 语言条件操作与知识迁移评测 | [arXiv](https://arxiv.org/abs/2306.03310) · [Project](https://libero-project.github.io/main.html) |
| CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks | 语言条件长时程闭环与任务链 | [arXiv](https://arxiv.org/abs/2112.03227) · [Code](https://github.com/mees/calvin) |
| ManiSkill2 | 大规模操作任务、数据生成与仿真评测 | [arXiv](https://arxiv.org/abs/2302.04659) · [Code](https://github.com/haosulab/ManiSkill) |
| Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning | 多任务操作与组合泛化 | [Project](https://meta-world.github.io/) · [Code](https://github.com/Farama-Foundation/Metaworld) |
| RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies（2026，预印本） | 统一仿真与真实世界的通用机器人操作评测；覆盖 42 个仿真任务和 18 个真实任务，关注泛化、记忆、精细操作、长时程执行与开放词汇指令 | [arXiv](https://arxiv.org/abs/2607.04434) · [Website](https://robodojo-benchmark.com/) · [Code](https://github.com/robodojo-benchmark/RoboDojo) |
| RoboTwin 2.0 | 双臂数字孪生、数据生成与跨场景泛化 | [Code](https://github.com/RoboTwin-Platform/RoboTwin) |
| RoboCasa | 家庭厨房中的长时程操作与接触任务 | [Code](https://github.com/robocasa/robocasa) |
| Open X-Embodiment | 跨机器人数据规模化与跨本体迁移 | [arXiv](https://arxiv.org/abs/2310.08864) · [Project](https://robotics-transformer-x.github.io/) |
