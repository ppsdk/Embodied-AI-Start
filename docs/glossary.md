# 术语表

| 缩写/术语 | 英文 | 简明解释 |
| --- | --- | --- |
| Embodied AI | Embodied Artificial Intelligence | 在环境中通过感知与动作闭环完成任务的智能系统。 |
| MDP | Markov Decision Process | 用状态、动作、转移、奖励和折扣描述序贯决策。 |
| POMDP | Partially Observable MDP | 智能体只能看到不完整观测，需要用历史或记忆推断状态。 |
| Configuration $q$ | Joint Configuration | 机器人所有关节位置/角度组成的配置向量。 |
| DOF | Degrees of Freedom | 可以独立改变的运动自由度数量。 |
| Coordinate Frame | Coordinate Reference Frame | 表示位置、姿态和速度的参考坐标系；必须明确 frame tree。 |
| SO(3) | Special Orthogonal Group in 3D | 三维合法旋转的集合；$R^\mathsf{T}R=I$ 且 $\det(R)=1$，只有 3 个旋转自由度。 |
| SE(3) | Special Euclidean Group in 3D | 三维刚体位姿的集合；元素是包含 $R\in SO(3)$ 和 $p\in\mathbb{R}^3$ 的 $4\times4$ 齐次变换，有 6 个自由度。 |
| 3D Rotation | Three-dimensional Rotation | $SO(3)$ 中的一个旋转；可用 Euler 角、旋转向量、四元数或旋转矩阵等不同参数化表示。 |
| Rotation Vector | Axis-angle / Rotation Vector | 三维向量 $r=\theta u$；方向是旋转轴，长度是弧度角，通过 Rodrigues/指数映射得到 $SO(3)$。 |
| Quaternion | Unit Quaternion | 用 4 个数表示三维旋转；必须满足单位范数，且 $q$ 与 $-q$ 表示同一旋转。ROS 2 消息字段顺序通常是 `x,y,z,w`。 |
| 6D Rotation | 6D Continuous Rotation Representation | 用两个三维向量经正交化恢复 $3\times3$ 旋转矩阵；是 6 个表示数，不是 6 个旋转自由度，也不是 $SE(3)$ 位姿。 |
| SE(3)-Equivariant | SE(3)-Equivariant Model | 输入整体经过刚体变换后，输出按相应的旋转/平移规则一起变换；例如向量输出 $v$ 变为 $Rv$。 |
| SE(3)-Invariant | SE(3)-Invariant Model/Feature | 输入整体经过刚体变换后，输出保持不变；例如物体类别、点间距离或碰撞判定。 |
| E(3) | Euclidean Group in 3D | 三维旋转、平移以及镜像反射的变换集合；相比 $SE(3)$ 还允许 $\det(R)=-1$ 的反射。 |
| Irrep | Irreducible Representation | 群作用下不能再拆分的基本表示；在 $SO(3)$ 等变网络中常按阶数 $l$ 组织标量、向量和高阶特征。 |
| Spherical Harmonics | Spherical Harmonic Basis | 定义在球面方向上的一组基函数；常用于把相对方向编码成可组合的旋转特征。 |
| Equivariant Layer | Equivariant Neural Layer | 输入和输出按群表示变换的网络层；常用相对位置、标量门控、张量积或几何 attention 保持等变。 |
| Equivariant Diffusion | Equivariant Diffusion / Score Model | 让加噪过程、噪声/score/速度预测和每一步去噪更新遵守同一群作用，使生成的动作或轨迹随坐标变换而变换。 |
| Score | Score Function | $s_t(x)=\nabla_x\log p_t(x)$，扩散模型中描述带噪分布对输入的对数密度梯度；等变 score 会按向量表示变换。 |
| FK | Forward Kinematics | 根据关节配置计算末端位姿，$x=f(q)$。 |
| IK | Inverse Kinematics | 根据目标末端位姿求关节配置，需处理多解、限位和碰撞。 |
| Jacobian | Geometric / Analytical Jacobian | 将关节速度映射为末端速度，$\dot{x}=J(q)\dot{q}$。 |
| Task Space | Task / Cartesian Space | 以末端位置、姿态或力描述动作的空间，与 joint space 相对。 |
| Proprioception | Proprioceptive Sensing | 关节位置、速度、力矩等机器人自身状态观测。 |
| Impedance Control | Impedance Control | 通过虚拟刚度和阻尼调节位置误差到力/运动的关系。 |
| Hand-eye Calibration | Hand-eye Calibration | 通过多组机器人末端位姿与标定板观测，估计相机、末端和机器人基座之间的刚性外参；分为 eye-in-hand 与 eye-to-hand。 |
| Policy | Policy | 从观测/状态到动作分布的映射 $\pi(a\mid s)$。 |
| Return $G_t$ | Discounted Return | 从时刻 $t$ 开始的折扣累计奖励，是价值函数要估计的长期目标。 |
| $V^\pi(s)$ | State-Value Function | 从状态 $s$ 出发并遵循策略 $\pi$ 时的期望累计回报。 |
| $Q^\pi(s,a)$ | Action-Value Function | 在状态 $s$ 先执行动作 $a$、之后遵循策略 $\pi$ 时的期望累计回报。 |
| $A^\pi(s,a)$ | Advantage Function | 动作相对状态基准的价值，$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$。 |
| TD | Temporal-Difference Learning | 用即时奖励和下一状态价值的 bootstrap target 更新当前价值。 |
| Bootstrap | Bootstrapping | 用已有价值估计构造新的价值监督目标，而不等待完整轨迹结束。 |
| Replay Buffer | Experience Replay Buffer | 保存历史转移并重复采样，以复用数据并降低连续轨迹相关性。 |
| Target Network | Target Network | 缓慢更新的网络副本，用于构造相对稳定的 Bellman target。 |
| RL | Reinforcement Learning | 通过最大化累计回报学习策略。 |
| Offline RL | Offline Reinforcement Learning | 只从固定数据集学习，训练时不继续与环境交互。 |
| Online RL | Online Reinforcement Learning | 训练中用当前策略继续与环境交互并收集新数据。 |
| Offline-to-Online | Offline-to-Online RL | 先用离线数据初始化，再通过在线交互继续优化。 |
| Model-free RL | Model-free Reinforcement Learning | 不显式学习并规划环境动力学，直接学习价值或策略。 |
| Model-based RL / MBRL | Model-based Reinforcement Learning | 学习/使用动力学或奖励模型做 imagined rollout、规划、价值估计或策略优化；它是决策/RL 路线，不等于所有 WM。 |
| On-policy | On-policy RL | 主要使用当前策略采样的数据更新当前策略，例如 PPO。 |
| Off-policy | Off-policy RL | 可用其他/旧策略数据更新当前策略，例如 SAC。 |
| GRPO | Group Relative Policy Optimization | 对同一输入成组采样，用组内相对奖励构造 Advantage，并以 PPO 式 clip 与 KL 正则更新策略；不训练独立 Value/Critic。 |
| SAPO | Soft Adaptive Policy Optimization | 在 group-based 后训练中以温度控制的 sigmoid 软门控替代 hard clip，并用更高负向温度更快抑制 off-policy 的负 Advantage 更新。 |
| BC | Behavior Cloning | 用监督学习直接拟合示范动作。 |
| IL | Imitation Learning | 从专家示范学习行为的总称，BC 是最常见形式。 |
| OOD | Out-of-Distribution | 超出训练数据覆盖范围的观测、动作、任务或环境。 |
| WM | World Model | 广义的环境表征、未来预测和场景生成范式，可覆盖 JEPA latent、视频、状态和 3D/4D 表示；不要求自带 planner。 |
| JEPA | Joint Embedding Predictive Architecture | 在表征空间预测目标时空块，通常避免逐像素重建，强调可预测和可迁移的 latent dynamics。 |
| Video World Model | Video World Model | 根据历史、动作或条件预测/生成未来视频或视频潜变量的世界模型。 |
| 3D/4D World Model | 3D/4D World Model | 建模几何、对象、视角与时间演化的世界表征，可使用点云、3D Gaussian 或隐式场。 |
| Action-conditioned WM | Action-conditioned World Model | 将 action 作为条件并检验未来表征/视频/3D 状态是否对动作敏感。 |
| WAM | World Action Model | 将世界未来预测与动作生成联合或紧密耦合的具身模型范式。 |
| VLM | Vision-Language Model | 联合处理视觉和语言的多模态模型。 |
| VLA | Vision-Language-Action Model | 由视觉、语言及可能的状态历史直接生成机器人动作的模型。 |
| Transformer | Transformer | 以 self-attention 和前馈网络处理 token 序列的 backbone；可接 next-token、回归、diffusion 或 flow head。 |
| Self-Attention | Self-Attention | 用 Query/Key/Value 计算序列内 token 关系的注意力机制。 |
| Q/K/V | Query / Key / Value | attention 中用于匹配、寻址和值聚合的三组线性投影。 |
| Causal Mask | Causal Attention Mask | 限制位置只能读取当前及过去 token 的 mask，常用于自回归生成。 |
| Hidden State | Hidden State / Representation | Transformer 中间层的上下文表征，常见形状为 `[B,L,D]`。 |
| Action Token | Action Tokenization | 将连续动作离散化为 token，用序列模型预测。 |
| Action Chunk | Action Chunking | 一次预测未来多个时间步动作，降低决策频率并利用局部时序结构。 |
| Diffusion | Diffusion / Denoising Diffusion | 先向数据加噪、再学习反向去噪的生成建模框架。 |
| Noise Schedule | Noise Schedule | 定义 diffusion 各时间步信噪比或噪声强度的日程。 |
| Epsilon / x0 / v Prediction | Epsilon / x0 / v Parameterization | diffusion 网络预测噪声、干净样本或速度参数化的三类常见目标。 |
| Denoising Step | Denoising Step | 给定带噪样本、时间步和条件，执行一次 scheduler 反向更新。 |
| Flow Matching | Flow Matching | 学习连续向量场，将简单分布传输到目标动作分布。 |
| Velocity Field | Conditional Velocity Field | flow matching 中的条件速度函数 $v_\theta(x,t,C)$，描述样本沿生成路径的变化方向。 |
| Probability Path | Probability Path | 从源分布到数据分布的连续概率路径；flow matching 在路径上回归速度。 |
| ODE Sampling | Ordinary Differential Equation Sampling | 从噪声初值出发，用 ODE solver 积分速度场得到样本。 |
| DiT | Diffusion Transformer | 用 Transformer block 作为 diffusion/视频生成 backbone 的架构范式。 |
| Diffusion Policy | Diffusion-based Policy | 通过条件去噪过程生成动作序列的策略。 |
| MPC | Model Predictive Control | 用模型滚动预测有限时域，优化动作后只执行前几步并重新规划。 |
| Latent Dynamics | Latent Dynamics Model | 在压缩的潜空间而非原始像素/状态空间中预测演化。 |
| Imagined Rollout | Imagination Rollout | 在学习到的动力学/奖励模型中生成虚拟轨迹，常用于 MBRL 的规划、价值或策略更新。 |
| Model Bias | Model Bias | 学习模型与真实环境不一致导致的系统性决策误差。 |
| Sim-to-Real | Simulation-to-Reality | 将仿真中训练的策略迁移到真实机器人。 |
| Teleoperation | Teleoperation | 人类远程控制机器人，用于完成任务或采集示范。 |
| Embodiment | Embodiment | 机器人的物理形态、自由度、传感器与动作接口。 |
| Generalist Policy | Generalist Robot Policy | 试图覆盖多任务、多场景乃至多本体的通用机器人策略。 |
| Post-training | Post-training | 在预训练模型基础上进行监督微调、偏好优化或 RL 等后续训练。 |
| Rollout | Rollout | 策略在环境或模型中从起点执行得到的一段轨迹。 |
| Stage | USD Stage | Isaac Sim 当前 USD 场景树，包含机器人、物体、灯光、相机和物理设置。 |
| Prim | USD Primitive | Stage 中的一个节点，例如机器人、关节、相机或碰撞体。 |
| Articulation | Articulation | Isaac Sim 中由关节连接的一组刚体，通常对应一个机器人。 |
| Physics step | Physics Step | 仿真器推进一次物理积分的时间步；可与渲染和策略步不同。 |
| Decimation | Action Decimation | 一个策略动作保持的物理步数，常写为 policy_dt / physics_dt。 |
| TF / tf2 | Transform Library | ROS 2 Humble 中带时间戳的坐标变换树与查询库；用于在 frame 之间转换位姿、点和向量。 |
| rclpy | ROS 2 Python Client Library | ROS 2 Humble 的 Python 客户端库；提供 `Node`、参数、定时器、通信接口和 executor 调度，常用于编写 TF/传感器/控制节点。 |
| Topic | ROS 2 Topic | ROS 2 中持续发布消息的一对多通信接口，常用于图像、关节状态、TF 和传感器数据。 |
| Service | ROS 2 Service | ROS 2 中一次请求对应一次响应的短操作接口，适合复位、查询和触发操作。 |
| Action | ROS 2 Action | ROS 2 中带反馈、可取消和最终结果的长任务接口，常用于轨迹执行和导航。 |
| QoS | Quality of Service | ROS 2 消息传输策略，包括 reliability、durability、history、depth、deadline 和 lifespan；发布者与订阅者的关键 QoS 必须兼容。 |
| OpenCV | Open Source Computer Vision Library | 计算机视觉库；ROS 2 中通常通过 `cv_bridge` 在 `sensor_msgs/Image` 与 `cv::Mat`/`numpy.ndarray` 之间转换。 |
| cv_bridge | ROS OpenCV Bridge | ROS 2 图像消息与 OpenCV 图像之间的转换包，需与当前 ROS 2 发行版和 Python/C++ 环境匹配。 |
| image_transport | ROS Image Transport | ROS 2 图像传输接口，统一处理 raw、compressed 等图像传输插件。 |
| RViz 2 | ROS 2 Visualization Tool | ROS 2 Humble 的三维可视化与交互工具；订阅 topic 并通过 TF 显示机器人、传感器和 MoveIt 2 规划场景，本身不负责仿真或底层控制。 |
| Planning Frame | Planning Frame | MoveIt 2 Humble 规划所使用的参考坐标系，目标位姿必须明确表达在哪个 frame 中。 |
| Planning Scene | Planning Scene | MoveIt 2 Humble 保存机器人状态、障碍物、附着物体和碰撞规则的场景快照。 |
| SRDF | Semantic Robot Description Format | 在 URDF 之上描述 MoveIt 2 Humble 语义信息的文件格式，例如 planning group、末端执行器和禁碰对。 |
| ros2_control | ROS 2 Control | ROS 2 Humble 的硬件接口与控制器框架，MoveIt 2 通常通过轨迹控制器执行规划结果。 |
| controller_manager | ros2_control Controller Manager | 管理硬件接口和控制器加载、配置、激活与切换的 ros2_control 节点。 |
| rosbag2 | ROS 2 Bag | ROS 2 的消息记录与回放工具，用于复查时间同步、TF、传感器和控制器反馈；回放不等于重现真实硬件动力学。 |
| Success Rate | Task Success Rate | 多次评测中满足成功判定的比例；需同时报告任务、初始条件和 seed。 |

## 容易混淆的概念

### Offline RL vs Off-policy RL

- **Offline RL**：训练期没有新的环境交互，数据集固定。
- **Off-policy RL**：更新策略时允许使用非当前策略产生的数据；它可以发生在 online 训练中。

### WM、MBRL、WAM 的边界

三者的详细判别见[知识图谱](knowledge-map.md)；本页只保留术语定义，不重复展开。

### VLA vs WAM

- VLA 关注从视觉/语言到动作的策略映射。
- WAM 强调未来世界与动作之间的联合或耦合建模。
- 两者边界可能重叠；判断时应看训练目标和部署时数据流，而不只看论文自称。

### Transformer vs Diffusion vs Flow Matching

- **Transformer** 通常是处理 token 和条件信息的 backbone；它本身不规定输出必须是离散 token、diffusion 还是 flow。
- **Diffusion** 训练去噪/重建参数化，推理沿反向噪声日程迭代；**Flow Matching** 训练路径上的速度场，推理用 ODE solver 积分。
- 二者都可以由 Transformer 实现条件网络，也都可以生成连续 action chunk；应比较实际 loss、路径、采样步数和闭环延迟，而不是只比较名称。
