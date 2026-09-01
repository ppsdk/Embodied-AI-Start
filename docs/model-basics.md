# 模型基础：Transformer、Diffusion 与 Flow Matching

> 🧠 先掌握序列建模和生成模型，再进入 VLA、WM、RL/MBRL 与 WAM 专题。

**预计阅读**：25 min  
**前置知识**：Python、深度学习、张量和概率基础  
**下一步**：[VLA 专题](vla.md) · [WM 专题](world-model-directions.md) · [WAM 专题](wam.md) · [RL / MBRL 专题](mbrl.md)

**本文路线**：统一记号 → Transformer → Diffusion → Flow Matching → 动作输出接口

输入是什么，张量怎样流动，训练时预测什么，部署时怎样得到动作或未来。公式是常见实现的简化写法，具体项目可能换 token 化方式、条件注入位置或采样器。

## 0. 统一记号

| 符号 | 含义 |
| --- | --- |
| $B$ | batch 索引 |
| $L$ | 序列或历史长度 |
| $D$ | 隐表示维度 |
| $T$、$H$ | 时间索引或预测 horizon |
| $A$ | 动作变量 |
| $C$ | 条件上下文 |

## 1. Transformer：序列建模骨架

### 1.1 从输入到 token

Transformer 不限定输入必须是文字。图像 patch、视频时空 patch、语言 token、状态向量和离散 action token 都可以先映射到同一个 hidden dimension：

```text
observation / language / state
        -> tokenizer or encoder
        -> embeddings + position/time information
        -> X: contextual token sequence
        -> Transformer blocks
        -> contextual features
        -> task head: token logits, action regression, diffusion or flow head
```

位置编码、旋转位置编码（RoPE）或相对位置偏置告诉模型 token 的顺序/时空关系。多模态系统通常还需要 modality/type embedding，区分图像、语言、状态和动作 token。

### 1.2 Self-attention 的张量流

对一个 attention head，输入 $X\in\mathbb{R}^{B\times L\times D}$ 经过线性投影得到：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

若 head dimension 为 $d_h$，则 $Q,K,V\in\mathbb{R}^{B\times L\times d_h}$，注意力输出为：

$$
\mathrm{Attn}(X)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
$$

其中 $M$ 是 mask。多头注意力把若干 head 的结果拼接后再投影回 $D$ 维。一个标准 block 还包含 residual connection、LayerNorm 和逐 token 的 MLP/FFN：

$$
X' = X + \mathrm{MHA}(\mathrm{LN}(X)),\qquad
X_{out}=X' + \mathrm{FFN}(\mathrm{LN}(X')).
$$

因此 Transformer block 通常保持序列结构；变化主要发生在任务 head，而不是 backbone 本身。

### 1.3 Mask 决定“能看见什么”

- **双向/全注意力**：一个 token 可以读取整个上下文，常见于图像理解、JEPA encoder 或条件编码器。
- **因果 mask**：位置 $i$ 只能读取不晚于 $i$ 的 token，常见于自回归语言/动作 token 生成。
- **混合 mask**：图像和指令 token 可双向互读，而待生成的动作 token 仍按时间因果读取；具体规则必须查看实现。

Mask 是信息流约束，不是训练目标本身。使用 Transformer 不代表模型一定是 next-token，也不代表它一定能生成视频或动作。

### 1.4 常见输出头与目标

1. **Next-token / action-token head**：对每个位置输出词表或动作码本 logits，用交叉熵训练：

   $$
   \mathcal{L}_{AR}=-\sum_i\log p_\theta(y_i\mid y_{<i},C).
   $$

   连续动作先量化为离散 bin/token；推理时自回归采样或贪心解码。
2. **连续回归 head**：直接回归动作变量，用 L1、L2、Huber 或高斯负对数似然拟合。简单但在多峰行为上可能产生平均动作。
3. **Diffusion head**：Transformer 提取条件 $C$，另一个网络接收带噪动作 $x_t$ 并预测去噪目标。
4. **Flow-matching head**：Transformer 提取条件 $C$，网络接收路径上的动作 $x_t$ 和时间 $t$，预测速度场 $v_\theta(x_t,t,C)$。

Transformer 是 backbone/表示与条件融合机制；diffusion 和 flow matching 是生成目标与推理路径。三者可以组合，但不是同义词。

## 2. Diffusion：逐步去噪生成动作或未来

### 2.1 前向加噪

以干净动作块 $x_0\in\mathbb{R}^{B\times H\times A}$ 为例，选一个噪声步 $t$ 和噪声 $\epsilon\sim\mathcal{N}(0,I)$，常见 DDPM 参数化为：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

噪声日程（noise schedule）决定 $\bar\alpha_t$ 如何从接近 1 变到接近 0。训练时通常随机采样 $t$，并让网络根据条件 $C$ 从 $x_t$ 恢复信息。

### 2.2 训练目标

最常见的是噪声预测：

$$
\mathcal{L}_{\epsilon}=\mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon-\epsilon_\theta(x_t,t,C)\|_2^2\right].
$$

也可以预测原始样本 $x_0$，或预测 $v$-parameterization（在信噪比变化下通常更稳定）。阅读代码时要确认网络输出到底对应 `epsilon`、`x0` 还是 `v`，以及 loss 是否有 timestep/SNR 加权；不能仅凭“diffusion”这个名称推断。

### 2.3 推理：反复执行反向更新

从高斯噪声 $x_T$ 开始，按 $T\rightarrow0$ 的顺序调用去噪网络和 scheduler，得到 $x_0$：

```text
x_T ~ Normal(0, I)
for t = T ... 1:
    prediction = denoiser(x_t, t, condition C)
    x_{t-1} = scheduler_step(prediction, x_t, t)
return x_0[0:H]  # action chunk or future latent
```

在机器人策略中，$ x $ 可以是未来 $H$ 步的关节/末端动作，而不是单个动作；执行一小段后重新观测和采样（receding horizon）。采样步数、action horizon、控制频率和端到端延迟应一并报告。

### 2.4 为什么适合连续、多峰动作

同一个观测可能对应多个合理的抓取方向或绕障路径。显式建模条件分布并从噪声采样，通常比单一 L2 回归更能保留多种模式；代价是多步采样、调度器敏感性和更高推理延迟。Diffusion Policy 是机器人动作 chunk 的代表性应用。

## 3. Flow Matching：学习速度场并积分 ODE

### 3.1 路径与条件向量场

Flow matching 不把推理写成一串离散去噪步，而是学习一个随时间变化的向量场，把简单分布传输到数据分布。最直观的直线路径取噪声 $x_0$ 和数据样本 $x_1$：

> 注意：这里的 $x_0$ 表示 flow path 的源噪声；它和上一节 diffusion 中表示“干净动作”的 $x_0$ 只是记号相同，语义不同。

$$
x_t=(1-t)x_0+t x_1,\qquad t\in[0,1],
$$

其目标速度为 $u_t=x_1-x_0$。模型学习：

$$
\mathcal{L}_{FM}=\mathbb{E}_{t,x_0,x_1,C}\left[\|v_\theta(x_t,t,C)-u_t\|_2^2\right].
$$

实际方法可以使用不同 probability path、条件构造和参数化；上式用于解释“预测速度”这一核心机制。

### 3.2 推理：ODE 数值积分

推理从噪声分布采样 $x(0)$，解条件常微分方程：

$$
\frac{d x(t)}{dt}=v_\theta(x(t),t,C),\qquad x(1)\approx x_1.
$$

Euler、Heun 或其他 ODE solver 用有限步近似积分：

```text
x = sample_noise()
for t in solver_grid(0 -> 1):
    x = x + step_size * velocity(x, t, condition C)
return x
```

因此 flow matching 的核心输出是 velocity/向量场，不是 DDPM 意义下的噪声残差；采样步数和 solver 仍会影响速度、稳定性和动作质量。π0 将 flow-based action expert 放在 VLM 条件之后，是具身场景中的重要例子。

### 3.3 与 diffusion 的区别

二者都可以从简单分布生成连续动作，也都能使用 Transformer 条件编码器，但训练/采样的语义不同：diffusion 学习去噪或等价 score 参数化，flow matching 学习路径上的速度场并做 ODE 积分。某些连续时间 diffusion、rectified flow 或蒸馏方法会让区别变得不那么明显；比较论文时应记录实际 loss、路径、solver 和采样步数，而不是只看方法标签。

## 4. 四类动作输出的对照

| 输出机制        | 训练目标                           | 推理方式        | 优点                           | 主要代价/风险                           | 典型位置                               |
| --------------- | ---------------------------------- | --------------- | ------------------------------ | --------------------------------------- | -------------------------------------- |
| 离散 next-token | token 交叉熵                       | 自回归解码      | 复用语言模型、接口清晰         | 量化误差、序列延迟、动作粒度受 bin 影响 | 离散 action-token VLA                  |
| 连续回归        | L1/L2/Huber/NLL                    | 一次前向        | 快、实现简单                   | 多峰分布可能平均化，长时程相关性弱      | VLA action head、低延迟控制            |
| Diffusion       | 预测$\epsilon/x_0/v$ 的去噪 loss | 多步反向去噪    | 多模态动作、平滑 action chunk  | 采样延迟、scheduler/SNR 敏感            | Diffusion Policy、部分 VLA action head |
| Flow matching   | 速度场回归                         | ODE solver 积分 | 连续路径、可用少步 solver/蒸馏 | path/solver 选择、速度场误差            | π0 类 flow action expert              |

Transformer、diffusion、flow 不是互斥选项：Transformer 往往是 backbone，后面接 token、回归、diffusion 或 flow head。

## 5. 进入研究专题

本页的基础模块会在不同方向中复用，但研究问题分别展开：

| 方向 | 关注点 | 入口 |
| --- | --- | --- |
| VLA | 多模态上下文如何变成可执行动作 | [VLA 专题](vla.md) |
| WM | 动作条件下未来视频、latent、对象或 3D/4D 如何演化 | [WM 专题](world-model-directions.md) |
| WAM | 未来预测如何参与动作生成、排序或约束 | [WAM 专题](wam.md) |
| RL / MBRL | 奖励、价值、策略更新和模型 rollout 如何闭环 | [RL / MBRL 专题](mbrl.md) · [强化学习基础](reinforcement-learning.md) |

这些专题会进一步说明数据字段、训练目标、推理流程、失败模式和评测协议；本页只保留可复用的模型原语。

## 6. 基础论文

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer 与 self-attention 的原始定义。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)：DDPM 前向加噪、反向去噪和变分目标。
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)：条件向量场与 probability path 的训练框架。
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)：Transformer 作为扩散生成 backbone 的代表。
- [Diffusion Policy](https://arxiv.org/abs/2303.04137)：连续机器人动作 chunk 的 diffusion policy。
