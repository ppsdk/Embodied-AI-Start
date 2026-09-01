# XPolicyLab 教程：把策略接到 RoboDojo

> 🛠️ 从 debug 评测开始，把策略接入 RoboDojo，再扩展到仿真和服务化运行。

**预计阅读**：15 min
**前置知识**：Python、Git、策略的 observation/action 定义
**下一步**：[Benchmark 指南](benchmarks.md) · [强化学习基础](reinforcement-learning.md) · [代码仓](codebases.md)

**本文路线**：官方入口 → 安装 → 数据 → debug 评测 → 适配器 → 仿真/服务

[XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 是机器人策略和评测环境之间的一层适配器。策略模型保留自己的依赖、checkpoint 和推理代码；XPolicyLab 负责把观测送给模型、把动作交给环境，并用同一套接口连接 RoboDojo、RoboTwin 或真实机器人。

```text
策略侧：模型、checkpoint、预处理、推理服务
                 <--- websocket --->
环境侧：RoboDojo / RoboTwin / Isaac Sim / 真实机器人客户端
```

## 1. 官方入口

- [XPolicyLab GitHub](https://github.com/XPolicyLab/XPolicyLab)
- [XPolicyLab 官网](https://xpolicylab.github.io/)
- [XPolicyLab 论文](https://arxiv.org/abs/2608.09892)
- [RoboDojo 官网](https://robodojo-benchmark.com/)
- [RoboDojo 代码](https://github.com/robodojo-benchmark/RoboDojo)

本文以 Linux shell 为例。策略的 Python、CUDA、模型权重和 Isaac Sim 版本必须以对应策略 README 及 RoboDojo 当前文档为准；不同策略不能共用一套环境假设。

## 2. 安装最小环境

```bash
mkdir -p xpolicylab-demo
cd xpolicylab-demo
git clone https://github.com/XPolicyLab/XPolicyLab.git
cd XPolicyLab
python -m pip install -e .
```

先确认包可以导入：

```bash
python -c "import XPolicyLab; print('XPolicyLab import ok')"
```

如果要运行真实的 RoboDojo 仿真，还要按 RoboDojo 文档安装 Isaac Sim、环境客户端和任务资产。没有仿真器时，可以先完成下面的 debug 检查。

## 3. 下载一小份 RoboDojo 数据

XPolicyLab 提供下载脚本，demo 数据会放到 `XPolicyLab/` 的上一级 `data/` 目录：

```bash
# 当前目录：xpolicylab-demo/XPolicyLab
bash scripts/RoboDojo/download_robodojo_data.sh demo
```

目录大致如下：

```text
xpolicylab-demo/
├── data/
└── XPolicyLab/
```

完整导出还包括 `hdf5`、`lerobot_v3.0`、`lerobot_v2.1` 和 `real`。先用 demo 验证数据读取、模型加载和接口连通，再考虑下载完整数据。

## 4. 先做 debug 评测

`debug` 后端不启动 Isaac Sim，也不连接机器人，只检查服务启动、观测序列化、动作键名、动作维度和 batch 逻辑：

```bash
export EVAL_ENV_TYPE=debug
cd policy/demo_policy
bash install.sh
bash eval.sh RoboDojo stack_bowls demo arx_x5 joint 0 0 0 base base
```

参数按顺序是：

| 参数                      | 含义                          | 示例                  |
| ------------------------- | ----------------------------- | --------------------- |
| `bench_name`            | benchmark 或数据族            | `RoboDojo`          |
| `task_name`             | 本次评测任务                  | `stack_bowls`       |
| `ckpt_name`             | checkpoint 名称或路径         | `demo`、`cotrain` |
| `env_cfg_type`          | 机器人/相机/场景配置          | `arx_x5`            |
| `action_type`           | 动作空间                      | `joint` 或 `ee`   |
| `seed`                  | 训练或评测 seed               | `0`                 |
| `policy_gpu_id`         | 策略服务使用的 GPU            | `0`                 |
| `env_gpu_id`            | 环境客户端使用的 GPU          | `0`                 |
| `policy_env_or_uv_path` | 策略侧 conda 环境名或 uv 路径 | `base`              |
| `eval_env_conda_env`    | 环境侧 conda 环境名           | `base`              |

这里的 `base` 只是 demo 的占位环境名；真实策略应换成该策略 README 要求的环境。

## 5. 适配器的核心接口

每个策略放在 `policy/<POLICY>/` 下。核心接口是 `model.py` 中的 `Model` 类：

| 方法                                    | 作用                                               |
| --------------------------------------- | -------------------------------------------------- |
| `__init__(model_cfg)`                 | 读取 `deploy.yml`，加载模型、处理器和 checkpoint |
| `update_obs(obs)`                     | 接收一条观测并更新模型状态                         |
| `update_obs_batch(obs_list)`          | 批量更新观测                                       |
| `get_action()`                        | 返回一段动作 chunk                                 |
| `get_action_batch(env_idx_list=None)` | 按环境索引返回批量动作 chunk                       |
| `reset()`                             | 清除当前 episode 的历史状态，不接收参数            |

策略服务已经把相机颜色解码成 RGB 数组，因此运行时 `obs["vision"][camera]["color"]` 是图像数组，`model.py` 不应再次解码。

## 6. 观测和动作长什么样

运行时观测是字典，常见字段如下：

```text
obs = {
  "instruction": "stack the bowls",
  "vision": {
    "cam_head": {
      "color": (H, W, 3) RGB,
      "depth": (H, W),
      "intrinsic_matrix": (3, 3),
      "extrinsics_matrix": (4, 4)
    }
  },
  "state": {
    "arm_joint_state": (DOF,),
    "ee_pose": (7,)  # [x, y, z, qw, qx, qy, qz]
  }
}
```

不同机器人可能使用 `left_*`、`right_*` 或单臂字段；不要根据字段名猜维度，先看 `env_cfg_type` 和 `get_robot_action_dim_info()`。轨迹文件使用复数键名（如 `colors`、`arm_joint_states`），相机外参键名是 `extrinsic_matrix`，这和运行时的 `extrinsics_matrix` 有意不同。

动作可以是关节目标、末端位姿或增量动作，通常按动作 chunk 返回。必须确认动作键名、顺序、单位、限幅和控制频率；策略输出不一定等于最终送入机器人的控制目标。

离线转换统一使用：

```python
from XPolicyLab.utils.load_file import load_hdf5
from XPolicyLab.utils.process_data import decode_image_bit, get_robot_action_dim_info
```

不要在数据转换脚本里自行用 `cv2.imdecode`、`np.frombuffer` 或 PIL 替换 `decode_image_bit`。运行时则不解码，因为服务端已经完成这一步。

## 7. 从 debug 走到仿真

确认 debug 通过后，再按 RoboDojo 文档安装环境侧依赖，并将 XPolicyLab 放在 RoboDojo 的环境配置、任务和客户端目录旁边。然后切换到仿真后端：

```bash
export EVAL_ENV_TYPE=sim
```

然后重新运行同一条 `eval.sh`。第一次只跑一个任务、一个 seed 和少量 episode，记录以下信息：RoboDojo 版本、Isaac Sim 版本、机器人配置、动作空间、相机分辨率、控制频率、seed、GPU 和成功判定。

## 8. 策略服务和环境客户端分机运行

策略模型很大时，可以把模型和仿真器拆到两台机器。策略机启动 websocket 服务：

```bash
cd policy/<POLICY>
bash setup_eval_policy_server.sh \
  RoboDojo <task> <ckpt> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <policy_env_or_uv_path> <port> 0.0.0.0
```

环境机连接策略机：

```bash
cd policy/<POLICY>
bash setup_eval_env_client.sh \
  RoboDojo <task> <ckpt> <env_cfg_type> <action_type> <seed> \
  <env_gpu_id> <eval_env_conda_env> \
  "ckpt_name=<ckpt>,action_type=<action_type>" <port> <policy_ip>
```

`0.0.0.0` 只用于服务端监听，客户端必须填写策略机的真实 IP。常见的连接超时、重连和模型冷启动由 XPolicyLab 传输层处理；先看策略服务端日志中的完整 traceback。

## 9. 接入自己的策略

复制最小模板：

```bash
bash scripts/create_policy.sh MY_POLICY
```

然后依次完成：

1. 在 `README.md` 写清楚安装、checkpoint、训练和评测命令。
2. 实现 `model.py` 的 `Model` 接口，先让 `EVAL_ENV_TYPE=debug` 通过。
3. 在 `deploy.yml` 固定协议、端口、checkpoint 和运行时默认值。
4. 需要数据转换或训练时，再补 `process_data.sh` 和 `train.sh`。
5. 运行静态检查：

```bash
git diff --check
bash -n policy/MY_POLICY/*.sh
python -m py_compile policy/MY_POLICY/model.py policy/MY_POLICY/deploy.py
```

完整模板见 [policy/demo_policy](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/demo_policy)；不要直接复制某个具体 VLA 的依赖和动作维度。

## 10. 常见问题

| 现象                        | 优先检查                                                          |
| --------------------------- | ----------------------------------------------------------------- |
| 服务能启动但动作维度错      | `env_cfg_type`、`action_type`、动作键名和机器人维度           |
| 图像颜色或形状不对          | 是否重复解码，是否把 RGB 当成 BGR                                 |
| 每个 episode 都带着上次历史 | 是否在环境 reset 后调用 `Model.reset()`                         |
| 仿真客户端连不上服务        | 端口、防火墙、策略机 IP，以及服务端是否监听 `0.0.0.0`           |
| debug 通过、sim 失败        | Isaac Sim/RoboDojo 版本、Prim/机器人配置、控制频率和单位          |
| 结果无法比较                | benchmark 任务子集、seed、评测次数、初始场景和 success 判定不一致 |

XPolicyLab 的价值是统一“策略如何被调用”和“环境如何拿到动作”。它不会替你解决模型训练、机器人标定、碰撞安全或 benchmark 协议设计；这些仍由策略、机器人和 RoboDojo 各自负责。
