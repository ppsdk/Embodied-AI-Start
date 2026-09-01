# MuJoCo 仿真教程

> 🧪 用 MuJoCo 和 Gymnasium 跑通一个基础的动力学与 RL 实验。

**预计阅读**：20 min  
**前置知识**：Python、虚拟环境和基本 RL 概念  
**下一步**：[强化学习基础](reinforcement-learning.md) · [机器人学基础](robotics.md)

**本文路线**：安装 → MJCF → 仿真循环 → 状态与传感器 → Gymnasium → 自行实验

MuJoCo 是一个轻量的机器人仿真器，适合先跑通动力学、接触和 RL 循环。机器人资产常从 URDF 开始，但 MuJoCo 真正运行的是 MJCF。本文只带你完成一条最短路径：安装、加载模型、推进仿真、读取状态，最后接上 Gymnasium。

官方入口：

- [MuJoCo 文档](https://mujoco.readthedocs.io/)
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo GitHub](https://github.com/google-deepmind/mujoco)

## 1. 安装

现代 Python 包不需要安装旧版 mujoco-py：

~~~bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
~~~

需要标准 RL 环境格式时再安装：

~~~bash
python -m pip install "gymnasium[mujoco]" stable-baselines3
~~~

记录实验时固定 Python、MuJoCo、Gymnasium、操作系统、渲染后端和随机种子。

## 1.1 现成的 Gymnasium + MuJoCo RL 项目

如果目标是先跑一个完整的连续动作 RL 训练循环，优先使用 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的单文件 PPO 实现。它的 `ppo_continuous_action.py` 直接导入 `gymnasium`，默认环境是 `HalfCheetah-v4`；代码里包含 rollout、GAE、PPO 更新、日志和可选模型保存。

```bash
git clone https://github.com/vwxyzjn/cleanrl.git
cd cleanrl
python -m pip install -r requirements/requirements-mujoco.txt
python cleanrl/ppo_continuous_action.py \
  --env-id HalfCheetah-v4 \
  --total-timesteps 100000
```

想要统一训练、评测、调参和画图脚本，可以改用 [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)：

```bash
git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
cd rl-baselines3-zoo
pip install -e .
python train.py --algo sac --env HalfCheetah-v4
```

两者定位不同：CleanRL 适合读懂一个算法文件；RL Baselines3 Zoo 适合批量实验。它们都使用 Gymnasium/Stable-Baselines3 生态，不等于 MuJoCo 本身提供了 PPO。

## 2. URDF、MJCF 和 MuJoCo

三者的关系如下：

~~~text
URDF：机器人资产描述和 ROS 生态交换
  -> 导入/转换后检查
MJCF：MuJoCo 的原生模型，补充执行器、传感器、接触和仿真选项
  -> MjModel + MjData
~~~

URDF 主要描述机器人树、关节、惯性、视觉和碰撞几何；它不是完整的仿真任务文件。导入 URDF 后，应检查关节轴、限位、惯性、网格路径、碰撞几何和单位，并确认执行器、传感器及接触参数是否需要在 MJCF 中补充。不要把 URDF 文件直接当成已经校验过的 MuJoCo 模型。

## 3. 第一个 MJCF 模型

MJCF 是 MuJoCo 使用的 XML 模型格式。下面的模型有一个铰链关节和一个位置执行器，保存为 model.xml：

~~~xml
<mujoco model="one_joint">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.05" density="500"/>
    </body>
  </worldbody>
  <actuator>
    <position name="hinge_pos" joint="hinge" kp="10"/>
  </actuator>
</mujoco>
~~~

几个核心对象：

| 对象 | 含义 |
| --- | --- |
| MjModel | 编译后的模型结构和固定参数，例如关节、几何体、执行器数量 |
| MjData | 当前仿真状态和计算结果，例如位置、速度、接触和时间 |
| joint | 定义自由度和坐标 |
| geom | 定义可视几何和碰撞几何 |
| actuator | 定义控制输入如何作用到关节 |

model.nq、model.nv 和 model.nu 分别是广义位置、广义速度和执行器的维度。它们不一定相等，不能凭经验 reshape。

## 4. 仿真循环

保存为 simulate.py：

~~~python
from pathlib import Path

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(str(Path("model.xml")))
data = mujoco.MjData(model)

physics_dt = model.opt.timestep
policy_dt = 0.02
steps_per_action = round(policy_dt / physics_dt)
if steps_per_action < 1 or not np.isclose(
    steps_per_action * physics_dt, policy_dt
):
    raise ValueError("policy_dt must be a multiple of physics_dt")

records = []
for _ in range(500):
    action = np.array([0.25], dtype=np.float64)
    data.ctrl[:] = action
    for _ in range(steps_per_action):
        mujoco.mj_step(model, data)
    records.append({
        "time": float(data.time),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
    })

np.savez("trajectory.npz",
         time=np.asarray([r["time"] for r in records]),
         qpos=np.stack([r["qpos"] for r in records]),
         qvel=np.stack([r["qvel"] for r in records]),
         ctrl=np.stack([r["ctrl"] for r in records]))
~~~

qpos 和 qvel 会被下一次 mj_step 原地更新，保存轨迹时必须 copy。data.ctrl 是实际施加给执行器的输入；如果中间还有控制器或限幅，也应同时记录控制器输出。

## 5. 读取状态、传感器和接触

~~~python
qpos = data.qpos.copy()       # 形状 [model.nq]
qvel = data.qvel.copy()       # 形状 [model.nv]
ctrl = data.ctrl.copy()       # 形状 [model.nu]
time_now = float(data.time)

joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "hinge"
)
qpos_addr = model.jnt_qposadr[joint_id]
hinge_angle = float(data.qpos[qpos_addr])
~~~

在 MJCF 的 sensor 节点中声明传感器，再从 data.sensordata 读取。接触数量可从 data.ncon 读取，具体接触对可检查 data.contact[i]。工程日志应保存传感器名称、维度、单位和时间戳。

## 6. 三种时间步

- physics step：引擎进行一次物理积分，长度是 physics_dt。
- policy step：策略产生一次新动作，长度是 policy_dt。
- render step：显示或保存一帧画面，频率可以和前两者不同。

常用动作保持步数为：

~~~text
steps_per_action = policy_dt / physics_dt
~~~

例如 physics_dt 为 0.002、policy_dt 为 0.02 时，一个动作保持 10 个物理步。policy_dt 可以按任务调整，但改变它会改变控制带宽、样本数量和结果可比性。

## 7. 接入 Gymnasium

先用官方环境确认 Python、动作空间和 reset 流程：

~~~python
import gymnasium as gym

env = gym.make("Ant-v5")
obs, info = env.reset(seed=0)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
~~~

一个 RL transition 可写成 (obs, action, reward, next_obs, terminated, truncated, info)：obs 和 next_obs 是前后观测，action 是策略动作，reward 是即时奖励，terminated 表示任务自然结束，truncated 表示达到时间或步数上限，info 保存碰撞和奖励分项等调试信息。

训练前先随机运行几十步，检查观测形状、动作范围、奖励和终止标记，再开始长时间训练。连续动作通常使用 PPO、SAC 或 TD3；DQN 只适合离散动作。PPO 使用新鲜 rollout，SAC/TD3 将 transition 放入 replay buffer。

## 8. 建议的自行实验

完成基础循环后，每次只改一个变量：

1. 改 gravity 或杆的密度，观察位置和速度变化。
2. 改执行器增益，比较控制误差和稳定时间。
3. 改 policy_dt，比较动作保持、轨迹和控制频率。
4. 加入一个传感器或接触几何，检查观测和碰撞是否符合预期。

这些实验跑通后，再自行扩展到机械臂、视觉、地图探索或更完整的 RL 任务。

## 9. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| XML 找不到 mesh 或 texture | 相对路径、meshdir 和工作目录 |
| viewer 打不开 | 显示环境；服务器改用无头循环 |
| 动作维度错误 | model.nu、执行器顺序和 action_space |
| 轨迹每行都一样 | qpos/qvel 是否调用 copy |
| 仿真发散或穿透 | 单位、质量、惯量、接触参数和 timestep |
| 训练与评测不一致 | reset、action repeat、随机化和终止条件 |
