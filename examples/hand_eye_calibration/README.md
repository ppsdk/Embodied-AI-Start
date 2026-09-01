# 手眼标定代码示例（ROS 2 Humble）

这个目录给出一个 eye-in-hand 的基础闭环：ROS 2 节点同步图像和 `CameraInfo`，按图像时间戳查询末端 TF，检测 ArUco，并把配对样本写入 CSV；离线脚本读取 CSV，调用 OpenCV `calibrateHandEye` 求出相机到末端的 `T^E_C`。

## 坐标和 CSV 约定

沿用 [`docs/robotics.md`](../../docs/robotics.md) 的记号：`T^A_B` 表示坐标系 B 在坐标系 A 中的位姿。CSV 中：

- `T_BE` 是 `T^B_E`，即 `base_frame <- ee_frame`；
- `T_CT` 是 `T^C_T`，即 `camera_frame <- target_frame`；
- `mounting` 记录 `eye_in_hand` 或 `eye_to_hand`，避免把两种安装关系的 CSV 混用；
- `camera_k_00` ... `camera_k_22` 和 `distortion_coeffs` 保存采样时的内参、畸变；`stamp_image`、`stamp_camera_info`、`stamp_tf` 用来检查时间配对；
- OpenCV 输入名 `gripper2base`、`target2cam` 分别对应这两个矩阵；输出 `cam2gripper` 就是 `T^E_C`。

## 1. 安装依赖

在 ROS 2 Humble 环境中：

```bash
sudo apt install ros-humble-cv-bridge ros-humble-message-filters ros-humble-tf2-ros python3-opencv
```

## 2. 采样

```bash
source /opt/ros/humble/setup.bash
python3 examples/hand_eye_calibration/collect_samples.py \
  --ros-args -p base_frame:=base_link -p ee_frame:=tool0 \
  -p camera_frame:=camera_color_optical_frame \
  -p output_dir:=/tmp/hand_eye_samples
```

节点检测到有效 ArUco 后，机器人停稳时触发保存：

```bash
ros2 service call /save_hand_eye_sample std_srvs/srv/Trigger {}
```

重复采集 15–30 个不同位置和姿态。每次保存还会写一张 `image_XXXX.png`，方便人工排查误检。示例默认只使用固定 ID（`target_marker_id:=0`）的 marker 作为 `target_frame`，这样每条样本的目标坐标系一致；使用棋盘格或多 marker 板时，应把检测器替换为整个板的统一坐标系。

相机固定、标定板装在末端时，将采样节点改为：

```bash
python3 examples/hand_eye_calibration/collect_samples.py \
  --ros-args -p mounting:=eye_to_hand -p output_dir:=/tmp/eye_to_hand_samples
```

## 3. 离线求解和验证

```bash
python3 examples/hand_eye_calibration/solve_hand_eye.py \
  /tmp/hand_eye_samples/samples.csv \
  --method DANIILIDIS
```

输出：

- `T_EC.npy`：4x4 的 `T^E_C`；
- `T_EC.yaml`：矩阵、求解方法和残差统计；
- `residuals.csv`：每个样本重建 `T^B_T = T^B_E T^E_C T^C_T` 相对第一条样本的平移/旋转残差。

`DANIILIDIS`、`TSAI`、`PARK` 都可尝试，但更换算法不能替代检查内参、板尺寸、TF 方向和时间同步。脚本只负责离线数值求解；它没有在当前 Windows 环境中实际连接 ROS 2 硬件运行。

## 4. Eye-to-hand 求解

```bash
python3 examples/hand_eye_calibration/solve_eye_to_hand.py \
  /tmp/eye_to_hand_samples/samples.csv
```

此脚本读取相同的 `T^B_E`、`T^C_T`，求出：

- `T_BC.npy/yaml`：固定相机在基座中的位姿 `T^B_C`；
- `T_ET.npy/yaml`：标定板在末端中的位姿 `T^E_T`；
- `residuals_eye_to_hand.csv`：由 `T^C_T_hat = (T^B_C)^-1 T^B_E T^E_T` 重建的观测残差。
