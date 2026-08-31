#!/usr/bin/env python3
"""Collect synchronized hand-eye samples for eye-in-hand or eye-to-hand.

The CSV convention is documented in README.md:
  T_BE = T^B_E (end-effector frame expressed in robot base frame)
  T_CT = T^C_T (target frame expressed in camera frame)
"""

from pathlib import Path
import csv
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
import tf2_ros


def transform_to_matrix(transform) -> np.ndarray:
    q = transform.rotation
    t = transform.translation
    rotation = np.array(
        [q.x, q.y, q.z, q.w], dtype=np.float64
    )
    # scipy is intentionally avoided; OpenCV provides a stable quaternion path.
    x, y, z, w = rotation
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = r
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


class HandEyeSampler(Node):
    def __init__(self) -> None:
        super().__init__("hand_eye_sampler")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("target_frame", "aruco_board")
        self.declare_parameter("output_dir", "hand_eye_samples")
        self.declare_parameter("aruco_dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_length_m", 0.04)
        self.declare_parameter("target_marker_id", 0)
        self.declare_parameter("min_markers", 1)
        self.declare_parameter("mounting", "eye_in_hand")

        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.marker_length = float(self.get_parameter("marker_length_m").value)
        self.target_marker_id = int(self.get_parameter("target_marker_id").value)
        self.min_markers = int(self.get_parameter("min_markers").value)
        self.mounting = str(self.get_parameter("mounting").value)
        if self.mounting not in {"eye_in_hand", "eye_to_hand"}:
            raise ValueError("mounting must be eye_in_hand or eye_to_hand")
        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "samples.csv"
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.latest: Optional[dict] = None

        dictionary_name = str(self.get_parameter("aruco_dictionary").value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, cv2.aruco.DICT_4X4_50)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary) if hasattr(cv2.aruco, "ArucoDetector") else None

        self.image_sub = Subscriber(self, Image, self.get_parameter("image_topic").value)
        self.info_sub = Subscriber(self, CameraInfo, self.get_parameter("camera_info_topic").value)
        self.sync = ApproximateTimeSynchronizer([self.image_sub, self.info_sub], queue_size=20, slop=0.05)
        self.sync.registerCallback(self.synced_callback)
        self.create_service(Trigger, "save_hand_eye_sample", self.save_sample)
        self.get_logger().info(f"Waiting for {self.mounting} frames; save service: /save_hand_eye_sample")

    def detect_target(self, image: np.ndarray, info: CameraInfo):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
        if ids is None or len(ids) < self.min_markers:
            return None
        camera_matrix = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        dist = np.asarray(info.d, dtype=np.float64)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, camera_matrix, dist)
        matches = np.flatnonzero(ids.reshape(-1) == self.target_marker_id)
        if len(matches) == 0:
            return None
        marker_index = int(matches[0])
        rvec, tvec = rvecs[marker_index].reshape(3), tvecs[marker_index].reshape(3)
        rotation, _ = cv2.Rodrigues(rvec)
        target_in_camera = np.eye(4, dtype=np.float64)
        target_in_camera[:3, :3] = rotation
        target_in_camera[:3, 3] = tvec
        half = self.marker_length / 2.0
        object_corners = np.array(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(object_corners, rvec, tvec, camera_matrix, dist)
        observed = corners[marker_index].reshape(-1, 2)
        reprojection_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - observed, axis=1)))
        return target_in_camera, reprojection_error, int(len(ids)), corners, ids

    def synced_callback(self, image_msg: Image, info_msg: CameraInfo) -> None:
        try:
            if info_msg.header.frame_id and info_msg.header.frame_id != self.camera_frame:
                self.get_logger().warning(
                    f"CameraInfo frame_id={info_msg.header.frame_id!r} does not match camera_frame={self.camera_frame!r}"
                )
                return
            image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            detected = self.detect_target(image, info_msg)
            if detected is None:
                return
            target_in_camera, reproj, marker_count, corners, ids = detected
            stamp = image_msg.header.stamp
            image_time = rclpy.time.Time.from_msg(stamp)
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, image_time,
                timeout=rclpy.duration.Duration(seconds=0.15),
            )
            ee_in_base = transform_to_matrix(transform.transform)
            self.latest = {
                "stamp_image": stamp.sec + stamp.nanosec * 1e-9,
                "stamp_camera_info": info_msg.header.stamp.sec + info_msg.header.stamp.nanosec * 1e-9,
                "stamp_tf": transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9,
                "T_BE": ee_in_base,
                "T_CT": target_in_camera,
                "camera_k": np.asarray(info_msg.k, dtype=np.float64).reshape(3, 3),
                "distortion": np.asarray(info_msg.d, dtype=np.float64).reshape(-1),
                "reprojection_error_px": reproj,
                "num_markers": marker_count,
                "image": image.copy(),
                "corners": corners,
                "ids": ids,
            }
        except (RuntimeError, tf2_ros.TransformException) as exc:
            self.get_logger().debug(f"Skipping frame: {exc}")

    def save_sample(self, request: Trigger.Request, response: Trigger.Response):
        del request
        if self.latest is None:
            response.success = False
            response.message = "No valid synchronized detection and TF yet."
            return response
        sample = self.latest
        sample_id = self.next_sample_id()
        image_path = self.output_dir / f"image_{sample_id:04d}.png"
        fields = ["sample_id", "mounting", "stamp_image", "stamp_camera_info", "stamp_tf", "base_frame", "ee_frame", "camera_frame", "target_frame", "image_path", "reprojection_error_px", "num_markers"]
        fields += [f"camera_k_{i}{j}" for i in range(3) for j in range(3)]
        fields += ["distortion_coeffs"]
        fields += [f"T_BE_{i}{j}" for i in range(4) for j in range(4)]
        fields += [f"T_CT_{i}{j}" for i in range(4) for j in range(4)]
        cv2.imwrite(str(image_path), sample["image"])
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fields).writeheader()
        row = {
            "sample_id": sample_id,
            "mounting": self.mounting,
            "stamp_image": f"{sample['stamp_image']:.9f}",
            "stamp_camera_info": f"{sample['stamp_camera_info']:.9f}",
            "stamp_tf": f"{sample['stamp_tf']:.9f}",
            "base_frame": self.base_frame,
            "ee_frame": self.ee_frame,
            "camera_frame": self.camera_frame,
            "target_frame": self.target_frame,
            "image_path": str(image_path.name),
            "reprojection_error_px": f"{sample['reprojection_error_px']:.6f}",
            "num_markers": sample["num_markers"],
            "distortion_coeffs": ";".join(f"{value:.12g}" for value in sample["distortion"]),
        }
        row.update({f"camera_k_{i}{j}": f"{sample['camera_k'][i, j]:.12g}" for i in range(3) for j in range(3)})
        row.update({f"T_BE_{i}{j}": f"{sample['T_BE'][i, j]:.12g}" for i in range(4) for j in range(4)})
        row.update({f"T_CT_{i}{j}": f"{sample['T_CT'][i, j]:.12g}" for i in range(4) for j in range(4)})
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        response.success = True
        response.message = f"Saved sample {sample_id} to {self.csv_path}"
        self.get_logger().info(response.message)
        return response

    def next_sample_id(self) -> int:
        if not self.csv_path.exists():
            return 0
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            return sum(1 for _ in handle) - 1


def main() -> None:
    rclpy.init()
    node = HandEyeSampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
