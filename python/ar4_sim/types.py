from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    """3x3 rotation matrix -> xyzw quaternion (w >= 0)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q if q[3] >= 0 else -q


def rpy_to_quat(roll, pitch, yaw) -> np.ndarray:
    """URDF convention (fixed axes X, then Y, then Z) -> xyzw."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


@dataclass
class Pose:
    """Position [m] and xyzw orientation, in the robot's base_link frame."""
    position: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))

    @classmethod
    def from_dict(cls, d: dict) -> "Pose":
        return cls(np.array(d["position"], dtype=float), np.array(d["orientation"], dtype=float))

    @classmethod
    def from_matrix(cls, T) -> "Pose":
        T = np.asarray(T)
        return cls(T[:3, 3].copy(), matrix_to_quat(T[:3, :3]))

    def matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = quat_to_matrix(self.orientation)
        T[:3, 3] = self.position
        return T


@dataclass
class State:
    step: int
    time: float
    q: np.ndarray            # joint positions [rad]
    dq: np.ndarray           # joint velocities [rad/s]
    effort: np.ndarray       # drive torques [Nm]
    goal: np.ndarray         # commanded joint goal [rad]
    at_goal: bool
    gripper: float           # jaw opening per side [m], 0 = closed
    gripper_goal: float
    flange_pose: Pose        # ee_link (tool mounting flange)
    tcp_pose: Pose           # between the finger pads

    @classmethod
    def from_dict(cls, d: dict) -> "State":
        return cls(
            step=d["step"], time=d["time"],
            q=np.array(d["q"]), dq=np.array(d["dq"]), effort=np.array(d["effort"]),
            goal=np.array(d["goal"]), at_goal=d["at_goal"],
            gripper=d["gripper"], gripper_goal=d["gripper_goal"],
            flange_pose=Pose.from_dict(d["flange_pose"]), tcp_pose=Pose.from_dict(d["tcp_pose"]),
        )
