from __future__ import annotations

import base64
import heapq
import itertools
from typing import Sequence

import numpy as np

from .connection import Connection
from .types import Pose, State, rpy_to_quat


class AR4Sim:
    """Lockstep client for the Unity AR4 simulator.

    Simulated time only advances inside step(), step_until() and the blocking move_* helpers.
    Units: radians, meters, xyzw quaternions, all in the robot's base_link frame.

    action_delay: emulate actuation latency. Commands (joint targets, gripper) take effect
    `action_delay` physics steps after they were issued, deterministically.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, *, timeout: float = 60.0,
                 connect_timeout: float = 10.0, action_delay: int = 0):
        self._conn = Connection(host, port, timeout=timeout, connect_timeout=connect_timeout)
        info = self._conn.call("get_info")
        self.dt: float = info["dt"]
        self.joint_names: list[str] = info["joint_names"]
        self.lower = np.array(info["lower"])
        self.upper = np.array(info["upper"])
        self.max_velocity = np.array(info["max_velocity"])
        self.has_gripper: bool = info["has_gripper"]
        self.gripper_range: tuple[float, float] = tuple(info["gripper_range"])
        self.tcp_offset: float = info["tcp_offset"]
        self.object_names: list[str] = info["objects"]
        self.action_delay = int(action_delay)
        self._pending: list = []          # heap of (due_step, seq, cmd, args)
        self._seq = itertools.count()
        self._step = self.get_state().step
        self._kin = None

    # -- lifecycle ---------------------------------------------------------
    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- raw protocol --------------------------------------------------------
    def call(self, cmd: str, **args) -> dict:
        """Send any bridge command and return the raw reply."""
        return self._conn.call(cmd, **args)

    def get_state(self) -> State:
        return State.from_dict(self._conn.call("get_state")["state"])

    def set_joint_targets(self, q: Sequence[float], speed: float = 1.0):
        """Set the joint goal [rad]. speed scales the URDF velocity limits (0..1]."""
        self._command("set_joint_targets", q=np.asarray(q, dtype=float), speed=float(speed))

    def set_gripper(self, position: float):
        """Jaw opening per side [m]: 0 = closed, gripper_range[1] = fully open."""
        self._command("set_gripper", position=float(position))

    def step(self, n: int = 1) -> State:
        """Advance the simulation by n physics steps of dt seconds."""
        state = None
        while n > 0 or state is None:
            chunk = n
            if self._pending:
                chunk = min(n, max(0, self._pending[0][0] - self._step))
            if chunk > 0 or state is None:
                state = State.from_dict(self._conn.call("step", n=chunk)["state"])
                self._step = state.step
                n -= chunk
            self._flush_due()
        return state

    def step_until(self, max_steps: int = 2000, tol: float = 0.002, settle_steps: int = 5) -> tuple[bool, State]:
        """Step until arm and gripper reached their goals (within tol rad) or max_steps passed."""
        start = self._step
        if self._pending:  # delayed commands must take effect first
            self.step(max(due for due, *_ in self._pending) - self._step)
        remaining = max(0, max_steps - (self._step - start))
        r = self._conn.call("step_until", max_steps=remaining, tol=tol, settle_steps=settle_steps)
        state = State.from_dict(r["state"])
        self._step = state.step
        return r["reached"], state

    def reset(self, q: Sequence[float] | None = None, gripper: float | None = None, reset_objects: bool = True) -> State:
        """Teleport the robot (default: all zeros, gripper open) and the objects back."""
        self._pending.clear()
        r = self._conn.call("reset", q=None if q is None else np.asarray(q, dtype=float),
                            gripper=gripper, reset_objects=reset_objects)
        state = State.from_dict(r["state"])
        self._step = state.step
        return state

    def get_objects(self) -> dict[str, Pose]:
        return {o["name"]: Pose.from_dict(o) for o in self._conn.call("get_objects")["objects"]}

    def set_object_pose(self, name: str, position: Sequence[float], orientation: Sequence[float] = (0, 0, 0, 1)):
        self._conn.call("set_object_pose", name=name, position=list(map(float, position)),
                        orientation=list(map(float, orientation)))

    def get_image(self, width: int = 640, height: int = 480, eye=None, target=None, fov: float | None = None) -> bytes:
        """PNG bytes from the scene camera, or from a temporary camera at `eye` looking at `target`
        (base_link coordinates). Needs a graphics device: don't start the player with -nographics."""
        r = self._conn.call("get_image", width=width, height=height, fov=fov,
                            eye=None if eye is None else list(map(float, eye)),
                            target=None if target is None else list(map(float, target)))
        return base64.b64decode(r["png"])

    # -- convenience -------------------------------------------------------
    def move_joints(self, q: Sequence[float], speed: float = 0.5, timeout: float = 20.0) -> State:
        """Move to q [rad] and block (in simulated time) until reached. Raises TimeoutError."""
        self.set_joint_targets(q, speed)
        reached, state = self.step_until(max_steps=int(timeout / self.dt))
        if not reached:
            err = np.abs(state.q - np.clip(q, self.lower, self.upper)).max()
            raise TimeoutError(f"joint goal not reached after {timeout}s (max error {np.degrees(err):.2f} deg)")
        return state

    def move_joints_deg(self, q_deg: Sequence[float], speed: float = 0.5, timeout: float = 20.0) -> State:
        return self.move_joints(np.radians(q_deg), speed, timeout)

    @property
    def kinematics(self):
        if self._kin is None:
            from .kinematics import AR4Kinematics
            self._kin = AR4Kinematics(tcp_offset=self.tcp_offset)
        return self._kin

    def solve_ik(self, position, orientation=None, rpy=None, q0=None) -> np.ndarray:
        if rpy is not None:
            orientation = rpy_to_quat(*rpy)
        return self.kinematics.ik(position, orientation, q0=self.get_state().q if q0 is None else q0)

    def move_pose(self, position, orientation=None, rpy=None, speed: float = 0.5, timeout: float = 20.0) -> State:
        """Move the TCP to position [m] with xyzw orientation (or rpy). Joint-space interpolation."""
        return self.move_joints(self.solve_ik(position, orientation, rpy), speed, timeout)

    def move_linear(self, position, orientation=None, rpy=None, speed: float = 0.05, waypoint_dist: float = 0.005) -> State:
        """Move the TCP along a straight line at roughly `speed` m/s (IK at each waypoint)."""
        if rpy is not None:
            orientation = rpy_to_quat(*rpy)
        state = self.get_state()
        start = state.tcp_pose.position
        orientation = state.tcp_pose.orientation if orientation is None else orientation
        n = max(1, int(np.ceil(np.linalg.norm(np.asarray(position) - start) / waypoint_dist)))
        q = state.q
        steps_per_wp = max(1, int(round(waypoint_dist / speed / self.dt)))
        for i in range(1, n + 1):
            p = start + (np.asarray(position) - start) * i / n
            q = self.kinematics.ik(p, orientation, q0=q)
            self.set_joint_targets(q, speed=1.0)
            self.step(steps_per_wp)
        return self.step_until()[1]

    def open_gripper(self, wait: bool = True) -> State | None:
        self.set_gripper(self.gripper_range[1])
        return self.step_until()[1] if wait else None

    def close_gripper(self, wait: bool = True, steps: int = 60) -> State | None:
        """Close until the jaws stop (on an object, the goal is never reached; we just wait)."""
        self.set_gripper(self.gripper_range[0])
        return self.step(steps) if wait else None

    # -- internals -----------------------------------------------------------
    def _command(self, cmd: str, **args):
        if self.action_delay <= 0:
            self._conn.call(cmd, **args)
        else:
            heapq.heappush(self._pending, (self._step + self.action_delay, next(self._seq), cmd, args))

    def _flush_due(self):
        while self._pending and self._pending[0][0] <= self._step:
            _, _, cmd, args = heapq.heappop(self._pending)
            self._conn.call(cmd, **args)
