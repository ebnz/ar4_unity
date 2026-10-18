"""Forward / inverse kinematics for the AR4 from the same URDF Unity imports (via ikpy)."""
from __future__ import annotations

import io
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .types import Pose, quat_to_matrix

DEFAULT_URDF = Path(__file__).parent / "data" / "ar4.urdf"
TCP_OFFSET = 0.055  # along ee_link z, centre of the finger pads; matches AR4Controller.tcpOffset


class IKError(RuntimeError):
    pass


class AR4Kinematics:
    def __init__(self, urdf: str | Path = DEFAULT_URDF, tcp_offset: float = TCP_OFFSET):
        import ikpy.chain  # imported lazily so the client works without ikpy installed

        # ikpy follows a single chain; drop the gripper links so it ends at ee_link,
        # then add the TCP offset as the final fixed segment.
        root = ET.parse(urdf).getroot()
        for tag in ("joint", "link"):
            for el in list(root.findall(tag)):
                if el.get("name", "").startswith("gripper_"):
                    root.remove(el)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chain = ikpy.chain.Chain.from_urdf_file(
                io.StringIO(ET.tostring(root, encoding="unicode")),
                base_elements=["world"], last_link_vector=[0, 0, tcp_offset], name="ar4")
        mask = [l.joint_type == "revolute" for l in chain.links]
        self.chain = ikpy.chain.Chain(chain.links, active_links_mask=mask, name="ar4")
        self._active = np.flatnonzero(mask)
        assert len(self._active) == 6, [l.name for l in chain.links]
        self.lower = np.array([self.chain.links[i].bounds[0] for i in self._active])
        self.upper = np.array([self.chain.links[i].bounds[1] for i in self._active])

    def _full(self, q) -> np.ndarray:
        full = np.zeros(len(self.chain.links))
        full[self._active] = q
        return full

    def fk(self, q) -> Pose:
        """TCP pose in base_link for joint angles q [rad]."""
        return Pose.from_matrix(self.chain.forward_kinematics(self._full(q)))

    def ik(self, position, orientation=None, q0=None, pos_tol=1e-3, rot_tol=1e-2, restarts=8, seed=0) -> np.ndarray:
        """Joint angles reaching `position` (and xyzw `orientation`, if given) with the TCP.

        Starts from q0 (e.g. the current joints) so consecutive solutions stay on the same
        branch; falls back to random restarts. Raises IKError if nothing is within tolerance.
        """
        target = np.eye(4)
        target[:3, 3] = position
        if orientation is not None:
            target[:3, :3] = quat_to_matrix(orientation)
        mode = "all" if orientation is not None else None
        rng = np.random.default_rng(seed)
        q0 = np.zeros(6) if q0 is None else np.clip(q0, self.lower, self.upper)

        best, best_err = None, np.inf
        for attempt in range(restarts + 1):
            start = q0 if attempt == 0 else rng.uniform(self.lower, self.upper)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = self.chain.inverse_kinematics_frame(target, initial_position=self._full(start), orientation_mode=mode)
            q = np.clip(sol[self._active], self.lower, self.upper)
            pe, re = self._error(q, target, orientation is not None)
            if pe <= pos_tol and re <= rot_tol:
                return q
            err = pe + 0.1 * re
            if err < best_err:
                best, best_err = q, err
        pe, re = self._error(best, target, orientation is not None)
        raise IKError(f"no IK solution: best position error {pe * 1000:.1f} mm, rotation error {np.degrees(re):.1f} deg")

    def _error(self, q, target, with_rot):
        T = self.chain.forward_kinematics(self._full(q))
        pe = float(np.linalg.norm(T[:3, 3] - target[:3, 3]))
        if not with_rot:
            return pe, 0.0
        R = T[:3, :3].T @ target[:3, :3]
        re = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return pe, re
