import numpy as np
import pytest

from ar4_sim.kinematics import AR4Kinematics, IKError
from ar4_sim.types import Pose, matrix_to_quat, quat_to_matrix, rpy_to_quat


@pytest.fixture(scope="module")
def kin():
    return AR4Kinematics()


def test_quaternion_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q *= np.sign(q[3])
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_rpy_down_points_tool_at_floor():
    R = quat_to_matrix(rpy_to_quat(np.pi, 0, 0))
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)


def test_ik_roundtrip(kin):
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(kin.lower * 0.7, kin.upper * 0.7)
        target = kin.fk(q)
        sol = kin.ik(target.position, target.orientation, q0=np.zeros(6))
        got = kin.fk(sol)
        np.testing.assert_allclose(got.position, target.position, atol=1e-3)
        assert abs(np.dot(got.orientation, target.orientation)) > 0.9999


def test_ik_unreachable(kin):
    with pytest.raises(IKError):
        kin.ik([2.0, 0, 0], restarts=2)


def test_pose_matrix_roundtrip():
    p = Pose(np.array([0.1, -0.2, 0.3]), rpy_to_quat(0.1, 0.2, 0.3))
    back = Pose.from_matrix(p.matrix())
    np.testing.assert_allclose(back.position, p.position)
    np.testing.assert_allclose(back.orientation, p.orientation, atol=1e-12)
