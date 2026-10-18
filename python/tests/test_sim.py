"""End-to-end tests against the built Unity player (skipped if it hasn't been built)."""
import numpy as np
import pytest

from ar4_sim import AR4Sim, BridgeError, SimProcess
from ar4_sim.launch import DEFAULT_PLAYER

pytestmark = pytest.mark.skipif(not DEFAULT_PLAYER.exists(), reason="Unity player not built")

PORT = 5099
DOWN = (np.pi, 0.0, 0.0)


@pytest.fixture(scope="module")
def sim():
    with SimProcess(headless=True, port=PORT), AR4Sim(port=PORT) as s:
        yield s


@pytest.fixture(autouse=True)
def fresh(sim):
    sim.action_delay = 0
    sim.reset()


def test_info(sim):
    assert sim.joint_names == [f"joint_{i}" for i in range(1, 7)]
    assert sim.dt == pytest.approx(0.01)
    np.testing.assert_allclose(sim.max_velocity, 1.0472, rtol=1e-4)
    assert sim.has_gripper and sim.gripper_range == pytest.approx((0.0, 0.014))


def test_time_only_advances_on_step(sim):
    t0 = sim.get_state().time
    sim.set_joint_targets([0.5, 0, 0, 0, 0, 0])
    assert sim.get_state().time == t0
    assert sim.get_state().q[0] == pytest.approx(0, abs=1e-4)
    s = sim.step(10)
    assert s.step == 10 and s.time == pytest.approx(0.1)
    # velocity-limited: 10 steps at 1.0472 rad/s
    assert s.q[0] == pytest.approx(0.10472, abs=0.01)


def test_unity_pose_matches_python_fk(sim):
    rng = np.random.default_rng(0)
    for _ in range(3):
        q = rng.uniform(sim.lower * 0.6, sim.upper * 0.6)
        s = sim.move_joints(q, speed=1.0)
        fk = sim.kinematics.fk(s.q)
        np.testing.assert_allclose(s.tcp_pose.position, fk.position, atol=1e-4)
        assert abs(np.dot(s.tcp_pose.orientation, fk.orientation)) == pytest.approx(1, abs=1e-5)


def test_move_pose_reaches_target(sim):
    target = np.array([0.05, -0.30, 0.15])
    s = sim.move_pose(target, rpy=DOWN)
    np.testing.assert_allclose(s.tcp_pose.position, target, atol=2e-3)


def test_joint_limits_are_clamped(sim):
    s = sim.move_joints([5.0, 0, 0, 0, 0, 0], speed=1.0)
    assert s.q[0] == pytest.approx(sim.upper[0], abs=0.003)


def test_arm_collides_with_floor(sim):
    # joint_2 at +90 deg would put the forearm through the ground: the move must stall.
    with pytest.raises(TimeoutError):
        sim.move_joints([0, np.pi / 2, 0, 0, 0, 0], speed=1.0, timeout=5.0)


def test_deterministic(sim):
    def run():
        sim.reset()
        sim.set_joint_targets([0.4, 0.3, -0.2, 0.5, 0.6, -0.4])
        sim.set_gripper(0.0)
        return sim.step(150)

    a, b = run(), run()
    np.testing.assert_array_equal(a.q, b.q)
    assert a.gripper == b.gripper


def test_action_delay(sim):
    sim.action_delay = 20
    sim.set_joint_targets([0.5, 0, 0, 0, 0, 0])
    s = sim.step(20)
    assert abs(s.q[0]) < 1e-3          # command not applied yet
    s = sim.step(10)
    assert s.q[0] > 0.05               # now moving
    s = sim.move_joints([0, 0, 0, 0, 0, 0])
    assert s.at_goal


def test_errors_are_reported(sim):
    with pytest.raises(BridgeError, match="unknown cmd"):
        sim.call("does_not_exist")
    with pytest.raises(BridgeError, match="6 elements"):
        sim.call("set_joint_targets", q=[0, 0])
    sim.get_state()  # connection still usable


def test_pick_up_cube(sim):
    cube = sim.get_objects()["cube_red"].position
    sim.open_gripper()
    sim.move_pose(cube + [0, 0, 0.08], rpy=DOWN, speed=1.0)
    sim.move_linear(cube + [0, 0, 0.002], rpy=DOWN, speed=0.1)
    s = sim.close_gripper()
    assert s.gripper == pytest.approx(0.0096, abs=5e-4)  # stopped by the 20 mm cube
    sim.move_linear(cube + [0, 0, 0.10], rpy=DOWN, speed=0.1)
    assert sim.get_objects()["cube_red"].position[2] > 0.09
