"""Pick the red cube and put it down 15 cm to the side.

Start the simulator first (or pass --launch to start the built player):
    python examples/pick_and_place.py --launch
"""
import argparse
import contextlib

import numpy as np

from ar4_sim import AR4Sim, SimProcess

DOWN = (np.pi, 0.0, 0.0)  # rpy: tool z pointing at the floor, jaws closing along base x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="start unity/Build/AR4Sim.x86_64")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--port", type=int, default=5005)
    args = ap.parse_args()

    with contextlib.ExitStack() as stack:
        if args.launch:
            stack.enter_context(SimProcess(port=args.port, headless=args.headless))
        sim = stack.enter_context(AR4Sim(port=args.port))

        sim.reset()
        cube = sim.get_objects()["cube_red"].position
        place = cube + np.array([0.15, 0.03, 0.0])
        print(f"cube at {cube.round(3)}, placing at {place.round(3)}")

        sim.open_gripper()
        sim.move_pose(cube + [0, 0, 0.08], rpy=DOWN, speed=0.6)
        sim.move_linear(cube + [0, 0, 0.002], rpy=DOWN, speed=0.05)
        s = sim.close_gripper()
        print(f"gripper closed at {s.gripper * 1000:.1f} mm per jaw")
        sim.move_linear(cube + [0, 0, 0.10], rpy=DOWN, speed=0.08)

        lifted = sim.get_objects()["cube_red"].position
        print(f"cube lifted to z={lifted[2]:.3f} m" + ("" if lifted[2] > 0.05 else "  <-- grasp failed"))

        sim.move_pose(place + [0, 0, 0.10], rpy=DOWN, speed=0.6)
        sim.move_linear(place + [0, 0, 0.004], rpy=DOWN, speed=0.05)
        sim.open_gripper()
        sim.move_linear(place + [0, 0, 0.10], rpy=DOWN, speed=0.08)
        s = sim.move_joints(np.zeros(6), speed=0.6)

        final = sim.get_objects()["cube_red"].position
        print(f"cube now at {final.round(3)} (error {np.linalg.norm(final[:2] - place[:2]) * 1000:.1f} mm), "
              f"sim time {s.time:.1f} s")


if __name__ == "__main__":
    main()
