"""Basic usage: blocking moves, then your own fixed-rate control loop (lockstep).

    python examples/joint_sweep.py --launch          # opens a window
"""
import argparse
import contextlib

import numpy as np

from ar4_sim import AR4Sim, SimProcess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--port", type=int, default=5005)
    args = ap.parse_args()

    with contextlib.ExitStack() as stack:
        if args.launch:
            stack.enter_context(SimProcess(port=args.port, headless=args.headless))
        sim = stack.enter_context(AR4Sim(port=args.port))
        sim.reset()

        # 1) Blocking moves: run simulated time until the goal is reached.
        for q in ([30, 20, -20, 0, 40, 0], [-30, 20, -20, 0, 40, 90], [0, 0, 0, 0, 0, 0]):
            s = sim.move_joints_deg(q, speed=0.8)
            print(f"t={s.time:5.2f}s  q={np.degrees(s.q).round(1)}  tcp={s.tcp_pose.position.round(3)}")

        # 2) Your own control loop: every iteration is exactly one physics step (dt),
        #    no matter how long your code takes. Here: a 0.25 Hz sine on joints 1 and 5
        #    (peak 35 deg/s, below the AR4's 60 deg/s limit).
        def trajectory(t):
            return np.radians([30 * np.sin(np.pi / 2 * t), 15, -15, 0, 30 + 20 * np.sin(np.pi / 2 * t), 0])

        sim.move_joints(trajectory(0.0))
        t0 = sim.get_state().time
        for k in range(400):
            target = trajectory(k * sim.dt)
            sim.set_joint_targets(target)
            s = sim.step(1)
            if k % 50 == 0:
                err = np.degrees(np.abs(s.q - target)).max()
                print(f"t={s.time - t0:4.2f}s  tracking error {err:.2f} deg")


if __name__ == "__main__":
    main()
