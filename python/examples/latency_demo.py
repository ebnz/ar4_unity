"""Effect of actuation latency on a closed-loop controller, reproducibly.

A proportional controller on the TCP height commands joint_2 velocity-style increments.
With action_delay=k each command reaches the robot k physics steps late; the run is
deterministic, so you can compare delays one-to-one.

    python examples/latency_demo.py --launch --headless
"""
import argparse
import contextlib

import numpy as np

from ar4_sim import AR4Sim, SimProcess

GAIN = 4.0  # rad of joint_2 per m of height error, per second


def run(sim: AR4Sim, delay_steps: int) -> np.ndarray:
    sim.action_delay = 0
    q = np.radians([0, 20, -20, 0, 50, 0])
    sim.reset(q)
    sim.action_delay = delay_steps
    target_z = sim.get_state().tcp_pose.position[2] - 0.10   # 10 cm step down
    heights = []
    for _ in range(300):
        s = sim.step(1)
        err = s.tcp_pose.position[2] - target_z
        q[1] += GAIN * err * sim.dt * 3   # aggressive gain to make the delay visible
        sim.set_joint_targets(q)
        heights.append(err)
    return np.array(heights)


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
        print("delay   overshoot   residual error after 3 s")
        for delay in (0, 5, 10, 20):
            e = run(sim, delay)
            print(f"{delay * sim.dt * 1000:4.0f} ms   {max(0, -e.min()) * 1000:6.1f} mm   {abs(e[-1]) * 1000:6.2f} mm")


if __name__ == "__main__":
    main()
