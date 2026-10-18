# AR4 Unity Simulation

A small Unity simulation of the Annin Robotics AR4 arm (with gripper), controlled from Python
over a plain TCP/JSON bridge. No ROS. Physics runs in **lockstep**: simulated time only advances
when Python calls `step()`, so runs are deterministic and independent of Python's speed.

```
Python (ar4_sim)  ── TCP 127.0.0.1:5005, one JSON object per line ──▶  Unity player / editor
  AR4Sim client                                                         BridgeServer.cs
  ikpy kinematics (same URDF)  ◀────────────────────────────────────── AR4Controller.cs
                                                                        ArticulationBody chain
```

Robot model: `annin_ar4_description` from [ycheng517/ar4_ros_driver](https://github.com/ycheng517/ar4_ros_driver)
(cloned at `4b88db1`), model **mk3**, gripper included.

## Quick start

```bash
# Python client (numpy + ikpy)
python3 -m venv python/.venv
python/.venv/bin/pip install -e "python[dev]"

# run an example against the prebuilt player (opens a window; --headless for none)
python/.venv/bin/python python/examples/pick_and_place.py --launch
```

```python
import numpy as np
from ar4_sim import AR4Sim, SimProcess

with SimProcess(headless=True), AR4Sim() as sim:   # or start the player / Unity Play mode yourself
    sim.reset()
    sim.move_joints_deg([30, 20, -20, 0, 40, 0])         # blocks in *simulated* time
    sim.move_pose([0.0, -0.3, 0.1], rpy=(np.pi, 0, 0))   # TCP pose via IK, tool pointing down
    sim.open_gripper(); sim.close_gripper()

    for k in range(500):                                 # your own control loop: 1 iteration = 1 dt
        s = sim.step()
        sim.set_joint_targets(my_controller(s))
```

Conventions: radians, meters, xyzw quaternions, all in the robot's `base_link` frame (ROS
convention; the robot faces −Y). The TCP is 55 mm past the flange (`ee_link`), between the finger pads.

## Lockstep and latency

Nothing moves between calls. `step(n)` advances n × `dt` (default 10 ms, `-dt` on the command line).
`step_until()` / `move_*()` run steps server-side until the goal is reached, which keeps them fast
(about 20–50× realtime headless).

To study latency reproducibly, `AR4Sim(action_delay=k)` makes every command take effect k steps
after it is issued. See `examples/latency_demo.py`.

## Python API (`ar4_sim.AR4Sim`)

| method | |
|---|---|
| `reset(q=None, gripper=None, reset_objects=True)` | teleport robot (default home, gripper open) and cubes |
| `step(n=1)` → `State` | advance physics n steps |
| `set_joint_targets(q, speed=1.0)` | joint goal; motion is velocity-limited (URDF: 60°/s × speed) |
| `set_gripper(position)` | jaw opening per side, 0 (closed) … 0.014 m |
| `step_until(max_steps, tol)` → `(reached, State)` | step until arm + gripper reached their goals |
| `move_joints(q)`, `move_joints_deg(q)`, `move_pose(pos, orientation/rpy)`, `move_linear(...)` | blocking helpers; raise `TimeoutError` if blocked (e.g. collision) |
| `open_gripper()`, `close_gripper()` | |
| `get_state()` → `State` | `q, dq, effort, goal, at_goal, gripper, flange_pose, tcp_pose, step, time` |
| `get_objects()`, `set_object_pose(name, pos, quat)` | the three 2 cm cubes |
| `get_image(width, height, eye=None, target=None)` → PNG bytes | needs a graphics device (not `headless=True`; use `SimProcess(offscreen=True)`) |
| `kinematics.fk(q)`, `kinematics.ik(pos, quat, q0)`, `solve_ik(...)` | ikpy on the same URDF |
| `call(cmd, **args)` | raw protocol access |

## Protocol

Request `{"id": 7, "cmd": "<name>", ...args}\n`, reply `{"id": 7, "ok": true, ...}\n` or
`{"id": 7, "ok": false, "error": "..."}\n`. Commands:

`ping`, `get_info`, `get_state`, `set_joint_targets {q[6], speed}`, `set_gripper {position}`,
`step {n}`, `step_until {max_steps, tol, settle_steps}`, `reset {q?, gripper?, reset_objects?}`,
`get_objects`, `set_object_pose {name, position, orientation}`, `get_image {width, height, eye?, target?, fov?}`,
`set_params {arm_stiffness, arm_damping, arm_force_limit, gripper_stiffness, gripper_damping, gripper_force_limit, gripper_max_velocity}`,
`get_bodies` (diagnostics).

Any language that can open a socket can drive the sim, e.g. `echo '{"id":1,"cmd":"get_state"}' | nc -q1 localhost 5005`.

## Running the simulator

- **Built player:** `unity/Build/AR4Sim.x86_64 [-port 5005] [-dt 0.01]`. Add `-batchmode -nographics`
  for no window and no GPU, or only `-batchmode` for no window with rendering kept (`get_image` works).
  `SimProcess` does this for you.
- **Unity Editor (6000.0.41f1):** open `unity/AR4Sim`, open scene `Assets/Scenes/AR4.unity`, press Play.
  The bridge listens while in Play mode.

## Rebuilding

```bash
python/.venv/bin/python tools/prepare_urdf.py [--model mk1|mk2|mk3] [--no-gripper]   # xacro -> URDF + meshes
tools/build_unity.sh                     # import URDF, build scene and Linux player (headless, ~1 min)
```

`prepare_urdf.py` flattens the xacro without ROS (`pip install xacro`) and writes
`unity/AR4Sim/Assets/URDF/ar4/` plus `python/ar4_sim/data/ar4.urdf`. It also works around
URDF Importer issues:
- Collision meshes get their own names and folder, and are decimated below 20k triangles. The
  importer otherwise overwrites a split visual mesh's chunks with its convex hulls.
- Each jaw's collision is a box fitted to the finger, because VHACD loses the 5 mm fingers.

The scene builder (`Assets/Scripts/Editor/AR4SceneBuilder.cs`, also under the **AR4** menu) imports
the URDF with Unity's URDF Importer (ArticulationBodies), adds the ground, three cubes, camera and
bridge, and copies joint velocity limits from the URDF.

## Tests

```bash
cd python && env -u PYTHONPATH .venv/bin/python -m pytest -q
```

`env -u PYTHONPATH` keeps a sourced ROS install from injecting its pytest plugins. The end-to-end
tests start the headless player and check FK agreement between Unity and ikpy (< 0.1 mm),
determinism, action delay, joint limits, floor collision and a cube pick.

## Simulation notes

- Joints are stiff position drives (like the AR4's steppers), velocity-limited to the URDF limits.
  Gravity is on and the steady-state error is below 0.01°.
- Robot self-collisions are disabled; robot ↔ world collisions (floor, cubes) are on.
- Gripper: both jaws are driven together (Unity has no mimic joints) with a 5 N force limit per jaw.
  Forces above ~10 N let the jaws push into light objects (a PhysX contact limitation); tune with `set_params`.
- URDF links without `<inertial>` (`ee_link`) are made massless at runtime. Unity's default
  (1 kg, 1 kg·m²) made the wrist oscillate.
