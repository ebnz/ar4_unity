#!/usr/bin/env python3
"""Flatten the AR4 xacro from ycheng517/ar4_ros_driver into a plain URDF.

No ROS install needed: `$(find annin_ar4_description)` is substituted textually
before xacro runs, and mesh references are rewritten to `package://meshes/...`
so the Unity URDF Importer resolves them relative to the URDF file.

Outputs (default):
  unity/AR4Sim/Assets/URDF/ar4/ar4.urdf      + meshes/   (for Unity)
  python/ar4_sim/data/ar4.urdf                           (for ikpy, no meshes)
"""
import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DESC = ROOT / "third_party/ar4_ros_driver/annin_ar4_description"
PKG = "annin_ar4_description"


def read_stl(path):
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    if data[:5] == b"solid" and len(data) != 84 + 50 * n:
        verts = [list(map(float, l.split()[1:4])) for l in data.decode().splitlines() if l.strip().startswith("vertex")]
        return np.asarray(verts, dtype=np.float32).reshape(-1, 3, 3)
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3)


def write_collision_stl(src, dst, max_tris=20000):
    """Copy src, decimating by vertex clustering if it would be split into several Unity meshes."""
    tris = read_stl(src)
    cell = 0.0005
    while len(tris) > max_tris:
        keys = np.round(tris.reshape(-1, 3) / cell).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1, 3)
        keep = (inv[:, 0] != inv[:, 1]) & (inv[:, 1] != inv[:, 2]) & (inv[:, 0] != inv[:, 2])
        snapped = (np.round(tris / cell) * cell).astype(np.float32)
        tris = tris if keep.all() else snapped[keep]
        cell *= 2
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["n"], rec["v"] = normals, tris.reshape(-1, 9)
    dst.write_bytes(b"\0" * 80 + struct.pack("<I", len(tris)) + rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mk3", choices=["mk1", "mk2", "mk3"])
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--unity-dir", default=ROOT / "unity/AR4Sim/Assets/URDF/ar4", type=Path)
    ap.add_argument("--python-urdf", default=ROOT / "python/ar4_sim/data/ar4.urdf", type=Path)
    args = ap.parse_args()

    if not DESC.is_dir():
        sys.exit(f"missing {DESC}; clone https://github.com/ycheng517/ar4_ros_driver into third_party/")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_desc = Path(tmp) / PKG
        shutil.copytree(DESC / "urdf", tmp_desc / "urdf")
        shutil.copytree(DESC / "config", tmp_desc / "config")
        for f in (tmp_desc / "urdf").glob("*.xacro"):
            f.write_text(f.read_text().replace(f"$(find {PKG})", str(tmp_desc)))

        xacro = Path(sys.executable).with_name("xacro")
        cmd = [str(xacro) if xacro.exists() else "xacro", str(tmp_desc / "urdf/ar.urdf.xacro"),
               f"ar_model:={args.model}", f"include_gripper:={'false' if args.no_gripper else 'true'}"]
        urdf = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

    # file:///tmp/.../annin_ar4_description/meshes/ar4_mk3/link_1.STL -> package://meshes/ar4_mk3/link_1.STL
    urdf = re.sub(r'filename="file://[^"]*?/meshes/', 'filename="package://meshes/', urdf)
    # The Unity importer rejects empty material names; give them one.
    urdf = urdf.replace('<material name="">', '<material name="ar4_yellow">')

    # VHACD hulls lose the 5 mm fingers, so give each jaw an exact box around its finger instead.
    for jaw in ("gripper_jaw1_link", "gripper_jaw2_link"):
        verts = read_stl(DESC / f"meshes/ar4_gripper/{jaw}.stl").reshape(-1, 3)
        finger = verts[verts[:, 1] < -0.01]  # below the slider block, in the jaw frame
        lo, hi = finger.min(0), finger.max(0)
        box = (f'<collision><origin xyz="{" ".join(f"{v:.5f}" for v in (lo + hi) / 2)}" rpy="0 0 0"/>'
               f'<geometry><box size="{" ".join(f"{v:.5f}" for v in hi - lo)}"/></geometry></collision>')
        urdf, n = re.subn(rf'(<link name="{jaw}">.*?)<collision>.*?</collision>', lambda m: m.group(1) + box,
                          urdf, count=1, flags=re.S)
        assert n == 1, jaw

    # Collision meshes get their own folder: the importer names VHACD hulls <mesh>_<i>.asset,
    # which collides with the chunk assets it creates for visual STLs > 21845 triangles.
    # Distinct base names too: the importer skips prefab creation for names already used as hull templates.
    urdf = re.sub(r'(<collision>.*?filename="package://meshes/)([^"]+?)(\.stl)"',
                  r'\1collision/\2_col\3"', urdf, flags=re.S | re.I)

    used = sorted(set(re.findall(r'package://meshes/([^"]+)', urdf)))
    if args.unity_dir.exists():
        shutil.rmtree(args.unity_dir)  # drop stale generated prefabs/assets
    args.unity_dir.mkdir(parents=True)
    (args.unity_dir / "ar4.urdf").write_text(urdf)
    for rel in used:
        dst = args.unity_dir / "meshes" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith("collision/"):
            src = re.sub(r"_col(\.stl)$", r"\1", rel[len("collision/"):], flags=re.I)
            write_collision_stl(DESC / "meshes" / src, dst)
        else:
            shutil.copy2(DESC / "meshes" / rel, dst)

    args.python_urdf.parent.mkdir(parents=True, exist_ok=True)
    args.python_urdf.write_text(urdf)
    print(f"model={args.model} gripper={not args.no_gripper}: {len(used)} meshes")
    print(f"  {args.unity_dir / 'ar4.urdf'}\n  {args.python_urdf}")


if __name__ == "__main__":
    main()
