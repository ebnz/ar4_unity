#!/usr/bin/env bash
# Regenerate the AR4 scene from the URDF and build the Linux player, headless.
#   UNITY=/path/to/Editor/Unity tools/build_unity.sh [--skip-scene]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITY="${UNITY:-$HOME/Unity/Hub/Editor/6000.0.41f1/Editor/Unity}"
PROJECT="$ROOT/unity/AR4Sim"
mkdir -p "$ROOT/unity/Logs"

run() {
  echo "==> $1"
  "$UNITY" -batchmode -nographics -quit -projectPath "$PROJECT" \
    -executeMethod "AR4Sim.EditorTools.AR4SceneBuilder.$1" -logFile "$ROOT/unity/Logs/$1.log" \
    || { grep -E "error CS|Exception" -A3 "$ROOT/unity/Logs/$1.log" | head -40; exit 1; }
}

[[ "${1:-}" == "--skip-scene" ]] || run BuildScene
run BuildPlayer
echo "player: $ROOT/unity/Build/AR4Sim.x86_64"
