#!/usr/bin/env bash
# Regenerate the AR4 scene from the URDF and build the players, headless.
#   UNITY=/path/to/Editor/Unity tools/build_unity.sh [--skip-scene] [--windows] [--macos]
# Always builds the Linux player; --windows / --macos also build those (need the editor's
# "Windows Build Support (Mono)" / "Mac Build Support (Mono)" modules).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNITY="${UNITY:-$HOME/Unity/Hub/Editor/6000.3.28f1/Editor/Unity}"
PROJECT="$ROOT/unity/AR4Sim"
mkdir -p "$ROOT/unity/Logs"

run() {
  echo "==> $1"
  "$UNITY" -batchmode -nographics -quit -projectPath "$PROJECT" \
    -executeMethod "AR4Sim.EditorTools.AR4SceneBuilder.$1" -logFile "$ROOT/unity/Logs/$1.log" \
    || { grep -E "error CS|Exception" -A3 "$ROOT/unity/Logs/$1.log" | head -40; exit 1; }
}

SCENE=1; WINDOWS=0; MACOS=0
for arg in "$@"; do
  case "$arg" in
    --skip-scene) SCENE=0 ;;
    --windows) WINDOWS=1 ;;
    --macos) MACOS=1 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done
[[ $SCENE == 1 ]] && run BuildScene
run BuildPlayer
echo "player: $ROOT/unity/Build/Linux/AR4Sim.x86_64"
if [[ $WINDOWS == 1 ]]; then run BuildWindowsPlayer; echo "player: $ROOT/unity/Build/Windows/AR4Sim.exe"; fi
if [[ $MACOS == 1 ]]; then run BuildMacPlayer; echo "player: $ROOT/unity/Build/macOS/AR4Sim.app"; fi
