from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .connection import Connection

DEFAULT_PLAYER = Path(__file__).resolve().parents[2] / "unity" / "Build" / "AR4Sim.x86_64"


class SimProcess:
    """Start the built Unity player and wait until its bridge answers.

    with SimProcess(headless=True) as sim_proc:
        sim = AR4Sim(port=sim_proc.port)
    """

    def __init__(self, player: str | Path | None = None, port: int = 5005, dt: float | None = None,
                 headless: bool = False, offscreen: bool = False, log: str | Path | None = None, startup_timeout: float = 60.0):
        player = Path(player or os.environ.get("AR4_SIM_PLAYER", DEFAULT_PLAYER))
        if not player.exists():
            raise FileNotFoundError(f"{player} not found; build it (see README) or set AR4_SIM_PLAYER")
        self.port = port
        args = [str(player), "-port", str(port), "-logFile", str(log) if log else "-"]
        if dt is not None:
            args += ["-dt", str(dt)]
        if headless:      # no window, no GPU: fastest, but get_image is unavailable
            args += ["-batchmode", "-nographics"]
        elif offscreen:   # no window, but renders (get_image works)
            args += ["-batchmode"]
        self._out = open(os.devnull, "w")  # with -logFile -, the log goes to stdout
        self.proc = subprocess.Popen(args, stdout=self._out, stderr=subprocess.STDOUT)
        try:
            Connection(port=port, connect_timeout=startup_timeout).close()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._out:
            self._out.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
