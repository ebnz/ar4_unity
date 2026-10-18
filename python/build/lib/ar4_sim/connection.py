from __future__ import annotations

import json
import socket
import time


class BridgeError(RuntimeError):
    """The simulator rejected a request (bad arguments, unknown command, ...)."""


class Connection:
    """Newline-delimited JSON request/reply over TCP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, timeout: float = 30.0,
                 connect_timeout: float = 0.0):
        deadline = time.monotonic() + connect_timeout
        while True:
            try:
                self._sock = socket.create_connection((host, port), timeout=timeout)
                break
            except (ConnectionRefusedError, socket.timeout, OSError):
                if time.monotonic() >= deadline:
                    raise ConnectionError(f"no AR4 simulator listening on {host}:{port}") from None
                time.sleep(0.2)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._reader = self._sock.makefile("rb")
        self._next_id = 0

    def call(self, cmd: str, **args) -> dict:
        self._next_id += 1
        req = {"id": self._next_id, "cmd": cmd, **{k: v for k, v in args.items() if v is not None}}
        self._sock.sendall((json.dumps(req, default=_to_json) + "\n").encode())
        line = self._reader.readline()
        if not line:
            raise ConnectionError("simulator closed the connection")
        reply = json.loads(line)
        if reply.get("id") != self._next_id:
            raise ConnectionError(f"reply id mismatch: sent {self._next_id}, got {reply.get('id')}")
        if not reply.get("ok"):
            raise BridgeError(f"{cmd}: {reply.get('error')}")
        return reply

    def close(self):
        try:
            self._reader.close()
            self._sock.close()
        except OSError:
            pass


def _to_json(o):
    # numpy arrays / scalars
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")
