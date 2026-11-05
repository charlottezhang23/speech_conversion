"""Start the web server as a detached process (survives terminal/task teardown)."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "server.log"


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    if port_open(port):
        print(f"already running on {port}")
        return 0

    env = dict(os.environ)
    env["PATH"] = str(ROOT / "tools" / "ffmpeg" / "bin") + os.pathsep + env.get("PATH", "")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    out = LOG.open("ab")
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(
        [str(PY), "-m", "app.main", "--no-browser", "--port", str(port)],
        cwd=str(ROOT),
        env=env,
        stdout=out,
        stderr=out,
        stdin=subprocess.DEVNULL,
        creationflags=flags,
    )
    for _ in range(60):
        time.sleep(0.5)
        if port_open(port):
            print(f"server started: http://127.0.0.1:{port} (log: logs/server.log)")
            return 0
    print("server did not start; check logs/server.log")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
