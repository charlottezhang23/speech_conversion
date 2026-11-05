"""Kill whatever process is listening on a TCP port (used to restart the dev server)."""
from __future__ import annotations

import re
import subprocess
import sys


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, errors="replace").stdout
    pids: set[str] = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line.upper():
            parts = re.split(r"\s+", line.strip())
            if parts and parts[-1].isdigit():
                pids.add(parts[-1])
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
        print(f"killed pid {pid} on port {port}")
    if not pids:
        print(f"nothing listening on {port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
