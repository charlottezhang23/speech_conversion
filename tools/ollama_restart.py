"""Restart the local Ollama service from Python (reliable on this machine)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"


def kill(name: str) -> None:
    subprocess.run(["taskkill", "/F", "/IM", name, "/T"], capture_output=True)


def main() -> int:
    kill("llama-server.exe")
    kill("ollama.exe")
    time.sleep(2.0)
    if not OLLAMA_EXE.exists():
        print("ollama.exe not found:", OLLAMA_EXE)
        return 1
    flags = 0
    if sys.platform == "win32":
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen([str(OLLAMA_EXE), "serve"], creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import requests

    for _ in range(40):
        time.sleep(0.5)
        try:
            if requests.get("http://127.0.0.1:11434/api/version", timeout=2).ok:
                print("ollama restarted")
                return 0
        except requests.RequestException:
            pass
    print("ollama did not come back")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
