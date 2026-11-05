"""Load a Whisper model, then hold it resident so external tools can measure memory.

Usage: python tools/bench_mem.py large-v3 --hold 40
Measures are taken by the caller (e.g. PowerShell Get-Process PeakWorkingSet64).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = args[0] if args else "large-v3"
    hold = 40
    if "--hold" in sys.argv:
        hold = int(sys.argv[sys.argv.index("--hold") + 1])

    from transcription.asr import AsrEngine

    t0 = time.perf_counter()
    engine = AsrEngine(model, compute_type="int8", threads=8, language="zh")
    print(f"READY model={model} load_sec={time.perf_counter() - t0:.1f}", flush=True)
    time.sleep(hold)
    engine.unload()
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
