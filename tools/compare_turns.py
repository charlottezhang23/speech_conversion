"""Compare two diarization turn lists (probe JSON vs project raw.json) over a window."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pid = sys.argv[1] if len(sys.argv) > 1 else "20261103-133124-b6812e"
probe_file = ROOT / "logs" / "diar_probe_pyannote_speaker-diarization-3.1.json"
lo, hi = (float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0-300").split("-"))

probe = json.loads(probe_file.read_text(encoding="utf-8"))
base = json.loads((ROOT / "projects" / pid / "diarization" / "raw.json").read_text(encoding="utf-8"))["turns"]

# probe times are relative to the slice, which started at `lo`
probe_norm = [
    (round(float(t["start"]) + lo, 2), round(float(t["end"]) + lo, 2), t["speaker"]) for t in probe
]
base_norm = [
    (round(float(t["start"]), 2), round(float(t["end"]), 2), t["speaker"])
    for t in base
    if float(t["end"]) > lo and float(t["start"]) < hi
]

print(f"probe turns={len(probe_norm)}  community-1 turns in window={len(base_norm)}")
same = probe_norm == base_norm
print(f"完全一致: {same}")
if not same:
    for a, b in zip(probe_norm, base_norm):
        mark = "  " if a == b else "!!"
        print(f"{mark} 3.1={a}  c1={b}")
    if len(probe_norm) != len(base_norm):
        print(f"数量不同：3.1={len(probe_norm)} c1={len(base_norm)}")
