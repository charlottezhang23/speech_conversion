"""Print diarization turns and ASR segments inside a time range (diagnostics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    pid = sys.argv[1]
    lo, hi = (float(x) for x in sys.argv[2].split("-"))
    pdir = ROOT / "projects" / pid
    raw = json.loads((pdir / "diarization" / "raw.json").read_text(encoding="utf-8"))
    lines = [f"diarization turns in {lo:.1f}-{hi:.1f}s:"]
    for t in raw["turns"]:
        s, e = float(t["start"]), float(t["end"])
        if e < lo or s > hi:
            continue
        lines.append(f"  {s:8.2f}-{e:8.2f} ({e - s:5.2f}s) {t['speaker']}")
    lines.append("")
    lines.append("asr segments in range:")
    for path in sorted((pdir / "asr" / "chunks").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for s in data.get("segments") or []:
            s0, s1 = float(s["start"]), float(s["end"])
            if s1 < lo or s0 > hi:
                continue
            lines.append(f"  {s0:8.2f}-{s1:8.2f} words={len(s.get('words') or [])} {s['text'][:40]}")
    out = ROOT / "logs" / "diag_turns.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
