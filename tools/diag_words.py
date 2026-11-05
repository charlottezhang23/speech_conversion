"""Print word-level timings with the diarization label at each word (diagnostics)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    pid = sys.argv[1]
    lo, hi = (float(x) for x in sys.argv[2].split("-"))
    pdir = ROOT / "projects" / pid
    turns = []
    raw = json.loads((pdir / "diarization" / "raw.json").read_text(encoding="utf-8"))
    for t in raw["turns"]:
        turns.append((float(t["start"]), float(t["end"]), t["speaker"]))
    turns.sort()

    def label_at(t: float) -> str:
        hits = [spk for s, e, spk in turns if s <= t <= e]
        if hits:
            return "/".join(sorted(set(hits)))
        return "—"

    lines = [f"turns {lo:.1f}-{hi:.1f}s:"]
    for s, e, spk in turns:
        if e < lo or s > hi:
            continue
        lines.append(f"  {s:8.2f}-{e:8.2f} ({e - s:5.2f}s) {spk}")

    lines.append("")
    lines.append("words (word start-end -> diarization label at midpoint):")
    for path in sorted((pdir / "asr" / "chunks").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seg in data.get("segments") or []:
            for w in seg.get("words") or []:
                ws, we = float(w["start"]), float(w["end"])
                if we < lo or ws > hi:
                    continue
                mid = (ws + we) / 2
                lines.append(f"  {ws:8.2f}-{we:8.2f} {label_at(mid):6s} {w.get('word')!r} score={w.get('score')}")

    out = ROOT / "logs" / "diag_words.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
