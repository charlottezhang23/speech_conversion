"""Compare the transcript built from the original vs refined diarization turns.

Reports a speaker-agnostic error proxy: in a two-person interview the questions
(lines ending with ？) belong to the interviewer, so any 受访人 line ending with ？
or any 采访人-style answer swap shows up here.

Usage: python tools/compare_refine.py [project]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transcription import clean as C  # noqa: E402
from transcription import merge as M  # noqa: E402

PID = sys.argv[1] if len(sys.argv) > 1 else "20261103-133124-b6812e"
PDIR = ROOT / "projects" / PID


def load(name: Path):
    return json.loads(name.read_text(encoding="utf-8"))


def build(turns):
    segments = []
    for path in sorted((PDIR / "asr" / "chunks").glob("*.json")):
        segments.extend(load(path).get("segments") or [])
    cleaned, _stats = C.clean_segments(segments)
    pieces = M.assign_speakers(cleaned, [(t["start"], t["end"], t["speaker"]) for t in turns])
    return M.build_utterances(pieces)


def report(name: str, utts) -> list[str]:
    questions = [u for u in utts if str(u["text"]).strip().endswith(("？", "?"))]
    by_speaker = Counter(u["speaker"] for u in questions)
    lines = [
        f"{name}: 段数={len(utts)}  问句数={len(questions)}  问句按说话人={dict(by_speaker)}",
        f"  平均每段字数={sum(len(u['text']) for u in utts) / len(utts):.1f}",
        f"  12–36s：" ,
    ]
    for u in utts:
        if float(u["end"]) < 12 or float(u["start"]) > 36:
            continue
        lines.append(f"    [{u['start']:.2f}-{u['end']:.2f}] {u['speaker']}: {u['text']}")
    return lines


def main() -> int:
    before = load(PDIR / "diarization" / "raw.before_refine.json")["turns"]
    after = load(PDIR / "diarization" / "raw.json")["turns"]
    lines = report("原始分离", build(before)) + [""] + report("二次声纹校验后", build(after))
    out = ROOT / "logs" / "compare_refine.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
