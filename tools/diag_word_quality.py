"""Inspect whisperx.align signature and analyse word-timing quality of a project."""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import whisperx

    lines = ["whisperx.align signature:"]
    try:
        lines.append("  " + str(inspect.signature(whisperx.align)))
    except Exception as exc:  # pragma: no cover
        lines.append(f"  (unavailable: {exc})")

    pid = sys.argv[1] if len(sys.argv) > 1 else "20261103-133124-b6812e"
    pdir = ROOT / "projects" / pid
    words = []
    for path in sorted((pdir / "asr" / "chunks").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for seg in data.get("segments") or []:
            for w in seg.get("words") or []:
                words.append((float(w["start"]), float(w["end"]), str(w.get("word"))))

    durs = sorted(e - s for s, e, _ in words)
    if durs:
        n = len(durs)
        lines.append("")
        lines.append(f"词级时间戳数量={n}  时长中位数={durs[n // 2]:.3f}s  p95={durs[int(n * 0.95)]:.3f}s  最大={durs[-1]:.3f}s")
        long_words = [(e - s, s, t) for s, e, t in words if e - s > 0.8]
        lines.append(f"超过 0.8s 的字/词：{len(long_words)} 个（{len(long_words) / n:.1%}）")
        for d, s, t in sorted(long_words, reverse=True)[:12]:
            lines.append(f"  {s:8.2f}s 持续 {d:5.2f}s  {t!r}")

    out = ROOT / "logs" / "diag_word_quality.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
