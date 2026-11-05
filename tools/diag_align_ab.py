"""A/B test: how much text lands on the wrong speaker without word-level alignment.

Current pipeline cuts a long ASR segment at diarization boundaries by *character
ratio* (no word timings). This script runs real forced alignment on one chunk,
derives the true character-level timeline, and measures how many characters the
current method attributes to the wrong speaker.

Usage: python tools/diag_align_ab.py <project> [chunk_index]
"""
from __future__ import annotations

import difflib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audio as A  # noqa: E402
from transcription import merge as M  # noqa: E402
from transcription.asr import AsrEngine  # noqa: E402

LOG = ROOT / "logs" / "align_ab.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def speaker_at(turns, t: float) -> str | None:
    for start, end, spk in turns:
        if start <= t <= end:
            return spk
    best = None
    for start, end, spk in turns:
        d = min(abs(t - start), abs(t - end))
        if d <= 1.0 and (best is None or d < best[0]):
            best = (d, spk)
    return best[1] if best else None


def char_times(text: str, words: list[dict]) -> dict[int, float]:
    """Map every character index of `text` to its (mid) time using aligned words."""
    joined = "".join(w.get("word", "") for w in words)
    jtimes: list[float] = []
    for w in words:
        n = len(w.get("word", ""))
        s, e = float(w["start"]), float(w["end"])
        for k in range(n):
            jtimes.append(s + (e - s) * (k + 0.5) / max(1, n))
    out: dict[int, float] = {}
    sm = difflib.SequenceMatcher(a=text, b=joined, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = jtimes[j1 + k]
        elif tag == "replace":
            span = jtimes[j1:j2] or [0.0]
            for k in range(i1, i2):
                out[k] = span[min(len(span) - 1, k - i1)]
    return out


def main() -> int:
    pid = sys.argv[1] if len(sys.argv) > 1 else "20261103-133124-b6812e"
    index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    LOG.write_text("", encoding="utf-8")

    pdir = ROOT / "projects" / pid
    man = load(pdir / "asr" / "manifest.json")
    chunk = man["chunks"][index]
    offset = float(chunk["start"])
    segs = load(pdir / "asr" / f"chunks" / f"{index:03d}.json")["segments"]
    raw = load(pdir / "diarization" / "raw.json")
    turns = sorted((float(t["start"]), float(t["end"]), t["speaker"]) for t in raw["turns"])

    log(f"项目 {pid} 分段 {index}：{offset:.1f}-{float(chunk['end']):.1f}s，{len(segs)} 段，"
        f"模型={man['settings'].get('model')} align={man['settings'].get('align')}")

    tmp = ROOT / "tmp" / "align_ab_chunk.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    A.slice_wav(pdir / "audio" / "normalized_16k.wav", offset, float(chunk["end"]), tmp)

    base = [{"start": float(s["start"]) - offset, "end": float(s["end"]) - offset, "text": s["text"]} for s in segs]
    engine = AsrEngine("medium", compute_type="int8", threads=8, language="zh")
    t0 = time.perf_counter()
    aligned = engine.align(base, tmp, "zh")
    log(f"对齐耗时 {time.perf_counter() - t0:.1f}s")
    aligned_segs = aligned.get("segments") or []
    engine.unload()
    if len(aligned_segs) != len(segs):
        log(f"注意：对齐返回 {len(aligned_segs)} 段，原始 {len(segs)} 段")

    total_chars = 0
    wrong_chars = 0
    segs_with_error = 0
    examples: list[str] = []
    for seg, al in zip(segs, aligned_segs):
        text = str(seg["text"])
        words = al.get("words") or []
        if not text.strip() or not words:
            continue
        times = char_times(text, words)

        # ground truth: which speaker actually said each character
        truth: dict[int, str] = {}
        for i, t in times.items():
            spk = speaker_at(turns, t + offset)
            if spk:
                truth[i] = spk

        # current pipeline: split at diarization boundaries, cut text by ratio
        pieces = M.assign_speakers([seg], turns)
        cuts = []
        acc = 0
        for piece in pieces:
            acc += len(str(piece["text"]).strip())
            cuts.append(acc)
        predicted: dict[int, str] = {}
        lo = 0
        for piece, cut in zip(pieces, cuts):
            for i in range(lo, min(cut, len(text))):
                predicted[i] = piece["speaker"]
            lo = cut

        bad = sum(1 for i, spk in predicted.items() if truth.get(i) and truth[i] != spk)
        total_chars += len(predicted)
        wrong_chars += bad
        if bad:
            segs_with_error += 1
            if len(examples) < 5:
                pos = sorted(i for i, spk in predicted.items() if truth.get(i) and truth[i] != spk)
                # show the first wrong stretch with one char of context
                a, b = pos[0], pos[-1]
                examples.append(
                    f"[{float(seg['start']):.1f}-{float(seg['end']):.1f}] 错标 {bad}/{len(predicted)} 字\n"
                    f"    当前: {text[:max(0, a - 10)]}|{text[max(0, a - 10):b + 1]}|{text[b + 1:b + 11]}\n"
                    f"    实说: {speaker_at(turns, times[a] + offset)} 说的那段 => {text[a:b + 1]}\n"
                    f"    分段数={len(pieces)} 切点(字)={cuts}"
                )

    pct = (wrong_chars / total_chars * 100) if total_chars else 0.0
    log(f"合计 {total_chars} 字，其中 {wrong_chars} 字（{pct:.1f}%）被归到了错误的说话人；"
        f"受影响分段 {segs_with_error}/{len(segs)}")
    for ex in examples:
        log(ex)
    log("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
