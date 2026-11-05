"""Real pyannote smoke test on a short clip (needs HF token + downloaded model).

Usage: python tools/diarize_probe.py [audio] [seconds] [num_speakers]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audio as A  # noqa: E402
from app.config import ensure_ffmpeg_on_path, get_hf_token  # noqa: E402
from transcription.diarization import DiarizationEngine  # noqa: E402


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "fixtures" / "audio" / "zh_interview_2speakers.wav"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
    num_speakers = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    ensure_ffmpeg_on_path()
    if not get_hf_token():
        print("没有 HF Token，无法测试")
        return 1

    clip = ROOT / "tmp" / "diar_probe.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    A.normalize(src, clip)
    info = A.probe(clip)
    if info.duration_sec and info.duration_sec > seconds:
        short = ROOT / "tmp" / "diar_probe_short.wav"
        A.slice_wav(clip, 0.0, seconds, short)
        clip = short
        info = A.probe(clip)

    t0 = time.perf_counter()
    engine = DiarizationEngine()
    t_load = time.perf_counter() - t0

    t1 = time.perf_counter()
    turns = engine.diarize(clip, num_speakers=num_speakers, progress=lambda p, m: None)
    t_run = time.perf_counter() - t1

    speakers = sorted({t["speaker"] for t in turns})
    lines = [
        f"音频：{src.name}（取前 {seconds:.0f}s）",
        f"模型加载：{t_load:.1f}s；分离：{t_run:.1f}s（音频 {info.duration_sec}）",
        f"turns={len(turns)} speakers={speakers}",
    ]
    for t in turns[:20]:
        lines.append(f"  {t['start']:7.2f} - {t['end']:7.2f}  {t['speaker']}")
    report = ROOT / "logs" / "diarize_probe.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
