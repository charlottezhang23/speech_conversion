"""Run one diarization model on a slice of a project's audio and print its turns.

Usage: python tools/diar_probe.py --model pyannote/speaker-diarization-3.1 --start 0 --end 300
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import audio as A  # noqa: E402
from transcription.diarization import DiarizationEngine  # noqa: E402

LOG = ROOT / "logs" / "diar_probe.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="20261103-133124-b6812e")
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=300.0)
    parser.add_argument("--speakers", type=int, default=2)
    parser.add_argument("--show", default="0-300", help="打印该范围内的 turn")
    parser.add_argument("--instantiate", default="", help='传给 pipeline.instantiate 的 JSON，如 {"segmentation":{"min_duration_off":0.5}}')
    parser.add_argument("--tag", default="", help="输出文件名后缀")
    args = parser.parse_args()

    pdir = ROOT / "projects" / args.project
    src = pdir / "audio" / "normalized_16k.wav"
    tmp = ROOT / "tmp" / f"diar_probe_{int(args.start)}_{int(args.end)}.wav"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    A.slice_wav(src, args.start, args.end, tmp)

    lines: list[str] = []

    def say(msg: str = "") -> None:
        lines.append(msg)
        print(msg, flush=True)

    say(f"模型={args.model} 音频={args.start:.0f}-{args.end:.0f}s 说话人数={args.speakers}")
    t0 = time.perf_counter()
    engine = DiarizationEngine(args.model)
    if args.instantiate:
        params = json.loads(args.instantiate)
        engine.pipeline.instantiate(params)
        say(f"已应用超参数：{json.dumps(params, ensure_ascii=False)}")
    say(f"模型加载 {time.perf_counter() - t0:.1f}s")
    t1 = time.perf_counter()
    turns = engine.diarize(tmp, num_speakers=args.speakers)
    elapsed = time.perf_counter() - t1
    say(f"分离耗时 {elapsed:.1f}s（{(args.end - args.start) / max(elapsed, 1e-6):.2f}× 实时）")
    say(f"turn 数={len(turns)} 说话人={sorted({t['speaker'] for t in turns})}")
    engine.unload()

    lo, hi = (float(x) for x in args.show.split("-"))
    say(f"\nturns in {lo:.0f}-{hi:.0f}s:")
    for t in turns:
        if float(t["end"]) < lo or float(t["start"]) > hi:
            continue
        say(f"  {t['start'] + args.start:8.2f}-{t['end'] + args.start:8.2f} "
            f"({float(t['end']) - float(t['start']):5.2f}s) {t['speaker']}")

    out = ROOT / "logs" / f"diar_probe_{args.model.replace('/', '_')}{args.tag}.json"
    out.write_text(json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\n写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
