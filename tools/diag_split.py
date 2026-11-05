"""Diagnose why speaker turns got merged inside one utterance.

Reads a project's diarization turns, ASR segments and draft utterances, then for
every utterance that contains speech from more than one speaker, explains which
step failed: no diarization boundary inside the ASR segment, the 0.6 s edge guard,
the 1.2 s minimum length, or a boundary that should have split but did not.

Usage: python tools/diag_split.py 20261103-133124-b6812e [--examples 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_SPLIT_PART = 0.6
MIN_SPLIT_SEGMENT = 1.2
LOG = ROOT / "logs" / "diag_split.txt"

_LINES: list[str] = []


def say(msg: str = "") -> None:
    _LINES.append(msg)
    print(msg, flush=True)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_share(turns, start: float, end: float) -> dict[str, float]:
    share: dict[str, float] = {}
    for turn in turns:
        ov = overlap(start, end, float(turn["start"]), float(turn["end"]))
        if ov > 0:
            share[turn["speaker"]] = share.get(turn["speaker"], 0.0) + ov
    return share


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--range", dest="span", default="", help="打印时间范围内的句子，如 90-120")
    args = parser.parse_args()

    pdir = ROOT / "projects" / args.project
    LOG.write_text("", encoding="utf-8")
    turns = load(pdir / "diarization" / "raw.json")["turns"]
    turns.sort(key=lambda t: float(t["start"]))
    draft = load(pdir / "transcript" / "draft.json")
    utts = draft.get("utterances") or []
    manifest = load(pdir / "asr" / "manifest.json")

    segs = []
    for path in sorted((pdir / "asr" / "chunks").glob("*.json")):
        segs.extend(load(path).get("segments") or [])
    segs.sort(key=lambda s: float(s["start"]))

    say(f"项目 {args.project}")
    say(f"  asr model={manifest['settings'].get('model')} align={manifest['settings'].get('align')} "
        f"num_speakers={manifest['settings'].get('num_speakers')}")
    say(f"  diarization turns={len(turns)}  asr segments={len(segs)}  utterances={len(utts)}")
    say(f"  asr segment 有 words 字段的数量={sum(1 for s in segs if s.get('words'))}")
    durations = sorted(float(s["end"]) - float(s["start"]) for s in segs)
    if durations:
        say(f"  asr segment 时长 中位数={durations[len(durations) // 2]:.1f}s 最大={durations[-1]:.1f}s")

    mixed = []
    for u in utts:
        start, end = float(u["start"]), float(u["end"])
        share = speaker_share(turns, start, end)
        total = sum(share.values()) or 1e-6
        if len(share) < 2:
            continue
        ranked = sorted(share.items(), key=lambda kv: -kv[1])
        minor = ranked[1][1]
        if minor >= max(1.5, 0.15 * total):
            mixed.append((u, share, minor / total))

    say(f"\n含两位说话人的 utterance：{len(mixed)} / {len(utts)}")
    cause: dict[str, int] = {}
    examples = []
    for u, share, minor_ratio in mixed:
        start, end = float(u["start"]), float(u["end"])
        inside = [s for s in segs if overlap(start, end, float(s["start"]), float(s["end"])) > 0.2]
        reasons = []
        for s in inside:
            s0, s1 = float(s["start"]), float(s["end"])
            internal = [float(t["start"]) for t in turns if s0 + 0.15 < float(t["start"]) < s1 - 0.15]
            if not internal:
                continue
            for b in internal:
                if not (s0 + MIN_SPLIT_PART < b < s1 - MIN_SPLIT_PART):
                    reasons.append("边界距段首/段尾不足 0.6s，被跳过")
                elif s1 - s0 < MIN_SPLIT_SEGMENT:
                    reasons.append("段长不足 1.2s，被跳过")
                else:
                    reasons.append("应已切分（需人工核对）")
        if not reasons:
            reasons = ["该段内没有分离边界（分离结果本身把两人判成同一人）"]
        for r in set(reasons):
            cause[r] = cause.get(r, 0) + 1
        if len(examples) < args.examples:
            examples.append((u, share, minor_ratio, reasons))

    for r, n in sorted(cause.items(), key=lambda kv: -kv[1]):
        say(f"  {n:3d} 处：{r}")

    say("\n示例：")
    for u, share, minor_ratio, reasons in examples:
        head = str(u.get("text") or "").replace("\n", " ")
        say(f"  [{u['start']:.1f}-{u['end']:.1f}] {u.get('speaker')} "
            f"(次说话人占 {minor_ratio:.0%}) 全文={head}")
        say(f"      分离占比={ {k: round(v, 1) for k, v in share.items()} } 原因={sorted(set(reasons))}")
    LOG.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    say(f"\n结果已写入 {LOG.relative_to(ROOT)}")

    if args.span:
        lo, _, hi = args.span.partition("-")
        lo_f, hi_f = float(lo), float(hi or lo)
        say(f"\n{lo_f:.0f}s - {hi_f:.0f}s 范围内的句子：")
        for u in utts:
            s, e = float(u["start"]), float(u["end"])
            if e < lo_f or s > hi_f:
                continue
            name = (draft.get("speaker_names") or {}).get(u.get("speaker"), u.get("speaker"))
            say(f"  [{s:7.2f}-{e:7.2f}] {name}: {u.get('text')}")
        LOG.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
