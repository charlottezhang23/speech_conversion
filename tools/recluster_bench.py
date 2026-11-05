"""Measure how many diarization turns the second-pass speaker check flips,
for several confidence margins (no project data is modified).

Usage: python tools/recluster_bench.py --start 0 --end 300 --margins 0,0.02,0.05
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transcription import recluster as R  # noqa: E402

LOG = ROOT / "logs" / "recluster_bench.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="20261103-133124-b6812e")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=300.0)
    parser.add_argument("--margins", default="0,0.01,0.02,0.05,0.1")
    args = parser.parse_args()

    pdir = ROOT / "projects" / args.project
    wav = pdir / "audio" / "normalized_16k.wav"
    turns = json.loads((pdir / "diarization" / "raw.json").read_text(encoding="utf-8"))["turns"]
    window = [t for t in turns if float(t["end"]) > args.start and float(t["start"]) < args.end]

    lines: list[str] = []

    def say(msg: str = "") -> None:
        lines.append(msg)
        print(msg, flush=True)

    say(f"窗口 {args.start:.0f}-{args.end:.0f}s，turn={len(window)}")

    kept, matrix = R.embed_turns(wav, window)
    say(f"可嵌入 turn={len(kept)}，embedding 维度={matrix.shape[1] if matrix.size else 0}")
    clusters = R.kmeans(matrix, 2)
    mapping = R.map_clusters(clusters, [str(t["speaker"]) for t in kept], 2)
    centroids = np.stack(
        [matrix[clusters == c].mean(axis=0) if (clusters == c).any() else np.zeros(matrix.shape[1]) for c in range(2)]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    similarity = matrix @ centroids.T
    say(f"簇→标签：{mapping}")

    for margin in [float(m) for m in args.margins.split(",")]:
        flips = []
        uncertain = 0
        for idx, turn in enumerate(kept):
            proposed = mapping[int(clusters[idx])]
            current = str(turn["speaker"])
            if proposed == current:
                continue
            own = [c for c in range(2) if mapping[c] == current]
            gain = float(similarity[idx][int(clusters[idx])] - (similarity[idx][own[0]] if own else -1.0))
            if gain < margin:
                uncertain += 1
                continue
            flips.append((float(turn["start"]), float(turn["end"]), current, proposed, gain))
        say(f"\nmargin={margin:<5} 翻转 {len(flips)}/{len(kept)}  保守跳过 {uncertain}")
        for start, end, a, b, gain in flips[:12]:
            say(f"   {start:8.2f}-{end:8.2f} ({end - start:5.2f}s) {a} -> {b}  (+{gain:.3f})")

    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\n写入 {LOG.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
