"""Second-pass speaker verification: re-cluster per-turn embeddings and compare
with pyannote's own labels.

If the two disagree on many turns, the labels are unstable and a re-clustering
pass (or per-utterance verification) is worth adding.

Usage: python tools/speaker_recluster.py --project 20261103-133124-b6812e --start 0 --end 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_hf_token  # noqa: E402

LOG = ROOT / "logs" / "speaker_recluster.txt"
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"


def kmeans2(x: np.ndarray, iters: int = 100, restarts: int = 20, seed: int = 0) -> np.ndarray:
    """Tiny 2-means on L2-normalised vectors (cosine == euclidean here)."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, None
    n = len(x)
    for _ in range(restarts):
        idx = rng.choice(n, size=2, replace=False)
        centers = x[idx].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(iters):
            dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            for k in range(2):
                if (labels == k).any():
                    centers[k] = x[labels == k].mean(axis=0)
        inertia = ((x - centers[labels]) ** 2).sum()
        if best_inertia is None or inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels if best_labels is not None else np.zeros(n, dtype=int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="20261103-133124-b6812e")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=300.0)
    parser.add_argument("--min-len", type=float, default=0.6)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model

    lines: list[str] = []

    def say(msg: str = "") -> None:
        lines.append(msg)
        print(msg, flush=True)

    pdir = ROOT / "projects" / args.project
    wav = pdir / "audio" / "normalized_16k.wav"
    turns = json.loads((pdir / "diarization" / "raw.json").read_text(encoding="utf-8"))["turns"]
    turns = [t for t in turns if float(t["end"]) > args.start and float(t["start"]) < args.end]

    token = get_hf_token()
    model = Model.from_pretrained(EMBED_MODEL, token=token, cache_dir=str(ROOT / "models" / "pyannote"))
    inference = Inference(model, window="whole")
    inference.to(torch.device("cpu"))

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    lo = int(args.start * sr)
    hi = int(args.end * sr)
    window = mono[lo:hi]

    kept: list[dict] = []
    embs: list[np.ndarray] = []
    for turn in turns:
        s, e = float(turn["start"]), float(turn["end"])
        if e - s < args.min_len:
            continue
        a = int(max(0.0, s - args.start) * sr)
        b = int(max(0.0, e - args.start) * sr)
        if b - a < int(0.3 * sr):
            continue
        chunk = window[a:b]
        if len(chunk) < int(0.3 * sr):
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(chunk[None, :]))
        emb = inference({"waveform": tensor, "sample_rate": int(sr)})
        embs.append(np.asarray(emb).reshape(-1))
        kept.append(turn)

    x = np.stack(embs)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    labels = kmeans2(x)
    orig = [t["speaker"] for t in kept]

    # map each cluster to the pyannote label it overlaps most
    mapping: dict[int, str] = {}
    for k in (0, 1):
        subset = [orig[i] for i in range(len(kept)) if labels[i] == k]
        mapping[k] = max(set(subset), key=subset.count) if subset else "?"
    new = [mapping[k] for k in labels]

    flips = [i for i in range(len(kept)) if new[i] != orig[i]]
    say(f"窗口 {args.start:.0f}-{args.end:.0f}s：参与比较的 turn={len(kept)}（过滤 <{args.min_len}s 的短 turn）")
    say(f"重新聚类后与 pyannote 标签不一致的 turn：{len(flips)}/{len(kept)}（{len(flips) / max(1, len(kept)):.1%}）")
    say(f"簇→标签映射：{mapping}")
    say("")
    say("不一致的 turn（时间 / pyannote / 重聚类）：")
    for i in flips[:40]:
        t = kept[i]
        say(f"  {float(t['start']):8.2f}-{float(t['end']):8.2f} ({float(t['end']) - float(t['start']):5.2f}s) "
            f"{orig[i]} -> {new[i]}")

    out = ROOT / "logs" / "speaker_recluster.json"
    out.write_text(
        json.dumps(
            {
                "window": [args.start, args.end],
                "turns": [
                    {"start": float(t["start"]), "end": float(t["end"]), "pyannote": orig[i], "recluster": new[i]}
                    for i, t in enumerate(kept)
                ],
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"\n写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
