"""Second-pass speaker verification.

pyannote decides speaker labels frame-by-frame and can swap two speakers for a
few seconds. Each finished turn, however, carries enough audio for a clean
speaker embedding, so we re-cluster the turn embeddings globally and flip the
turns that are confidently on the wrong side.

`refine_turns` is a no-op when there is only one speaker or no turn long enough.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from app.config import MODELS_DIR, get_hf_token

logger = logging.getLogger("interviewai.recluster")

EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_TURN_SEC = 0.6          # shorter turns carry too little audio for an embedding
DEFAULT_MARGIN = 0.05       # cosine margin required before overriding pyannote
_CACHE: dict[str, Any] = {}


def _load_inference(model_name: str = EMBED_MODEL):
    """Load (and cache) the embedding model + inference wrapper."""
    cached = _CACHE.get(model_name)
    if cached is not None:
        return cached
    import torch
    from pyannote.audio import Inference, Model

    token = get_hf_token()
    model = Model.from_pretrained(model_name, token=token, cache_dir=str(MODELS_DIR / "pyannote"))
    inference = Inference(model, window="whole")
    inference.to(torch.device("cpu"))
    _CACHE[model_name] = inference
    logger.info("声纹模型载入完成：%s", model_name)
    return inference


def unload() -> None:
    _CACHE.clear()


def embed_turns(
    wav_path: str | Path,
    turns: Sequence[dict],
    *,
    min_len: float = MIN_TURN_SEC,
    inference: Any = None,
    model_name: str = EMBED_MODEL,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """Return (turns long enough for an embedding, L2-normalised embedding matrix)."""
    import soundfile as sf
    import torch

    infer = inference or _load_inference(model_name)
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)

    kept: list[dict] = []
    vectors: list[np.ndarray] = []
    min_samples = int(0.3 * sr)
    total = max(1, len(turns))
    for index, turn in enumerate(turns):
        start, end = float(turn["start"]), float(turn["end"])
        if end - start < min_len:
            continue
        a = max(0, int(start * sr))
        b = min(len(mono), int(end * sr))
        chunk = mono[a:b]
        if len(chunk) < min_samples:
            continue
        tensor = torch.from_numpy(np.ascontiguousarray(chunk[None, :]))
        embedding = infer({"waveform": tensor, "sample_rate": int(sr)})
        vectors.append(np.asarray(embedding).reshape(-1))
        kept.append(turn)
        if progress and index % 20 == 0:
            progress(5.0 + 60.0 * index / total, f"二次声纹校验：提取声纹 {index + 1}/{total}")
    if not kept:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack(vectors).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return kept, matrix


def kmeans(matrix: np.ndarray, k: int, *, iters: int = 100, restarts: int = 12, seed: int = 0) -> np.ndarray:
    """Plain k-means on unit vectors (cosine distance == euclidean distance)."""
    n = len(matrix)
    if k <= 1 or n <= k:
        return np.zeros(n, dtype=int)
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = np.zeros(n, dtype=int), None
    for _ in range(restarts):
        centers = matrix[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1)
        for _ in range(iters):
            dist = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            if (new_labels == labels).all():
                break
            labels = new_labels
            for cluster in range(k):
                if (labels == cluster).any():
                    centers[cluster] = matrix[labels == cluster].mean(axis=0)
        inertia = float(((matrix - centers[labels]) ** 2).sum())
        if best_inertia is None or inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def map_clusters(labels: np.ndarray, original: Sequence[str], k: int) -> dict[int, str]:
    """Assign each cluster the pyannote label it covers most (one-to-one greedy)."""
    counts: dict[int, dict[str, int]] = {c: {} for c in range(k)}
    for idx, cluster in enumerate(labels):
        bucket = counts[int(cluster)]
        bucket[original[idx]] = bucket.get(original[idx], 0) + 1
    mapping: dict[int, str] = {}
    used: set[str] = set()
    pairs = sorted(
        ((count, cluster, label) for cluster, bucket in counts.items() for label, count in bucket.items()),
        reverse=True,
    )
    for _count, cluster, label in pairs:
        if cluster in mapping or label in used:
            continue
        mapping[cluster] = label
        used.add(label)
    remaining_labels = [lbl for lbl in sorted(set(original)) if lbl not in used]
    for cluster in range(k):
        if cluster not in mapping:
            mapping[cluster] = remaining_labels.pop(0) if remaining_labels else sorted(set(original))[0]
    return mapping


def refine_turns(
    wav_path: str | Path,
    turns: Sequence[dict],
    *,
    margin: float = DEFAULT_MARGIN,
    min_len: float = MIN_TURN_SEC,
    progress: Callable[[float, str], None] | None = None,
    model_name: str = EMBED_MODEL,
) -> tuple[list[dict], dict[str, Any]]:
    """Return (turns with verified labels, stats). Original list is not mutated."""
    labels = [str(t["speaker"]) for t in turns]
    k = len(set(labels))
    stats: dict[str, Any] = {
        "enabled": True,
        "model": EMBED_MODEL if model_name == EMBED_MODEL else model_name,
        "speakers": k,
        "embedded": 0,
        "flipped": 0,
        "uncertain": 0,
        "margin": margin,
        "examples": [],
    }
    if k < 2:
        stats["enabled"] = False
        stats["reason"] = "只有一位说话人"
        return [dict(t) for t in turns], stats

    if progress:
        progress(5.0, "二次声纹校验：提取每段声纹…")
    kept, matrix = embed_turns(wav_path, turns, min_len=min_len, model_name=model_name, progress=progress)
    stats["embedded"] = len(kept)
    if len(kept) < max(4, 2 * k):
        stats["enabled"] = False
        stats["reason"] = "可用的长 turn 太少"
        return [dict(t) for t in turns], stats

    if progress:
        progress(70.0, "二次声纹校验：重新聚类…")
    clusters = kmeans(matrix, k)
    mapping = map_clusters(clusters, [str(t["speaker"]) for t in kept], k)

    # cluster centroids, for the confidence margin
    centroids = np.stack(
        [matrix[clusters == c].mean(axis=0) if (clusters == c).any() else np.zeros(matrix.shape[1]) for c in range(k)]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    similarity = matrix @ centroids.T  # (n, k)

    out = [dict(t) for t in turns]
    by_id = {id(t): idx for idx, t in enumerate(turns)}
    for idx, turn in enumerate(kept):
        position = by_id[id(turn)]
        proposed = mapping[int(clusters[idx])]
        current = str(turn["speaker"])
        if proposed == current:
            continue
        own_col = [c for c in range(k) if mapping[c] == current]
        own_sim = similarity[idx][own_col[0]] if own_col else -1.0
        gain = float(similarity[idx][int(clusters[idx])] - own_sim)
        if gain < margin:
            stats["uncertain"] += 1
            continue
        out[position]["speaker"] = proposed
        out[position]["refined_from"] = current
        stats["flipped"] += 1
        if len(stats["examples"]) < 8:
            stats["examples"].append(
                f"{float(turn['start']):.1f}-{float(turn['end']):.1f}s {current}→{proposed}（+{gain:.3f}）"
            )
    if progress:
        progress(98.0, "二次声纹校验完成")
    return out, stats
