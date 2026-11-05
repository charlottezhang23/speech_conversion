"""Second-pass speaker verification (recluster) logic with a fake embedder."""
from __future__ import annotations

import numpy as np

from transcription import recluster as R


def test_kmeans_separates_two_clear_groups():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    labels = R.kmeans(matrix, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_kmeans_single_cluster_is_noop():
    matrix = np.ones((4, 3), dtype=np.float32)
    assert set(R.kmeans(matrix, 1).tolist()) == {0}


def test_map_clusters_is_one_to_one():
    labels = np.array([0, 0, 1, 1, 1])
    mapping = R.map_clusters(labels, ["A", "A", "B", "B", "B"], 2)
    assert mapping == {0: "A", 1: "B"}
    assert len(set(mapping.values())) == 2


def test_refine_flips_confidently_wrong_turn(monkeypatch):
    # 4 个 turn：前两个是 SPEAKER_00，第三个声纹属于另一人，第四个被 pyannote 判成
    # SPEAKER_01 但声纹明显和 SPEAKER_00 一致 → 应该被翻转回 SPEAKER_00
    embeddings = {
        0.0: [1.0, 0.0, 0.0],
        1.0: [0.98, 0.02, 0.0],
        2.0: [0.0, 1.0, 0.0],
        3.0: [1.0, 0.0, 0.0],
    }
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_01"},
    ]

    def fake_embed(wav_path, items, **kwargs):  # noqa: ANN001
        kept = [t for t in items if float(t["end"]) - float(t["start"]) >= 0.6]
        matrix = np.stack([np.array(embeddings[float(t["start"])], dtype=np.float32) for t in kept])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return kept, matrix

    monkeypatch.setattr(R, "embed_turns", fake_embed)
    refined, stats = R.refine_turns("unused.wav", turns, margin=0.05)
    assert stats["flipped"] == 1
    assert refined[3]["speaker"] == "SPEAKER_00"
    assert refined[3]["refined_from"] == "SPEAKER_01"
    assert [t["speaker"] for t in refined[:3]] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]


def test_refine_keeps_labels_when_gain_below_margin(monkeypatch):
    # 两个 turn 的声纹几乎相同，翻转的收益低于 margin，应保守保留原标签
    embeddings = {0.0: [1.0, 0.0], 1.0: [1.0, 0.01], 2.0: [0.0, 1.0], 3.0: [1.0, 0.0]}

    def fake_embed(wav_path, turns, **kwargs):  # noqa: ANN001
        kept = [t for t in turns if float(t["end"]) - float(t["start"]) >= 0.6]
        matrix = np.stack([np.array(embeddings[float(t["start"])], dtype=np.float32) for t in kept])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        return kept, matrix

    monkeypatch.setattr(R, "embed_turns", fake_embed)
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_01"},
        {"start": 3.0, "end": 4.0, "speaker": "SPEAKER_00"},
    ]
    # margin 大于可能的最大收益（cosine 差值最大 2.0），任何翻转都应被跳过
    refined, stats = R.refine_turns("unused.wav", turns, margin=1.9)
    assert stats["flipped"] == 0
    assert [t["speaker"] for t in refined] == [t["speaker"] for t in turns]


def test_refine_is_noop_for_single_speaker():
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]
    refined, stats = R.refine_turns("unused.wav", turns)
    assert stats["enabled"] is False
    assert refined == turns
