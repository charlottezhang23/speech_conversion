"""Diarization/ASR merge logic."""
from __future__ import annotations

import pytest

from transcription.merge import assign_speakers, build_utterances, speaker_stats


def _seg(start, end, text, words=None):
    return {"start": start, "end": end, "text": text, "words": words or []}


def test_assigns_dominant_speaker():
    turns = [(0.0, 5.0, "SPEAKER_00"), (5.0, 10.0, "SPEAKER_01")]
    segs = [_seg(0.5, 4.5, "hello"), _seg(5.5, 9.5, "world")]
    out = assign_speakers(segs, turns)
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert out[0]["speaker_confidence"] == pytest.approx(1.0)


def test_splits_segment_on_speaker_change():
    turns = [(0.0, 3.0, "SPEAKER_00"), (3.0, 6.0, "SPEAKER_01")]
    segs = [_seg(0.0, 6.0, "前半段后半段", words=[
        {"word": "前半段", "start": 0.5, "end": 2.5, "score": 0.9},
        {"word": "后半段", "start": 3.5, "end": 5.5, "score": 0.9},
    ])]
    out = assign_speakers(segs, turns)
    assert len(out) == 2
    assert out[0]["text"] == "前半段" and out[0]["speaker"] == "SPEAKER_00"
    assert out[1]["text"] == "后半段" and out[1]["speaker"] == "SPEAKER_01"


def test_splits_text_proportionally_without_words():
    turns = [(0.0, 2.5, "SPEAKER_00"), (2.5, 5.0, "SPEAKER_01")]
    segs = [_seg(0.0, 5.0, "0123456789")]
    out = assign_speakers(segs, turns)
    assert len(out) == 2
    assert out[0]["text"] + out[1]["text"] == "0123456789"


def test_short_tail_is_not_split_into_fragments():
    turns = [(0.0, 4.8, "SPEAKER_00"), (4.8, 5.0, "SPEAKER_01")]
    segs = [_seg(0.2, 5.0, "一句话")]
    out = assign_speakers(segs, turns)
    assert len(out) == 1
    assert out[0]["speaker"] == "SPEAKER_00"


def test_word_straddling_boundary_goes_to_the_side_of_its_middle():
    # 边界 2.4s 落在「中间」这个词的内部（2.0-3.0），按词中心 2.5 判定应归后者
    turns = [(0.0, 2.4, "SPEAKER_00"), (2.4, 5.0, "SPEAKER_01")]
    words = [
        {"word": "前面", "start": 0.4, "end": 1.4, "score": 0.9},
        {"word": "中间", "start": 2.0, "end": 3.0, "score": 0.9},
        {"word": "后面", "start": 3.2, "end": 4.6, "score": 0.9},
    ]
    out = assign_speakers([_seg(0.0, 5.0, "前面中间后面", words=words)], turns)
    assert [p["text"] for p in out] == ["前面", "中间后面"]
    assert [p["speaker"] for p in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_words_split_segment_into_multiple_runs():
    # 同一段里 A 说一句、B 插一句、A 继续说
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01"), (3.0, 5.0, "SPEAKER_00")]
    words = [
        {"word": "我家", "start": 0.3, "end": 1.0, "score": 0.9},
        {"word": "两只", "start": 1.0, "end": 1.9, "score": 0.9},
        {"word": "嗯嗯", "start": 2.2, "end": 2.8, "score": 0.9},
        {"word": "都是", "start": 3.2, "end": 4.0, "score": 0.9},
        {"word": "六岁", "start": 4.0, "end": 4.8, "score": 0.9},
    ]
    out = assign_speakers([_seg(0.0, 5.0, "我家两只嗯嗯都是六岁", words=words)], turns)
    assert [p["text"] for p in out] == ["我家两只", "嗯嗯", "都是六岁"]
    assert [p["speaker"] for p in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert [p["start"] for p in out] == [0.3, 2.2, 3.2]


def test_word_timestamps_do_not_lose_text():
    turns = [(0.0, 2.0, "SPEAKER_00"), (2.0, 4.0, "SPEAKER_01")]
    words = [
        {"word": "甲", "start": 0.2, "end": 0.6, "score": 0.9},
        {"word": "，", "start": 0.6, "end": 0.7, "score": 0.9},
        {"word": "乙", "start": 0.8, "end": 1.4, "score": 0.9},
        {"word": "丙", "start": 2.2, "end": 2.9, "score": 0.9},
    ]
    out = assign_speakers([_seg(0.2, 2.9, "甲，乙丙", words=words)], turns)
    assert "".join(p["text"] for p in out) == "甲，乙丙"
    assert out[0]["speaker"] == "SPEAKER_00" and out[-1]["speaker"] == "SPEAKER_01"


def test_nearest_speaker_when_no_overlap():
    turns = [(0.0, 1.0, "SPEAKER_00"), (5.0, 6.0, "SPEAKER_01")]
    segs = [_seg(1.2, 1.6, "嗯")]
    out = assign_speakers(segs, turns)
    assert out[0]["speaker"] == "SPEAKER_00"
    assert "nearest_speaker" in out[0]["flags"]


def test_no_diarization_falls_back_to_default_speaker():
    out = assign_speakers([_seg(0.0, 1.0, "hi")], [])
    assert out[0]["speaker"] == "SPEAKER_00"
    assert "no_diarization" in out[0]["flags"]


def test_silence_padded_word_does_not_steal_next_speaker():
    # 对齐把「了」拉长到 1.8s（23.80–25.58），跨过了 A 的静音和 B 的开头；
    # 收紧到语音区间后，「了」只应属于 A，B 从 25.60 开始
    turns = [(23.30, 23.86, "SPEAKER_00"), (25.14, 25.97, "SPEAKER_00"), (25.97, 28.36, "SPEAKER_01")]
    words = [
        {"word": "大", "start": 23.54, "end": 23.80, "score": 0.9},
        {"word": "了", "start": 23.80, "end": 25.58, "score": 0.9},
        {"word": "两", "start": 25.58, "end": 25.90, "score": 0.9},
        {"word": "都", "start": 26.02, "end": 26.18, "score": 0.9},
        {"word": "是", "start": 26.18, "end": 26.72, "score": 0.9},
    ]
    out = assign_speakers([_seg(23.54, 26.72, "大了两都是", words=words)], turns)
    assert "".join(p["text"] for p in out) == "大了两都是"
    assert out[-1]["speaker"] == "SPEAKER_01"
    # 被拉长的「了」被收紧到实际说话的 23.80–23.86，不再横跨整段静音
    trimmed = [w for p in out for w in p["words"] if w["word"] == "了"][0]
    assert trimmed["end"] <= 23.9


def test_overlap_flag():
    turns = [(0.0, 5.0, "SPEAKER_00"), (1.0, 4.0, "SPEAKER_01")]
    out = assign_speakers([_seg(0.5, 4.5, "对话")], turns)
    assert "overlap" in out[0]["flags"]


def test_build_utterances_merges_same_speaker():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "你好", "speaker": "A", "flags": []},
        {"start": 2.5, "end": 4.0, "text": "我很好", "speaker": "A", "flags": []},
        {"start": 4.5, "end": 6.0, "text": "谢谢", "speaker": "B", "flags": ["overlap"]},
    ]
    utts = build_utterances(segs)
    assert len(utts) == 2
    assert utts[0]["text"] == "你好我很好"  # CJK: no space inserted
    assert utts[0]["end"] == 4.0
    assert utts[1]["flags"] == ["overlap"]


def test_build_utterances_keeps_latin_spacing():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "hello", "speaker": "A", "flags": []},
        {"start": 2.2, "end": 3.0, "text": "world", "speaker": "A", "flags": []},
    ]
    utts = build_utterances(segs)
    assert utts[0]["text"] == "hello world"


def test_speaker_stats():
    stats = speaker_stats([
        {"speaker": "A", "start": 0, "end": 2, "text": "abc"},
        {"speaker": "A", "start": 2, "end": 4, "text": "de"},
        {"speaker": "B", "start": 4, "end": 5, "text": "f"},
    ])
    assert stats["A"]["turns"] == 2
    assert stats["A"]["speech_sec"] == 4.0
    assert stats["B"]["chars"] == 1
