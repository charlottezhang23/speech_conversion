"""Prompt-echo / repetition filtering for ASR output."""
from __future__ import annotations

import pytest

from transcription.clean import clean_segments, clean_text, echo_ratio, repetition_ratio


@pytest.mark.parametrize(
    "text",
    [
        "请用简体中文并加上标点符号。",
        "文并加上标点符号。请用简体中文并加上标点符号。请用",
        "简体中文并加上标点符号。并加上标点符号。并",
        "请用简体中",
        "简体中文",
        "加上标点符号。请",
        "以下是一段普通话访谈录音",
    ],
)
def test_prompt_fragments_are_dropped(text):
    cleaned, action = clean_text(text)
    assert cleaned == ""
    assert action in ("echo", "trimmed")


def test_known_phrase_is_trimmed_from_real_sentence():
    text = "请用简体中文并加上标点符号。我家两只猫一只四岁一只三岁半。"
    cleaned, action = clean_text(text)
    assert action == "trimmed"
    assert "简体中文" not in cleaned
    assert "我家两只猫" in cleaned


def test_normal_speech_is_kept():
    for text in [
        "我家两只，一只四岁，一只三岁半。",
        "我最看重吸力和续航，价格两千以内可以接受。",
        "はい、そうですね。",
    ]:
        cleaned, action = clean_text(text)
        assert action == "kept"
        assert cleaned == text


def test_repetition_loop_is_dropped():
    loop = "谢谢观看。" * 12
    assert repetition_ratio(loop) > 0.5
    cleaned, action = clean_text(loop)
    assert cleaned == ""
    assert action == "loop"


def test_echo_ratio_is_low_for_real_text():
    assert echo_ratio("我们家两只猫都已经绝育了") < 0.5


def test_clean_segments_reports_stats_and_keeps_input_untouched():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "请用简体中文并加上标点符号。"},
        {"start": 1.0, "end": 2.0, "text": "请用简体中文并加上标点符号。我家两只猫。"},
        {"start": 2.0, "end": 3.0, "text": "一只四岁，一只三岁半。"},
        {"start": 3.0, "end": 4.0, "text": "谢谢观看。" * 10},
    ]
    kept, stats = clean_segments(segments)
    assert len(kept) == 2
    assert stats["dropped_echo"] == 1
    assert stats["dropped_loop"] == 1
    assert stats["trimmed"] == 1
    assert stats["filtered_total"] == 2
    assert stats["examples"][0]["reason"] == "echo"
    # input list must not be mutated
    assert segments[0]["text"] == "请用简体中文并加上标点符号。"
    assert "我家两只猫" in kept[0]["text"]
