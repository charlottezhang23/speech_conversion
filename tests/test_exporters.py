"""Export formats."""
from __future__ import annotations

import json

from exporters.format import to_json_bundle, to_markdown, to_srt, to_txt, split_markdown_sections

UTTS = [
    {"id": "u1", "start": 12.0, "end": 18.5, "speaker": "SPEAKER_00", "text": "你目前养几只猫？"},
    {"id": "u2", "start": 18.5, "end": 25.0, "speaker": "SPEAKER_01", "text": "两只，一只是英短。"},
]
NAMES = {"SPEAKER_00": "访谈者 A", "SPEAKER_01": "受访者 B"}
META = {"name": "宠物访谈", "original_name": "cat.m4a", "duration_sec": 61.4}


def test_txt_has_timestamp_and_speaker():
    text = to_txt(UTTS, NAMES, META, "draft")
    assert "[00:00:12] 访谈者 A：" in text
    assert "两只，一只是英短。" in text
    assert "机器初稿" in text


def test_markdown_structure():
    md = to_markdown(UTTS, NAMES, META, "edited")
    assert md.startswith("# 宠物访谈")
    assert "**访谈者 A**" in md
    assert "> [00:00:12] 你目前养几只猫？" in md
    assert "人工校对版" in md


def test_srt_timestamps_and_index():
    srt = to_srt(UTTS, NAMES)
    assert srt.startswith("1\n00:00:12,000 --> 00:00:18,500\n访谈者 A: 你目前养几只猫？")
    assert "2\n00:00:18,500 --> 00:00:25,000\n受访者 B: 两只，一只是英短。" in srt


def test_srt_clamps_last_end_to_duration():
    utts = [{"id": "u", "start": 58.0, "end": 999.0, "speaker": "A", "text": "x"}]
    srt = to_srt(utts, {}, fallback_end=60.0)
    assert "00:01:00,000" in srt


def test_json_bundle_is_valid_and_complete():
    payload = json.loads(to_json_bundle(project_id="p1", meta=META, utterances=UTTS, names=NAMES, source="edited", summary=None))
    assert payload["project_id"] == "p1"
    assert payload["transcript"]["speaker_names"]["SPEAKER_01"] == "受访者 B"
    assert len(payload["transcript"]["utterances"]) == 2


def test_split_markdown_sections():
    md = "# 标题\n\n## 1. 会议信息\n内容 A\n\n## 2. 主题\n- 条目"
    sections = split_markdown_sections(md)
    assert sections[0][0] == "1. 会议信息"
    assert sections[0][1] == "内容 A"
    assert sections[1][0] == "2. 主题"
