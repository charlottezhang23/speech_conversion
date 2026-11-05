"""Pipeline merge stage without loading any model (canned ASR + diarization)."""
from __future__ import annotations

import json

from app.pipeline import build_draft
from app.projects import get_project
from app.transcript import load_draft


ASR = {
    "language": "zh",
    "chunks": 1,
    "segments": [
        {
            "start": 0.5,
            "end": 12.0,
            "text": "你好谢谢你今天接受采访先请你简单介绍一下自己好的我今年二十九岁在杭州做产品设计家里养了两只猫",
            "words": [],
            "avg_logprob": -0.05,
        },
        {
            "start": 12.4,
            "end": 20.0,
            "text": "两只猫分别是什么品种一只是英国短毛猫另一只是挪威森林猫",
            "words": [],
            "avg_logprob": -0.05,
        },
    ],
}

DIARIZATION = {
    "model": "pyannote/speaker-diarization-community-1",
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "turns": [
        {"start": 0.0, "end": 6.2, "speaker": "SPEAKER_00"},
        {"start": 6.3, "end": 12.0, "speaker": "SPEAKER_01"},
        {"start": 12.4, "end": 15.0, "speaker": "SPEAKER_00"},
        {"start": 15.1, "end": 20.0, "speaker": "SPEAKER_01"},
    ],
}


def test_build_draft_splits_by_speaker(sample_project):
    draft = build_draft(sample_project, ASR, DIARIZATION)
    assert len(draft["utterances"]) >= 4
    speakers = [u["speaker"] for u in draft["utterances"]]
    assert "SPEAKER_00" in speakers and "SPEAKER_01" in speakers
    assert draft["diarization"]["enabled"] is True
    assert draft["diarization"]["turn_count"] == 4
    # every utterance must keep a timestamp inside the audio
    for u in draft["utterances"]:
        assert 0 <= u["start"] <= u["end"] <= 20.5
    # names are generated automatically
    assert sample_project.speaker_names["SPEAKER_00"]
    assert get_project(sample_project.id).data["speaker_names"]["SPEAKER_01"]


def test_build_draft_without_diarization(sample_project):
    draft = build_draft(sample_project, ASR, None)
    assert {u["speaker"] for u in draft["utterances"]} == {"SPEAKER_00"}
    assert draft["diarization"]["enabled"] is False


def test_draft_is_persisted_and_reloadable(sample_project):
    build_draft(sample_project, ASR, DIARIZATION)
    reloaded = load_draft(get_project(sample_project.id))
    assert len(reloaded["utterances"]) == 4
    assert json.loads(json.dumps(reloaded))["schema_version"] == 1
    # the project folder survives a fresh read from disk
    assert get_project(sample_project.id).has_draft()
