"""Project storage and draft/edited transcript handling."""
from __future__ import annotations

import json

import pytest

import app.transcript as T
from app.projects import create_project, delete_project, get_project, list_projects, rename_project


def test_create_list_rename_delete(tmp_projects, tmp_path):
    from tests.conftest import _write_silence

    wav = _write_silence(tmp_path / "a.wav", 1.0)
    project = create_project("第一个访谈", wav, "a.wav", duration_sec=1.0)
    assert project.dir.exists()
    assert project.original_audio.exists()

    items = list_projects()
    assert len(items) == 1
    assert items[0]["name"] == "第一个访谈"
    assert items[0]["stages"]["transcribe"] == "pending"

    rename_project(project.id, "改名后")
    assert get_project(project.id).data["name"] == "改名后"

    delete_project(project.id)
    assert list_projects() == []


def test_invalid_project_id_is_rejected(tmp_projects):
    from app.projects import ProjectError, get_project

    with pytest.raises(ProjectError):
        get_project("../../etc/passwd")


def test_save_edits_overlays_draft(sample_project):
    draft = T.load_draft(sample_project)
    utts = [dict(u) for u in draft["utterances"]]
    utts[0]["text"] = "你目前养几只猫呀？"
    utts[1]["speaker"] = "SPEAKER_00"

    edited = T.save_edits(sample_project, utts, {"SPEAKER_01": "受访者"})
    assert edited["source"] == "human"
    assert edited["utterances"][0]["text"] == "你目前养几只猫呀？"
    assert edited["utterances"][0]["machine_text"] == "你目前养几只猫？"
    assert edited["utterances"][0]["edited"] is True

    payload = T.transcript_payload(sample_project)
    assert payload["version"] == "edited"
    assert payload["speaker_names"]["SPEAKER_01"] == "受访者"
    assert sample_project.speaker_names["SPEAKER_01"] == "受访者"

    # the machine draft must remain untouched
    assert T.load_draft(sample_project)["utterances"][0]["text"] == "你目前养几只猫？"


def test_draft_stale_detection(sample_project):
    import app.util as util

    draft = T.load_draft(sample_project)
    T.save_edits(sample_project, draft["utterances"], {})
    assert T.transcript_payload(sample_project)["draft_stale"] is False

    mutated = json.loads(json.dumps(draft))
    mutated["utterances"].append(
        {"id": "u0003", "start": 5.0, "end": 6.0, "speaker": "SPEAKER_00", "text": "新内容", "machine_text": "新内容", "flags": []}
    )
    util.write_json(sample_project.draft_transcript, mutated)
    assert T.transcript_payload(sample_project)["draft_stale"] is True


def test_reset_edits(sample_project):
    draft = T.load_draft(sample_project)
    T.save_edits(sample_project, draft["utterances"], {})
    assert T.load_edited(sample_project) is not None
    T.reset_edits(sample_project)
    assert T.load_edited(sample_project) is None
    assert T.transcript_payload(sample_project)["version"] == "draft"


def test_save_edits_rejects_bad_payload(sample_project):
    with pytest.raises(T.TranscriptError):
        T.save_edits(sample_project, "not-a-list", {})
    with pytest.raises(T.TranscriptError):
        T.save_edits(sample_project, [{"id": "x", "start": "abc", "end": 1, "text": "x"}], {})


def test_transcript_payload_requires_draft(sample_project):
    sample_project.draft_transcript.unlink()
    with pytest.raises(T.TranscriptError):
        T.transcript_payload(sample_project)
