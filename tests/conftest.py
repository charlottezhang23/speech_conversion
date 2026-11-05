"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_projects(tmp_path, monkeypatch):
    """Point the storage layer at a temporary projects directory."""
    import app.projects as projects

    target = tmp_path / "projects"
    target.mkdir()
    monkeypatch.setattr(projects, "PROJECTS_DIR", target)
    return target


@pytest.fixture()
def sample_project(tmp_projects, tmp_path):
    """A project with a tiny generated WAV and a canned transcript."""
    from app.projects import create_project
    from app.util import write_json

    wav = tmp_path / "voice.wav"
    _write_silence(wav, seconds=2.0)
    project = create_project("测试项目", wav, "voice.wav", duration_sec=2.0)
    project.settings.update({"language": "zh", "model": "medium"})
    project.save()
    write_json(
        project.draft_transcript,
        {
            "schema_version": 1,
            "source": "machine",
            "language": "zh",
            "created_at": "2026-11-03T00:00:00Z",
            "speaker_names": {"SPEAKER_00": "说话人 A", "SPEAKER_01": "说话人 B"},
            "utterances": [
                {
                    "id": "u0001",
                    "start": 0.5,
                    "end": 2.0,
                    "speaker": "SPEAKER_00",
                    "text": "你目前养几只猫？",
                    "machine_text": "你目前养几只猫？",
                    "flags": [],
                },
                {
                    "id": "u0002",
                    "start": 2.5,
                    "end": 4.0,
                    "speaker": "SPEAKER_01",
                    "text": "两只，一只是英短。",
                    "machine_text": "两只，一只是英短。",
                    "flags": ["speaker_uncertain"],
                },
            ],
        },
    )
    return project


def _write_silence(path: Path, seconds: float = 1.0, sample_rate: int = 16000) -> Path:
    """Write a minimal WAV without external tools."""
    import struct
    import wave

    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
    return path


@pytest.fixture()
def silence_wav(tmp_path):
    return _write_silence(tmp_path / "silence.wav")
