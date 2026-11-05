"""HTTP API smoke tests (no models required)."""
from __future__ import annotations

import io
import wave
import struct

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_projects, tmp_path, monkeypatch):
    import app.config as config
    import app.server as server

    # keep tests away from the real config/ and secrets/ folders
    monkeypatch.setattr(config, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(config, "TOKEN_FILE", tmp_path / "hf_token.txt")
    return TestClient(server.app)


def _wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<" + "h" * frames, *([0] * frames)))
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_system_endpoint(client):
    r = client.get("/api/system")
    assert r.status_code == 200
    body = r.json()
    assert "env" in body and "packages" in body
    assert "interview" in body["templates"]


def test_settings_roundtrip(client):
    r = client.get("/api/settings")
    assert r.status_code == 200
    assert "hf_token_set" in r.json()
    r = client.post("/api/settings", json={"ollama_model": "qwen3:8b", "summary_chunk_chars": 3000})
    assert r.status_code == 200
    assert r.json()["summary_chunk_chars"] == 3000
    # unknown keys are ignored
    r = client.post("/api/settings", json={"evil": "x"})
    assert "evil" not in r.json()


def test_hf_token_is_never_returned(client):
    r = client.post("/api/settings/hf-token", json={"token": "hf_secret_value"})
    assert r.status_code == 200
    assert r.json() == {"hf_token_set": True}
    body = client.get("/api/settings").json()
    assert "hf_secret_value" not in str(body)
    client.post("/api/settings/hf-token", json={"token": ""})


def test_hf_verify_without_token(client):
    r = client.get("/api/hf/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["reason"] == "missing"
    assert "Token" in body["message"]


def test_project_create_list_delete(client):
    files = {"file": ("sample.wav", _wav_bytes(1.0), "audio/wav")}
    r = client.post("/api/projects", files=files, data={"name": "接口测试"})
    assert r.status_code == 200, r.text
    project = r.json()["project"]
    pid = project["id"]

    listing = client.get("/api/projects").json()
    assert any(p["id"] == pid for p in listing)

    detail = client.get(f"/api/projects/{pid}")
    assert detail.status_code == 200
    assert detail.json()["name"] == "接口测试"

    audio = client.get(f"/api/projects/{pid}/audio", headers={"Range": "bytes=0-99"})
    assert audio.status_code == 206
    assert audio.headers["content-range"].startswith("bytes 0-99/")
    assert len(audio.content) <= 100

    rename = client.patch(f"/api/projects/{pid}", json={"name": "改名"})
    assert rename.json()["name"] == "改名"

    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_upload_rejects_unknown_extension(client):
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    r = client.post("/api/projects", files=files, data={"name": "bad"})
    assert r.status_code == 400


def test_unknown_project_returns_404(client):
    assert client.get("/api/projects/20260101-000000-abcdef").status_code == 404
    assert client.get("/api/projects/../../etc/passwd").status_code in (404, 405)


def test_transcript_endpoints(client, sample_project):
    pid = sample_project.id
    r = client.get(f"/api/projects/{pid}/transcript")
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "draft"
    assert len(body["utterances"]) == 2

    utts = body["utterances"]
    utts[0]["text"] = "修改后的文字"
    r = client.put(f"/api/projects/{pid}/transcript", json={"utterances": utts, "speaker_names": {"SPEAKER_00": "访谈者"}})
    assert r.status_code == 200
    assert r.json()["stats"]["edited_utterances"] >= 1

    r = client.get(f"/api/projects/{pid}/transcript")
    assert r.json()["version"] == "edited"
    assert r.json()["utterances"][0]["text"] == "修改后的文字"

    r = client.delete(f"/api/projects/{pid}/transcript/edited")
    assert r.status_code == 200
    assert client.get(f"/api/projects/{pid}/transcript").json()["version"] == "draft"


def test_export_formats(client, sample_project):
    pid = sample_project.id
    for fmt in ("txt", "md", "srt", "json", "docx"):
        r = client.get(f"/api/projects/{pid}/export/{fmt}")
        assert r.status_code == 200, f"{fmt}: {r.text[:200]}"
        assert len(r.content) > 50
    assert "你目前养几只猫" in client.get(f"/api/projects/{pid}/export/txt").text
    assert "00:00:12,000" not in client.get(f"/api/projects/{pid}/export/srt").text  # starts at 0.5s
    assert "00:00:00,500" in client.get(f"/api/projects/{pid}/export/srt").text


def test_summary_without_data_returns_404(client, sample_project):
    assert client.get(f"/api/projects/{sample_project.id}/summary").status_code == 404


def test_job_endpoints_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404
    assert client.post("/api/jobs/deadbeef/cancel").status_code == 404
