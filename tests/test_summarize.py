"""Summarisation pipeline with a stubbed Ollama (no model download needed)."""
from __future__ import annotations

import json

import pytest

from summarization import summarize as S
from summarization import prompts as P
from summarization.ollama_client import ChatResult

fake_map_schema = P.MAP_SCHEMA


def _facts_json(n: int = 2) -> str:
    return json.dumps(
        {
            "chunk_summary": "受访者谈到导出流程繁琐。",
            "facts": [
                {
                    "type": "pain_point",
                    "statement": f"痛点 {i}",
                    "speaker": "受访者",
                    "time": "00:00:10",
                    "quote": "每次导出一份录音要点五次",
                    "quote_zh": "",
                }
                for i in range(n)
            ],
        },
        ensure_ascii=False,
    )


class FakeOllama:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model, messages, *, base_url=None, num_ctx=0, json_format=None, timeout_sec=0, on_token=None, **_kw):
        self.calls.append({"json_format": json_format, "system": messages[0]["content"], "user": messages[-1]["content"]})
        if json_format is None:
            return ChatResult(content="# 访谈纪要\n\n## 1. 访谈基本信息\n- 时长 1 分钟\n\n## 5. 主要痛点和困难\n- 导出流程繁琐 [00:00:10]", model=model)
        if "事实核查" in messages[0]["content"]:
            return ChatResult(
                content=json.dumps({"verdict": "ok", "unsupported": [], "notes": "全部有据"}, ensure_ascii=False),
                model=model,
            )
        if "合并" in messages[0]["content"]:
            return ChatResult(content=_facts_json(2), model=model)
        return ChatResult(content=_facts_json(2), model=model)


@pytest.fixture()
def utterances():
    return [
        {"id": "u1", "start": 5.0, "end": 10.0, "speaker": "SPEAKER_00", "text": "平时怎么整理录音？"},
        {"id": "u2", "start": 10.0, "end": 20.0, "speaker": "SPEAKER_01", "text": "每次导出一份录音要点五次，太麻烦了。"},
    ]


def test_summarize_writes_final_and_caches(monkeypatch, tmp_path, utterances):
    fake = FakeOllama()
    monkeypatch.setattr(S, "chat", fake)
    out = S.summarize(
        utterances=utterances,
        names={"SPEAKER_00": "访谈者", "SPEAKER_01": "受访者"},
        meta={"name": "测试", "duration_sec": 20, "transcript_source": "draft"},
        summary_dir=tmp_path,
        template="interview",
        model="qwen3:8b",
        base_url="http://127.0.0.1:11434",
        verify=True,
        max_chunk_chars=2000,
    )
    assert out["markdown"].startswith("# 访谈纪要")
    assert (tmp_path / "final.json").exists()
    assert (tmp_path / "final.md").read_text(encoding="utf-8").startswith("# 访谈纪要")
    assert out["verification"]["verdict"] == "ok"

    # compose + verify + at least one map call
    kinds = ["compose" if c["json_format"] is None else "json" for c in fake.calls]
    assert "compose" in kinds and "json" in kinds

    n_calls = len(fake.calls)
    S.summarize(
        utterances=utterances,
        names={"SPEAKER_00": "访谈者", "SPEAKER_01": "受访者"},
        meta={"name": "测试", "duration_sec": 20},
        summary_dir=tmp_path,
        template="interview",
        model="qwen3:8b",
        base_url="http://127.0.0.1:11434",
        verify=True,
        max_chunk_chars=2000,
    )
    # the map stage must be served from the chunk cache on the second run, and a
    # single chunk skips the LLM reduce step: only compose + verify hit the model
    assert len(fake.calls) - n_calls == 2
    assert all(c["json_format"] != fake_map_schema for c in fake.calls[n_calls:])


def test_summarize_rejects_unknown_template(monkeypatch, tmp_path, utterances):
    monkeypatch.setattr(S, "chat", FakeOllama())
    with pytest.raises(S.SummaryError):
        S.summarize(
            utterances=utterances,
            names={},
            meta={},
            summary_dir=tmp_path,
            template="nope",
            model="m",
            base_url="http://127.0.0.1:11434",
        )


def test_cancel_is_honoured(monkeypatch, tmp_path, utterances):
    monkeypatch.setattr(S, "chat", FakeOllama())
    with pytest.raises(S.SummaryError):
        S.summarize(
            utterances=utterances,
            names={},
            meta={},
            summary_dir=tmp_path,
            template="meeting",
            model="m",
            base_url="http://x",
            is_cancelled=lambda: True,
        )


def test_compress_facts_fits_budget(utterances):
    facts = [
        {"type": "pain_point", "statement": "x" * 200, "time": "00:00:01", "quote": "y" * 300}
        for _ in range(50)
    ]
    text = S._compress_facts(facts, 2000)
    assert len(text) <= 2000
    assert json.loads(text)[0]["type"] == "pain_point"
