"""Long-text chunking for summarisation."""
from __future__ import annotations

from summarization.chunking import chunk_utterances, format_evidence, speaker_label, turn_line


def _utts(n=40, chars=100, gap=2.0, speakers=("A", "B")):
    out = []
    t = 0.0
    for i in range(n):
        out.append(
            {
                "id": f"u{i:04d}",
                "start": t,
                "end": t + 5,
                "speaker": speakers[i % len(speakers)],
                "text": "字" * chars,
            }
        )
        t += 5 + gap
    return out


def test_every_utterance_lands_in_exactly_one_chunk():
    utts = _utts(30)
    chunks = chunk_utterances(utts, {"A": "访谈者", "B": "受访者"}, max_chars=700)
    ids = [uid for c in chunks for uid in c.utterance_ids]
    assert sorted(ids) == sorted(u["id"] for u in utts)
    assert len(chunks) > 1


def test_chunk_size_is_bounded():
    utts = _utts(30)
    chunks = chunk_utterances(utts, None, max_chars=700)
    # a chunk may exceed the soft limit by at most a couple of lines
    assert all(len(c.text) <= 700 + 400 for c in chunks)


def test_chunk_timestamps_are_monotonic():
    chunks = chunk_utterances(_utts(20), None, max_chars=600)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    for c in chunks:
        assert c.start <= c.end


def test_evidence_lines_use_timestamps_and_names():
    utts = [{"id": "u1", "start": 3725.4, "end": 3730, "speaker": "SPEAKER_01", "text": "我平时用另一款"}]
    text = format_evidence(utts, {"SPEAKER_01": "受访者 B"})
    assert text == "[01:02:05] 受访者 B: 我平时用另一款"


def test_speaker_label_fallbacks():
    assert speaker_label("SPEAKER_02", {}) == "说话人 02"
    assert speaker_label("SPEAKER_02", {"SPEAKER_02": "受访者"}) == "受访者"


def test_turn_line_handles_missing_start():
    line = turn_line({"speaker": "A", "text": " hi "}, None)
    assert line == "[--:--:--] A: hi"
