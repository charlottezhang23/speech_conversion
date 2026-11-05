"""Audio probing / normalisation / chunk planning (requires bundled FFmpeg)."""
from __future__ import annotations

import pytest

from app import audio as A
from app.config import find_ffmpeg


pytestmark = pytest.mark.skipif(find_ffmpeg() is None, reason="bundled FFmpeg not installed")


def test_probe_reports_duration(silence_wav):
    info = A.probe(silence_wav)
    assert info.duration_sec == pytest.approx(1.0, abs=0.15)
    assert info.sample_rate == 16000
    assert info.channels == 1


def test_normalize_to_16k_mono(tmp_path, silence_wav):
    out = tmp_path / "norm.wav"
    A.normalize(silence_wav, out)
    info = A.probe(out)
    assert info.sample_rate == 16000
    assert info.channels == 1


def test_slice_wav_produces_requested_duration(tmp_path, silence_wav):
    out = tmp_path / "slice.wav"
    A.slice_wav(silence_wav, 0.2, 0.7, out)
    info = A.probe(out)
    assert info.duration_sec == pytest.approx(0.5, abs=0.12)


def test_detect_silences_on_silent_file(silence_wav):
    silences = A.detect_silences(silence_wav)
    assert silences, "a fully silent file should be reported as silence"
    start, end = silences[0]
    assert end > start


def test_plan_chunks_short_audio_is_single_chunk():
    assert A.plan_chunks(90.0, [], target_sec=600) == [(0.0, 90.0)]


def test_plan_chunks_cuts_near_silences():
    duration = 3600.0
    silences = [(598.0, 601.0), (1197.0, 1202.0), (1799.0, 1803.0)]
    chunks = A.plan_chunks(duration, silences, target_sec=600.0)
    assert len(chunks) >= 5
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == pytest.approx(duration)
    # consecutive chunks must be contiguous
    for (a_start, a_end), (b_start, _) in zip(chunks, chunks[1:]):
        assert a_end == pytest.approx(b_start)


def test_plan_chunks_merges_short_tail():
    # 610s audio with a silence exactly at 600s -> tail of 10s must merge back
    chunks = A.plan_chunks(610.0, [(598.0, 602.0)], target_sec=600.0)
    assert chunks == [(0.0, 610.0)] or (chunks[-1][1] - chunks[-1][0]) >= 30


def test_load_waveform_shape(tmp_path, silence_wav):
    import numpy as np

    wave = A.load_waveform(silence_wav)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert len(wave) == pytest.approx(16000, abs=200)


def test_ffmpeg_error_for_missing_file(tmp_path):
    with pytest.raises(A.FfmpegError):
        A.probe(tmp_path / "nope.wav")
