"""Regression test for pyannote device handling (no model download needed)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from transcription.diarization import _to_torch_device  # noqa: E402


def test_string_device_is_converted():
    device = _to_torch_device("cpu")
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_torch_device_is_passed_through():
    device = torch.device("cpu")
    assert _to_torch_device(device) is device
