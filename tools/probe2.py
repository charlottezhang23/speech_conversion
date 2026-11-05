"""Probe library signatures and write JSON to logs/probe.json (called manually)."""
from __future__ import annotations

import inspect
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "logs" / "probe.json"
report: dict[str, object] = {}

t0 = time.perf_counter()
import app.config  # noqa: E402  (adds FFmpeg to PATH + warning filters)

report["config_import_sec"] = round(time.perf_counter() - t0, 2)

import torch  # noqa: E402

report["torch"] = torch.__version__
t0 = time.perf_counter()
import whisperx  # noqa: E402
import whisperx.asr  # noqa: E402
import whisperx.diarize  # noqa: E402

report["whisperx_import_sec"] = round(time.perf_counter() - t0, 2)
report["whisperx_all"] = [n for n in dir(whisperx) if not n.startswith("_")]
for name in ("load_model", "load_align_model", "align"):
    fn = getattr(whisperx, name, None)
    report[name] = str(inspect.signature(fn)) if fn else "MISSING"
report["WhisperModel.__init__"] = str(inspect.signature(whisperx.asr.WhisperModel.__init__))
report["WhisperModel.transcribe"] = str(inspect.signature(whisperx.asr.WhisperModel.transcribe))
report["DiarizationPipeline.__init__"] = str(inspect.signature(whisperx.diarize.DiarizationPipeline.__init__))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    t0 = time.perf_counter()
    import pyannote.audio  # noqa: E402

    report["pyannote_import_sec"] = round(time.perf_counter() - t0, 2)
    report["pyannote_warnings"] = [str(w.message)[:120] for w in caught]

from pyannote.audio import Pipeline  # noqa: E402

report["pyannote"] = pyannote.audio.__version__
report["Pipeline.from_pretrained"] = str(inspect.signature(Pipeline.from_pretrained))
report["Pipeline.__call__"] = str(inspect.signature(Pipeline.__call__))

import faster_whisper  # noqa: E402

report["faster_whisper"] = faster_whisper.__version__
report["WhisperModel(fw).__init__"] = str(inspect.signature(faster_whisper.WhisperModel.__init__))
report["WhisperModel(fw).transcribe"] = str(inspect.signature(faster_whisper.WhisperModel.transcribe))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print("written", OUT)
