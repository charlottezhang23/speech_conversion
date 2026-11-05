"""One-off environment introspection (not part of the app)."""
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

report = {}


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except Exception as exc:  # pragma: no cover
        return f"<{exc}>"


try:
    import torch

    report["torch"] = torch.__version__
    report["torch_cuda"] = torch.cuda.is_available()
except Exception as exc:
    report["torch_error"] = str(exc)

try:
    import whisperx

    report["whisperx"] = getattr(whisperx, "__version__", "?")
    report["load_model"] = sig(whisperx.load_model)
    report["load_align_model"] = sig(whisperx.load_align_model)
    report["align"] = sig(whisperx.align)
    report["transcribe"] = sig(whisperx.asr.WhisperModel.transcribe)
    report["whisper_model_init"] = sig(whisperx.asr.WhisperModel.__init__)
    report["load_audio"] = sig(whisperx.audio.load_audio)
except Exception as exc:
    report["whisperx_error"] = f"{type(exc).__name__}: {exc}"

try:
    import pyannote.audio
    from pyannote.audio import Pipeline

    report["pyannote"] = getattr(pyannote.audio, "__version__", "?")
    report["pipeline_from_pretrained"] = sig(Pipeline.from_pretrained)
    report["pipeline_call"] = sig(Pipeline.__call__)
except Exception as exc:
    report["pyannote_error"] = f"{type(exc).__name__}: {exc}"

try:
    import torchcodec

    report["torchcodec"] = getattr(torchcodec, "__version__", "?")
except Exception as exc:
    report["torchcodec_error"] = f"{type(exc).__name__}: {exc}"

try:
    import ctranslate2

    report["ctranslate2"] = ctranslate2.__version__
except Exception as exc:
    report["ctranslate2_error"] = str(exc)

try:
    import fastapi
    import uvicorn
    import soundfile
    import docx
    import httpx

    report["fastapi"] = fastapi.__version__
    report["uvicorn"] = uvicorn.__version__
    report["soundfile"] = soundfile.__version__
    report["python-docx"] = docx.__version__ if hasattr(docx, "__version__") else "?"
    report["httpx"] = httpx.__version__
except Exception as exc:
    report["web_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(report, ensure_ascii=False, indent=2))
