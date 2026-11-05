"""Stepwise import probe: writes progress to logs/probe.txt so hangs are visible."""
from __future__ import annotations

import sys
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "probe.txt"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    log(f"python {sys.version.split()[0]}")
    t0 = time.perf_counter()
    import torch

    log(f"torch {torch.__version__} in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    import ctranslate2

    log(f"ctranslate2 {ctranslate2.__version__} in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    import whisperx

    log(f"whisperx in {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    try:
        import torchcodec

        log(f"torchcodec {getattr(torchcodec, '__version__', '?')} in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        log(f"torchcodec FAILED {type(exc).__name__}: {exc}")
    t0 = time.perf_counter()
    try:
        import pyannote.audio

        log(f"pyannote.audio in {time.perf_counter() - t0:.1f}s")
    except Exception as exc:
        log(f"pyannote FAILED {type(exc).__name__}: {exc}")
    import inspect

    log("load_model" + str(inspect.signature(whisperx.load_model)))
    log("transcribe" + str(inspect.signature(whisperx.asr.WhisperModel.transcribe)))
    try:
        from pyannote.audio import Pipeline

        log("from_pretrained" + str(inspect.signature(Pipeline.from_pretrained)))
        log("call" + str(inspect.signature(Pipeline.__call__)))
    except Exception as exc:
        log(f"pipeline sig FAILED {type(exc).__name__}: {exc}")
    log("DONE")


if __name__ == "__main__":
    main()
