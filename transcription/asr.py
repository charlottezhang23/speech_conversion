"""WhisperX ASR wrapper (CPU / int8 by default)."""
from __future__ import annotations

import gc
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any

from app.config import MODELS_DIR, ensure_ffmpeg_on_path

logger = logging.getLogger("interviewai.asr")

WHISPER_CACHE = MODELS_DIR / "whisper"

ALIGN_MODELS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "en": "facebook/wav2vec2-large-960h-lv60-self",
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
}

# Whisper decodes an `initial_prompt` as if it were speech when the audio is noisy or
# silent, producing segments like "请用简体中文并加上标点符号". We therefore never set a
# language prompt automatically; `transcription/clean.py` also strips leftovers from
# transcripts that were produced by older versions of this app.
ASR_PROMPT_VERSION = "v2-no-prompt"


class AsrError(RuntimeError):
    pass


def set_threads(threads: int | None) -> None:
    if not threads or threads <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch

        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(max(1, int(threads) // 2))
        except RuntimeError:
            pass
    except ImportError:
        pass


def available() -> tuple[bool, str]:
    """Check that whisperx + torch + ctranslate2 import cleanly."""
    try:
        ensure_ffmpeg_on_path()
        import torch  # noqa: F401
        import whisperx  # noqa: F401
        import ctranslate2

        return True, f"whisperx OK, ctranslate2 {ctranslate2.__version__}, torch {torch.__version__}"
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


class AsrEngine:
    """Loads one WhisperX model and transcribes WAV chunks."""

    def __init__(
        self,
        model_name: str = "medium",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        threads: int | None = None,
        vad_method: str = "silero",
        initial_prompt: str | None = None,
        beam_size: int = 5,
        language: str | None = None,
    ) -> None:
        ensure_ffmpeg_on_path()
        set_threads(threads)
        try:
            import whisperx
        except Exception as exc:  # pragma: no cover
            raise AsrError(f"无法导入 whisperx：{exc}") from exc

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.language = None if language in (None, "", "auto") else language
        WHISPER_CACHE.mkdir(parents=True, exist_ok=True)

        # whisperx 3.8.6 defaults `without_timestamps=True`; segment boundaries come
        # from the VAD pass, which is what we want for timestamped transcripts.
        asr_options: dict[str, Any] = {
            "beam_size": int(beam_size),
            "best_of": int(beam_size),
            "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "condition_on_previous_text": False,  # avoids runaway repetition on long audio
        }
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt

        t0 = time.perf_counter()
        try:
            self.model = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type,
                asr_options=asr_options,
                language=self.language,
                vad_method=vad_method,
                download_root=str(WHISPER_CACHE),
                threads=int(threads or 4),
            )
        except Exception as exc:
            msg = str(exc)
            hint = ""
            if any(k in msg for k in ("Connection", "resolve", "offline", "timed out")):
                hint = "（首次使用需要联网下载 Whisper 权重，之后可离线运行）"
            raise AsrError(f"加载 Whisper 模型 {model_name} 失败：{exc}{hint}") from exc
        self.load_sec = round(time.perf_counter() - t0, 1)
        logger.info("WhisperX 模型 %s 载入完成 %.1fs", model_name, self.load_sec)

    # ---------------------------------------------------------------- asr
    def transcribe_chunk(
        self,
        wav_path: str | Path,
        *,
        language: str | None = None,
        batch_size: int = 8,
        progress: Any = None,
    ) -> dict[str, Any]:
        """Transcribe one WAV chunk; returns whisperx' result dict."""
        lang = None if language in (None, "", "auto") else language
        if self.language and lang is None:
            lang = self.language
        try:
            return self.model.transcribe(
                str(wav_path),
                batch_size=max(1, int(batch_size)),
                language=lang,
                task="transcribe",
                chunk_size=30,
                print_progress=False,
                verbose=False,
                progress_callback=progress,
            )
        except TypeError:
            return self.model.transcribe(str(wav_path), batch_size=batch_size, language=lang)
        except Exception as exc:
            raise AsrError(f"识别失败：{exc}") from exc

    def align(self, segments: list[dict], wav_path: str | Path, language: str) -> list[dict]:
        """Optional forced alignment (word timestamps) via wav2vec2."""
        return align_segments(segments, wav_path, language, device=self.device)

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def align_segments(
    segments: list[dict], wav_path: str | Path, language: str, *, device: str = "cpu"
) -> dict[str, Any]:
    """Force-align known text onto the audio with wav2vec2 (word-level timestamps).

    Loads the alignment model on demand, so callers do not need a Whisper model.
    """
    import gc

    import whisperx

    model_name = ALIGN_MODELS.get(language)
    if not model_name:
        raise AsrError(f"语言 {language} 没有配置对齐模型（支持 zh / ja / en / ko）")
    ensure_ffmpeg_on_path()
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device, model_name=model_name, model_dir=str(WHISPER_CACHE)
    )
    try:
        return whisperx.align(segments, align_model, metadata, str(wav_path), device, return_char_alignments=False)
    finally:
        del align_model
        gc.collect()


def normalize_segments(raw: dict[str, Any], offset: float = 0.0) -> list[dict[str, Any]]:
    """Convert a WhisperX result into plain segment dicts with a time offset."""
    out: list[dict[str, Any]] = []
    for seg in raw.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0) + offset
        end = float(seg.get("end") or start) + offset
        if end < start:
            end = start
        words = []
        for w in seg.get("words") or []:
            if w.get("start") is None:
                continue
            words.append(
                {
                    "word": w.get("word", ""),
                    "start": round(float(w["start"]) + offset, 3),
                    "end": round(float(w.get("end", w["start"])) + offset, 3),
                    "score": round(float(w["score"]), 3) if w.get("score") is not None else None,
                }
            )
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "words": words,
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
            }
        )
    return out
