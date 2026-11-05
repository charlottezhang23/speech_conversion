"""pyannote.audio speaker diarization (local, Community-1 model)."""
from __future__ import annotations

import gc
import logging
import re
import time
from pathlib import Path
from typing import Any, Callable

from app.config import MODELS_DIR, get_hf_token

logger = logging.getLogger("interviewai.diarization")

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


class DiarizationError(RuntimeError):
    pass


def _to_torch_device(device: object):
    """pyannote 4 `Pipeline.to()` requires a torch.device, not a string."""
    import torch

    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _gated_hint(exc: Exception, model_name: str) -> str:
    """Name the exact gated repos mentioned in a pyannote/HF download error."""
    msg = str(exc)
    repos = set(re.findall(r"huggingface\.co/([\w.\-]+/[\w.\-]+)", msg))
    repos |= set(re.findall(r"Model from ([\w.\-]+/[\w.\-]+)", msg))
    gated = ("gated" in msg.lower()) or ("403" in msg) or bool(repos)
    if not gated:
        return "\n请检查：1) 网络可访问 huggingface.co；2) 首次运行需要联网下载模型。"
    if not repos:
        repos = {model_name}
    lines = [
        "",
        f"该流水线还需要下列仓库的访问权限（接受条款不等于全链路授权，需逐个确认）：",
    ]
    for repo in sorted(repos):
        lines.append(f"  - https://huggingface.co/{repo}")
    lines.append("每个页面都要点 Agree / Request access，状态显示 authorized 后再重试。")
    return "\n".join(lines)


def available() -> tuple[bool, str]:
    try:
        import pyannote.audio  # noqa: F401

        return True, f"pyannote.audio {getattr(pyannote.audio, '__version__', '?')}"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"


class DiarizationEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        token = get_hf_token()
        if not token:
            raise DiarizationError(
                "缺少 Hugging Face Token。说话人分离模型需要先在 Hugging Face 上同意条款，"
                "然后在设置页填入 Token（只会保存在本机 secrets/hf_token.txt）。"
            )
        try:
            from pyannote.audio import Pipeline
        except Exception as exc:  # pragma: no cover
            raise DiarizationError(f"无法导入 pyannote.audio：{exc}") from exc

        self.model_name = model_name
        self.device = _to_torch_device(device)
        cache = Path(cache_dir or (MODELS_DIR / "pyannote"))
        cache.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            self.pipeline = Pipeline.from_pretrained(model_name, token=token, cache_dir=str(cache))
        except Exception as exc:  # gated repo / network / revision issues
            raise DiarizationError(
                f"加载说话人分离模型失败（{model_name}）：{exc}"
                + _gated_hint(exc, model_name)
            ) from exc
        if self.pipeline is None:
            raise DiarizationError("pyannote 返回空模型，请检查 Token 权限。")
        self.pipeline.to(self.device)
        logger.info("pyannote 模型载入完成 %.1fs", time.perf_counter() - t0)

    # ----------------------------------------------------------------- diarize
    def diarize(
        self,
        wav_path: str | Path,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[dict[str, Any]]:
        import numpy as np
        import soundfile as sf
        import torch

        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        waveform = torch.from_numpy(np.ascontiguousarray(mono[None, :]))
        audio_input = {"waveform": waveform, "sample_rate": int(sr)}

        hook = None
        if progress is not None:
            ranges = {"segmentation": (0.0, 55.0), "embeddings": (55.0, 95.0), "discrete_diarization": (80.0, 98.0)}
            last = [0.0]

            def hook(step_name, step_artifact, file=None, total=None, completed=None):  # noqa: ANN001
                if total and completed is not None:
                    lo, hi = ranges.get(step_name, (0.0, 98.0))
                    pct = lo + min(completed / total, 1.0) * (hi - lo)
                    if pct > last[0]:
                        last[0] = pct
                        progress(pct, f"说话人分离：{step_name} {completed}/{total}")

        kwargs: dict[str, Any] = {}
        if num_speakers:
            kwargs["num_speakers"] = int(num_speakers)
        else:
            if min_speakers:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers:
                kwargs["max_speakers"] = int(max_speakers)
        if hook is not None:
            kwargs["hook"] = hook

        t0 = time.perf_counter()
        try:
            try:
                output = self.pipeline(audio_input, **kwargs)
            except TypeError as exc:
                # 只有确实是 hook 参数不被接受时才重试，避免掩盖真正的错误
                if "hook" not in str(exc):
                    raise
                kwargs.pop("hook", None)
                output = self.pipeline(audio_input, **kwargs)
        finally:
            del waveform
            gc.collect()

        annotation = getattr(output, "speaker_diarization", output)
        if annotation is None:
            raise DiarizationError("pyannote 未返回 speaker_diarization（可能是模型版本不兼容）")
        turns: list[dict[str, Any]] = []
        try:
            itertracks = annotation.itertracks
        except AttributeError as exc:
            raise DiarizationError("pyannote 输出格式无法识别") from exc
        for turn, _, label in itertracks(yield_label=True):
            turns.append(
                {
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker": str(label),
                }
            )
        turns.sort(key=lambda t: t["start"])
        logger.info("diarization done in %.1fs, %d turns", time.perf_counter() - t0, len(turns))
        if progress:
            progress(100.0, f"说话人分离完成，共 {len(turns)} 段")
        return turns

    def unload(self) -> None:
        self.pipeline = None
        gc.collect()
