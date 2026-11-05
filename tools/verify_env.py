"""Environment self-check used by install.ps1 / start.ps1.

Prints a human-readable report in Chinese. Never prints the HF token itself.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def line(label: str, value: str, ok: bool | None = None) -> None:
    mark = " " if ok is None else ("✔" if ok else "✘")
    print(f"    {mark} {label:<22} {value}")


def main() -> int:
    print("\n    环境自检")
    print("    " + "-" * 60)

    import platform

    line("系统", f"{platform.system()} {platform.release()} ({platform.version()})")
    line("Python", f"{sys.version.split()[0]}  {sys.executable}")
    line("CPU 逻辑核心", str(os.cpu_count()))

    from app.config import find_ffmpeg, get_hf_token, load_settings

    ff = find_ffmpeg()
    line("FFmpeg", str(ff) if ff else "未找到（请运行 install.ps1）", bool(ff))

    packages = {}
    for name in ("torch", "whisperx", "faster-whisper", "ctranslate2", "pyannote.audio", "fastapi", "python-docx", "soundfile", "torchcodec"):
        try:
            from importlib.metadata import version

            packages[name] = version(name)
        except Exception:
            packages[name] = None
    for name, ver in packages.items():
        line(name, ver or "未安装", bool(ver))

    # import checks (expensive, but definitive)
    try:
        import torch

        line("torch 运行时", f"{torch.__version__} · CUDA={torch.cuda.is_available()}")
    except Exception as exc:
        line("torch 运行时", f"失败：{exc}", False)

    try:
        import ctranslate2

        line("ctranslate2 CPU", f"{ctranslate2.__version__} ({ctranslate2.get_supported_compute_types('cpu')})")
    except Exception as exc:
        line("ctranslate2 CPU", f"失败：{exc}", False)

    whisperx_ok = True
    try:
        import whisperx  # noqa: F401
    except Exception as exc:
        whisperx_ok = False
        line("whisperx 导入", f"失败：{exc}", False)
    if whisperx_ok:
        line("whisperx 导入", "正常", True)

    pyannote_ok = True
    try:
        import pyannote.audio  # noqa: F401

        line("pyannote 导入", "正常", True)
    except Exception as exc:
        pyannote_ok = False
        line("pyannote 导入", f"失败：{exc}", False)

    try:
        import torchcodec  # noqa: F401

        line("torchcodec 导入", "正常", True)
    except Exception as exc:
        line("torchcodec 导入", f"不可用（本应用不依赖它解码音频）：{exc}", False)

    token = get_hf_token()
    line("Hugging Face Token", "已配置（内容不显示）" if token else "未配置（说话人分离需要）", bool(token))

    settings = load_settings()
    try:
        import requests

        r = requests.get(settings["ollama_url"].rstrip("/") + "/api/tags", timeout=3)
        models = [m.get("name") for m in r.json().get("models", [])] if r.ok else []
        line("Ollama", f"{settings['ollama_url']} · {len(models)} 个模型 {models[:4]}", bool(models))
    except Exception as exc:
        line("Ollama", f"未连接：{exc}", False)

    models_dir = ROOT / "models" / "whisper"
    cached = sorted(p.name for p in models_dir.glob("*") if p.is_dir()) if models_dir.exists() else []
    line("Whisper 模型缓存", ", ".join(cached) if cached else "空（首次转录会自动下载）")

    print("    " + "-" * 60)
    summary = {
        "ffmpeg": bool(ff),
        "whisperx": whisperx_ok,
        "pyannote": pyannote_ok,
        "hf_token": bool(token),
    }
    print("    " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
