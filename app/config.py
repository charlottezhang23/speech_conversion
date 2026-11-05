"""Central configuration: paths, FFmpeg discovery, defaults, secrets.

Nothing here ever sends data off the machine. The Hugging Face token is read
from an environment variable or a local file and is never written to logs.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from app.util import read_json, write_json

ROOT = Path(__file__).resolve().parent.parent

APP_DIR = ROOT / "app"
STATIC_DIR = APP_DIR / "static"
PROJECTS_DIR = Path(os.environ.get("INTERVIEWAI_PROJECTS", ROOT / "projects"))
MODELS_DIR = Path(os.environ.get("INTERVIEWAI_MODELS", ROOT / "models"))
CONFIG_DIR = ROOT / "config"
SECRETS_DIR = ROOT / "secrets"
LOGS_DIR = ROOT / "logs"
TMP_DIR = ROOT / "tmp"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
TOKEN_FILE = SECRETS_DIR / "hf_token.txt"

for _d in (PROJECTS_DIR, MODELS_DIR, CONFIG_DIR, SECRETS_DIR, LOGS_DIR, TMP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Hugging Face cache lives inside the workspace (models/) so the project is portable.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODELS_DIR / "hf" / "hub"))
os.environ.setdefault("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
# Privacy / noise: no HF telemetry, no symlink warnings on Windows.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_SETTINGS: dict[str, Any] = {
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen3:8b",
    "ollama_num_ctx": 16384,
    "ollama_timeout_sec": 900,
    "default_asr_model": "medium",
    "default_compute_type": "int8",
    "default_threads": max(2, (os.cpu_count() or 4) - 4),
    "default_beam_size": 5,
    "default_batch_size": 8,
    "chunk_minutes": 10.0,
    "diarization_model": "pyannote/speaker-diarization-community-1",
    "diarize_enabled": True,
    "summary_chunk_chars": 4500,
    "verify_summary": True,
}


def load_settings() -> dict[str, Any]:
    stored = read_json(SETTINGS_FILE, {}) or {}
    merged = dict(DEFAULT_SETTINGS)
    if isinstance(stored, dict):
        merged.update({k: v for k, v in stored.items() if v is not None})
    return merged


def save_settings(patch: dict[str, Any]) -> dict[str, Any]:
    current = load_settings()
    for key, value in patch.items():
        if key in DEFAULT_SETTINGS:
            current[key] = value
    write_json(SETTINGS_FILE, current)
    return current


def set_hf_token(token: str | None) -> None:
    """Persist the HF token locally (secrets/). Empty string removes it."""
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    if token:
        TOKEN_FILE.write_text(token.strip(), encoding="utf-8")
    elif TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def get_hf_token() -> str | None:
    """Env var wins, then the local secrets file. Never logged."""
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env and env.strip():
        return env.strip()
    if TOKEN_FILE.exists():
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        return tok or None
    return None


def has_hf_token() -> bool:
    return bool(get_hf_token())


# ------------------------------------------------------------------ FFmpeg
def find_ffmpeg() -> Path | None:
    """Locate ffmpeg.exe: env override -> bundled tools/ -> PATH."""
    env = os.environ.get("INTERVIEWAI_FFMPEG")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe")
    candidates.append(ROOT / "tools" / "ffmpeg" / "ffmpeg.exe")
    for c in candidates:
        if c and c.exists():
            return c
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def find_ffprobe() -> Path | None:
    env = os.environ.get("INTERVIEWAI_FFPROBE")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "tools" / "ffmpeg" / "bin" / "ffprobe.exe")
    candidates.append(ROOT / "tools" / "ffmpeg" / "ffprobe.exe")
    for c in candidates:
        if c and c.exists():
            return c
    found = shutil.which("ffprobe")
    return Path(found) if found else None


def ensure_ffmpeg_on_path() -> Path | None:
    """Prepend the bundled FFmpeg directory to PATH (needed by whisperx/faster-whisper/torchcodec)."""
    exe = find_ffmpeg()
    if exe:
        bindir = str(exe.parent)
        parts = os.environ.get("PATH", "").split(os.pathsep)
        if bindir not in parts:
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:  # let torchcodec find avcodec-*.dll
                os.add_dll_directory(bindir)
            except OSError:
                pass
    return exe


# FFmpeg must be discoverable before torch/pyannote/torchcodec are imported.
ensure_ffmpeg_on_path()

# pyannote.audio 4 probes torchcodec on import. torchcodec 0.7 only supports
# FFmpeg <= 7, and there is no FFmpeg 7 shared Windows build to link against,
# so the probe always fails here. This app never uses torchcodec: audio is
# decoded by the FFmpeg CLI and pyannote receives an in-memory waveform.
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=r".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")


def env_report() -> dict[str, Any]:
    """Diagnostics for the Settings page (no secrets included)."""
    import platform

    ff = find_ffmpeg()
    fp = find_ffprobe()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "cpu_count": os.cpu_count(),
        "ffmpeg": str(ff) if ff else None,
        "ffprobe": str(fp) if fp else None,
        "workspace": str(ROOT),
        "projects_dir": str(PROJECTS_DIR),
        "models_dir": str(MODELS_DIR),
        "hf_token_set": has_hf_token(),
    }
