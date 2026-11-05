"""Audio handling via the FFmpeg CLI (no librosa / torchcodec dependency)."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from app.config import find_ffmpeg, find_ffprobe

SAMPLE_RATE = 16000


class FfmpegError(RuntimeError):
    pass


def _run(args: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    creationflags = 0
    try:
        import sys

        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    except Exception:  # pragma: no cover
        pass
    return subprocess.run(
        args,
        capture_output=True,
        timeout=timeout,
        creationflags=creationflags,
    )


def ffmpeg_bin() -> str:
    exe = find_ffmpeg()
    if not exe:
        raise FfmpegError("未找到 FFmpeg。请运行 install.ps1 安装，或设置 INTERVIEWAI_FFMPEG。")
    return str(exe)


def ffprobe_bin() -> str:
    exe = find_ffprobe()
    if not exe:
        raise FfmpegError("未找到 FFprobe。请运行 install.ps1 安装。")
    return str(exe)


@dataclass
class AudioInfo:
    duration_sec: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    bit_rate: int | None
    format: str | None
    size_bytes: int

    def as_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "codec": self.codec,
            "bit_rate": self.bit_rate,
            "format": self.format,
            "size_bytes": self.size_bytes,
        }


def probe(path: str | Path) -> AudioInfo:
    path = Path(path)
    if not path.exists():
        raise FfmpegError("音频文件不存在")
    proc = _run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=120,
    )
    if proc.returncode != 0:
        raise FfmpegError("ffprobe 无法解析该文件: " + proc.stderr.decode("utf-8", "replace")[:400])
    data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio:
        raise FfmpegError("该文件不包含音频轨道")
    duration = fmt.get("duration") or audio.get("duration")
    return AudioInfo(
        duration_sec=float(duration) if duration not in (None, "N/A") else None,
        sample_rate=int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        channels=int(audio["channels"]) if audio.get("channels") else None,
        codec=audio.get("codec_name"),
        bit_rate=int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        format=fmt.get("format_name"),
        size_bytes=path.stat().st_size,
    )


def normalize(src: str | Path, dst: str | Path) -> Path:
    """Decode any input to 16 kHz mono PCM WAV (what Whisper and pyannote want)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    if proc.returncode != 0 or not dst.exists():
        raise FfmpegError("音频转码失败: " + proc.stderr.decode("utf-8", "replace")[:400])
    return dst


_SIL_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SIL_END = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silences(wav: str | Path, noise_db: int = -35, min_silence: float = 0.45) -> list[tuple[float, float]]:
    """Return [(start, end)] silent regions, used to cut chunks on natural pauses."""
    proc = _run(
        [
            ffmpeg_bin(),
            "-hide_banner",
            "-nostats",
            "-i",
            str(wav),
            "-af",
            f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f",
            "null",
            "-",
        ]
    )
    text = proc.stderr.decode("utf-8", "replace")
    silences: list[tuple[float, float]] = []
    pending: float | None = None
    for line in text.splitlines():
        m = _SIL_START.search(line)
        if m:
            pending = float(m.group(1))
            continue
        m = _SIL_END.search(line)
        if m and pending is not None:
            silences.append((pending, float(m.group(1))))
            pending = None
    return silences


def plan_chunks(
    duration_sec: float,
    silences: Iterable[tuple[float, float]],
    target_sec: float = 600.0,
    min_tail_sec: float = 30.0,
) -> list[tuple[float, float]]:
    """Split [0, duration] into chunks of ~target_sec, preferring a silence cut.

    Never returns a trailing chunk shorter than min_tail_sec when avoidable.
    """
    duration_sec = float(duration_sec)
    if duration_sec <= 0:
        return [(0.0, 0.0)]
    if duration_sec <= target_sec * 1.35:
        return [(0.0, duration_sec)]

    mids = [((a + b) / 2.0) for a, b in silences]
    cuts: list[float] = []
    start = 0.0
    while duration_sec - start > target_sec * 1.35:
        target = start + target_sec
        window = target_sec * 0.25
        candidates = [m for m in mids if target - window <= m <= target + window and m > start + 60]
        cut = min(candidates, key=lambda m: abs(m - target)) if candidates else target
        cuts.append(cut)
        start = cut
    bounds = [0.0] + cuts + [duration_sec]
    chunks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    # merge a too-short tail into the previous chunk
    if len(chunks) > 1 and (chunks[-1][1] - chunks[-1][0]) < min_tail_sec:
        prev = chunks[-2]
        chunks = chunks[:-2] + [(prev[0], chunks[-1][1])]
    return [(round(a, 3), round(b, 3)) for a, b in chunks]


def slice_wav(src: str | Path, start: float, end: float, dst: str | Path) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            ffmpeg_bin(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-t",
            f"{max(0.05, end - start):.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    if proc.returncode != 0 or not dst.exists():
        raise FfmpegError("切分音频失败: " + proc.stderr.decode("utf-8", "replace")[:400])
    return dst


def load_waveform(path: str | Path):
    """Load a 16 kHz mono WAV as float32 numpy array (used by pyannote)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise FfmpegError(f"期望 {SAMPLE_RATE} Hz，实际 {sr} Hz——请先执行标准化转码")
    return mono


def extract_segment_wav(src: str | Path, start: float, end: float, dst: str | Path) -> Path:
    return slice_wav(src, start, end, dst)


def run_cancellable(args: list[str], on_poll: Callable[[], bool] | None = None, poll_sec: float = 0.5) -> int:
    """Run a subprocess, polling a cancel callback. Returns the exit code."""
    import sys
    import time

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creationflags)
    while proc.poll() is None:
        if on_poll and on_poll():
            proc.kill()
            return -1
        time.sleep(poll_sec)
    return proc.returncode or 0
