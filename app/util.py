"""Small shared helpers: timestamps, atomic JSON IO, ids."""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------- time helpers
def now_iso() -> str:
    """UTC timestamp, second precision, e.g. 2026-11-03T08:21:05Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hhmmss(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS (hours are not truncated)."""
    if seconds is None:
        return "--:--:--"
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float | None) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    seconds = max(0.0, float(seconds or 0.0))
    ms_total = int(round(seconds * 1000))
    h, rem = divmod(ms_total, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def human_bytes(n: int | float | None) -> str:
    if not n:
        return "0 B"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def slugify(text: str, maxlen: int = 40) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "-", text.strip())
    text = text.strip("-.")[:maxlen]
    return text or "project"


def new_project_id(name: str) -> str:
    """Filesystem-safe, sortable id: 20261103-082105-a1b2c3"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


# ------------------------------------------------------------------- JSON IO
def _atomic_replace(tmp_path: Path, final_path: Path) -> None:
    os.replace(tmp_path, final_path)


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Write JSON atomically (UTF-8, unescaped CJK)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=indent)
            fh.write("\n")
        _atomic_replace(Path(tmp), path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        _atomic_replace(Path(tmp), path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_text(path: str | Path, default: str = "") -> str:
    path = Path(path)
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def sha256_of_file(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
