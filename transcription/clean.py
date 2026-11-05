"""Remove Whisper prompt-echo and repetition loops from ASR segments.

Why this exists
---------------
An `initial_prompt` such as "请用简体中文并加上标点符号" is sometimes *decoded*
as if it were speech, especially on silence or very noisy audio. The result is
segments that repeat the prompt (or fragments of it) instead of the recording.

The raw ASR output is never modified: chunks in `asr/chunks/*.json` keep
everything. This filter only decides what goes into the merged transcript.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Iterable

# Prompts that this app (or an older version of it) may have sent to Whisper.
ECHO_SOURCES = [
    "以下是一段普通话访谈录音，请用简体中文并加上标点符号。",
    "请用简体中文并加上标点符号。",
    "请用简体中文",
    "简体中文并加上标点符号",
    "并加上标点符号",
    "加上标点符号",
    "标点符号",
    "普通话访谈录音",
    "以下は日本語のインタビュー音声です。句読点を付けて書き起こしてください。",
    "句読点を付けて書き起こしてください",
    "書き起こしてください",
]

_NORM_RE = re.compile(r"[\s，,。．.、；;：:！!？?“”\"'（）()\[\]【】《》—\-…~～]+")
_EDGE_PUNCT = "，,。.、；;：:！!？? \t\n"

# Drop a segment when this share of its characters can be matched against a prompt.
DROP_ECHO_RATIO = 0.75
# Drop a segment when 6-grams repeat this much (Whisper "谢谢观看" style loops).
DROP_LOOP_RATIO = 0.55


def normalize(text: str) -> str:
    return _NORM_RE.sub("", text or "")


_PROMPTS_NORM = [normalize(p) for p in ECHO_SOURCES]


def echo_ratio(text: str) -> float:
    """Share of characters in `text` that also appear, in order, in some prompt."""
    t = normalize(text)
    if not t:
        return 0.0
    best = 0.0
    for prompt in _PROMPTS_NORM:
        if not prompt:
            continue
        blocks = difflib.SequenceMatcher(None, t, prompt).get_matching_blocks()
        matched = sum(b.size for b in blocks)
        best = max(best, matched / len(t))
        if best >= 0.99:
            break
    return best


def repetition_ratio(text: str, n: int = 6) -> float:
    """1 - (unique n-grams / total n-grams). High value means a repeating loop."""
    t = normalize(text)
    if len(t) < n * 3:
        return 0.0
    grams = [t[i : i + n] for i in range(len(t) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _strip_known_phrases(text: str) -> str:
    out = text
    for phrase in ECHO_SOURCES:
        out = out.replace(phrase, "")
    # also handle the punctuation-free variants that Whisper likes to emit
    for phrase in ("请用简体中文并加上标点符号", "简体中文并加上标点符号", "并加上标点符号"):
        out = out.replace(phrase, "")
    return out


def clean_text(text: str) -> tuple[str, str]:
    """Return (cleaned_text, action) where action is 'kept' | 'trimmed' | 'echo' | 'loop'."""
    raw = (text or "").strip()
    if not raw:
        return "", "echo"

    if repetition_ratio(raw) >= DROP_LOOP_RATIO:
        return "", "loop"

    ratio = echo_ratio(raw)
    if ratio >= DROP_ECHO_RATIO:
        return "", "echo"

    stripped = _strip_known_phrases(raw)
    if stripped != raw:
        stripped = stripped.strip(_EDGE_PUNCT)
        remaining = normalize(stripped)
        if len(remaining) < 2:
            return "", "echo"
        # 去掉提示词后只剩零星字符，同样视为残片
        if len(remaining) < max(3, len(normalize(raw)) * 0.25):
            return "", "echo"
        if echo_ratio(stripped) >= DROP_ECHO_RATIO:
            return "", "echo"
        return stripped, "trimmed"

    return raw, "kept"


def clean_segments(segments: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter prompt echoes / loops. The input dicts are not modified."""
    kept: list[dict[str, Any]] = []
    dropped_echo = 0
    dropped_loop = 0
    trimmed = 0
    examples: list[dict[str, Any]] = []

    for seg in segments:
        text = str(seg.get("text") or "")
        cleaned, action = clean_text(text)
        if action in ("echo", "loop"):
            if action == "echo":
                dropped_echo += 1
            else:
                dropped_loop += 1
            if len(examples) < 5:
                examples.append({"start": seg.get("start"), "text": text[:80], "reason": action})
            continue
        if action == "trimmed":
            trimmed += 1
        kept.append({**seg, "text": cleaned})

    stats = {
        "dropped_echo": dropped_echo,
        "dropped_loop": dropped_loop,
        "trimmed": trimmed,
        "filtered_total": dropped_echo + dropped_loop,
        "examples": examples,
    }
    return kept, stats
