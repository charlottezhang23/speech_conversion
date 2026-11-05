"""Merge ASR segments with diarization turns into speaker-attributed utterances."""
from __future__ import annotations

from typing import Any, Iterable, Sequence

MAX_MERGE_GAP = 2.0       # seconds of silence still merged into one utterance
MIN_SPLIT_PART = 0.6      # seconds; do not split a segment into tiny pieces
MIN_SPLIT_SEGMENT = 1.2   # seconds; never split shorter segments
MIN_WORD_RUN = 0.35       # seconds; shorter diarization runs are treated as jitter
MAX_WORD_SEC = 1.0        # seconds; longer word spans are treated as silence-padded


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _dominant_speaker(
    start: float, end: float, turns: Sequence[tuple[float, float, str]]
) -> tuple[str | None, float, dict[str, float]]:
    """Speaker with the largest time overlap in [start, end]."""
    weights: dict[str, float] = {}
    for t_start, t_end, label in turns:
        if t_end <= start or t_start >= end:
            continue
        ov = _overlap(start, end, t_start, t_end)
        if ov > 0:
            weights[label] = weights.get(label, 0.0) + ov
    if not weights:
        return None, 0.0, {}
    label, dur = max(weights.items(), key=lambda kv: kv[1])
    covered = sum(weights.values())
    return label, (dur / max(1e-6, end - start)), (weights if covered else {})


def _nearest_speaker(
    start: float, end: float, turns: Sequence[tuple[float, float, str]], max_gap: float = 2.0
) -> str | None:
    mid = (start + end) / 2.0
    best: tuple[float, str] | None = None
    for t_start, t_end, label in turns:
        dist = 0.0 if (t_start <= mid <= t_end) else min(abs(mid - t_start), abs(mid - t_end))
        if dist <= max_gap and (best is None or dist < best[0]):
            best = (dist, label)
    return best[1] if best else None


def _words_in(words: Sequence[dict], start: float, end: float) -> list[dict]:
    out = []
    for w in words:
        w_start = w.get("start")
        w_end = w.get("end", w_start)
        if w_start is None:
            continue
        if _overlap(float(w_start), float(w_end or w_start), start, end) > 0:
            out.append(w)
    return out


def _split_by_time(segment: dict, boundary: float) -> tuple[dict, dict]:
    """Split a segment at `boundary`, using word timings when available."""
    start, end = float(segment["start"]), float(segment["end"])
    words = segment.get("words") or []
    text = segment.get("text", "")
    left_words = []
    right_words = []
    if words:
        for w in words:
            w_start = float(w.get("start") or start)
            w_end = float(w.get("end") or w_start)
            # a word belongs to the side its middle point falls on, so a boundary that
            # lands inside a word does not drag the whole word into the next piece
            (left_words if (w_start + w_end) / 2.0 < boundary else right_words).append(w)
    left_text = "".join(w.get("word", "") for w in left_words).strip()
    right_text = "".join(w.get("word", "") for w in right_words).strip()
    if not left_text or not right_text:
        ratio = (boundary - start) / max(1e-6, end - start)
        cut = max(1, min(len(text) - 1, int(round(len(text) * ratio))))
        left_text, right_text = text[:cut].strip(), text[cut:].strip()
        left_words, right_words = [], []
    left = {**segment, "end": round(boundary, 3), "text": left_text, "words": left_words}
    right = {**segment, "start": round(boundary, 3), "text": right_text, "words": right_words}
    return left, right


def _word_speaker(word: dict, turns: Sequence[tuple[float, float, str]]) -> str | None:
    """Speaker with the largest overlap with this word."""
    w_start = float(word.get("start") or 0.0)
    w_end = float(word.get("end") or w_start)
    label, _ratio, _weights = _dominant_speaker(w_start, max(w_end, w_start + 1e-3), turns)
    if label:
        return label
    return _nearest_speaker(w_start, w_end, turns)


def _speech_regions(turns: Sequence[tuple[float, float, str]]) -> list[tuple[float, float]]:
    """Union of all diarization turns (speech activity, labels ignored)."""
    spans = sorted((s, e) for s, e, _ in turns)
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _trim_words(
    words: Sequence[dict], regions: Sequence[tuple[float, float]], cap: float = MAX_WORD_SEC
) -> list[dict]:
    """Shrink word spans that swallowed a silence gap.

    wav2vec2 alignment stretches a character across following silence when the ASR
    segment is much longer than the text, e.g. a single「了」 owning 1.8 s. Such a
    word then wins a speaker vote it never took part in, so we intersect it with the
    speech regions and keep the overlapping piece (largest, or the one holding its
    middle). Spans are kept monotonic and never extended.
    """
    if not regions:
        return [dict(w) for w in words]
    out: list[dict] = []
    prev_end = float("-inf")
    for word in words:
        start = float(word.get("start") or 0.0)
        end = float(word.get("end") or start)
        if end - start > cap:
            expected = max(start, prev_end if prev_end != float("-inf") else start)
            pieces = [
                (max(start, rs), min(end, re))
                for rs, re in regions
                if min(end, re) - max(start, rs) > 0.02
            ]
            if pieces:
                # the word is spoken right after the previous one, so keep the speech
                # piece nearest to that expected position (not the longest one)
                pick = min(pieces, key=lambda p: (abs(p[0] - expected), -(p[1] - p[0])))
                start, end = pick
            else:
                end = start + cap
        if start < prev_end:
            start = prev_end
        if end <= start:
            end = start + 0.06
        prev_end = end
        out.append({**word, "start": round(start, 3), "end": round(end, 3)})
    return out


def _word_runs(
    segment: dict, turns: Sequence[tuple[float, float, str]], min_run: float = MIN_WORD_RUN
) -> list[dict]:
    """Group the segment's words into consecutive same-speaker runs (exact word bounds)."""
    raw_words = [w for w in (segment.get("words") or []) if w.get("start") is not None]
    if not raw_words or not turns:
        return []
    words = _trim_words(raw_words, _speech_regions(turns))
    labelled: list[tuple[str | None, dict]] = []
    for w in words:
        token = str(w.get("word") or "")
        if not token.strip() and labelled:
            # punctuation / spacing token follows the previous word's speaker
            labelled.append((labelled[-1][0], w))
        else:
            labelled.append((_word_speaker(w, turns), w))

    runs: list[dict] = []
    for label, w in labelled:
        if runs and runs[-1]["speaker"] == label:
            runs[-1]["words"].append(w)
        else:
            runs.append({"speaker": label, "words": [w]})

    def run_span(run: dict) -> float:
        first = run["words"][0]
        last = run["words"][-1]
        return float(last.get("end") or last["start"]) - float(first["start"])

    # diarization jitter produces 0.1–0.4 s "turns": fold those into the longer neighbour
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if run_span(run) >= min_run:
                continue
            prev_run = runs[i - 1] if i > 0 else None
            next_run = runs[i + 1] if i + 1 < len(runs) else None
            target = None
            if prev_run and next_run:
                target = prev_run if run_span(prev_run) >= run_span(next_run) else next_run
            else:
                target = prev_run or next_run
            if target is None:
                continue
            if target is prev_run:
                prev_run["words"].extend(run["words"])
            else:
                next_run["words"] = run["words"] + next_run["words"]
            runs.pop(i)
            changed = True
            break
    return runs


def assign_speakers(
    segments: Iterable[dict],
    turns: Sequence[tuple[float, float, str]],
    *,
    agree_threshold: float = 0.55,
) -> list[dict]:
    """Attach a `speaker` label to each ASR segment, splitting on speaker changes."""
    turns = sorted(turns, key=lambda t: t[0])
    out: list[dict] = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        if end - start <= 0:
            end = start + 0.4

        # candidate split points: diarization boundaries strictly inside the segment
        internal = sorted({round(t[0], 3) for t in turns if start + 0.15 < t[0] < end - 0.15})
        runs = _word_runs(segment, turns) if (segment.get("words") and internal) else []
        forced: list[str | None] = []
        if len(runs) > 1:
            # word timestamps available: cut exactly between words, no ratio guessing
            pieces = []
            for run in runs:
                run_words = run["words"]
                r_start = round(float(run_words[0]["start"]), 3)
                r_end = round(float(run_words[-1].get("end") or run_words[-1]["start"]), 3)
                r_text = "".join(str(w.get("word") or "") for w in run_words).strip()
                if not r_text:
                    continue
                pieces.append({**segment, "start": r_start, "end": r_end, "text": r_text, "words": run_words})
                forced.append(run["speaker"])
        else:
            pieces = [dict(segment)]
            for boundary in internal:
                new_pieces: list[dict] = []
                for piece in pieces:
                    p_start, p_end = float(piece["start"]), float(piece["end"])
                    if not (p_start + MIN_SPLIT_PART < boundary < p_end - MIN_SPLIT_PART):
                        new_pieces.append(piece)
                        continue
                    if p_end - p_start < MIN_SPLIT_SEGMENT:
                        new_pieces.append(piece)
                        continue
                    left, right = _split_by_time(piece, boundary)
                    if not left["text"] or not right["text"]:
                        new_pieces.append(piece)
                    else:
                        new_pieces.extend([left, right])
                pieces = new_pieces
            forced = [None] * len(pieces)

        for idx_piece, piece in enumerate(pieces):
            p_start, p_end = float(piece["start"]), float(piece["end"])
            speaker, ratio, weights = _dominant_speaker(p_start, p_end, turns)
            if forced[idx_piece]:
                speaker = forced[idx_piece]
                ratio = weights.get(speaker, 0.0) / max(1e-6, p_end - p_start)
            flags: list[str] = []
            if speaker is None:
                speaker = _nearest_speaker(p_start, p_end, turns)
                if speaker:
                    flags.append("nearest_speaker")
                else:
                    speaker = "SPEAKER_00"
                    flags.append("no_diarization")
            if ratio and ratio < agree_threshold:
                flags.append("speaker_uncertain")
            if len([w for w, d in weights.items() if d > 0.15 * max(1e-6, p_end - p_start)]) > 1:
                flags.append("overlap")

            words = piece.get("words") or []
            probs = [float(w.get("score", 1.0)) for w in words if w.get("score") is not None]
            if probs and (sum(probs) / len(probs)) < 0.55:
                flags.append("low_confidence")
            if not str(piece.get("text", "")).strip():
                continue

            out.append(
                {
                    "start": round(p_start, 3),
                    "end": round(p_end, 3),
                    "text": str(piece.get("text", "")).strip(),
                    "speaker": speaker,
                    "speaker_confidence": round(float(ratio), 3) if ratio else 0.0,
                    "flags": flags,
                    "words": words,
                }
            )
    return out


def build_utterances(
    segments: Sequence[dict],
    *,
    max_gap: float = MAX_MERGE_GAP,
    max_chars: int = 500,
    max_duration: float = 60.0,
) -> list[dict]:
    """Group consecutive same-speaker segments into editable utterances."""
    utterances: list[dict] = []
    current: dict | None = None
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        if (
            current
            and current["speaker"] == seg["speaker"]
            and float(seg["start"]) - float(current["end"]) <= max_gap
            and len(current["text"]) + len(text) <= max_chars
            and float(seg["end"]) - float(current["start"]) <= max_duration
        ):
            current["end"] = seg["end"]
            current["text"] = (current["text"] + " " + text).strip() if _needs_space(current["text"], text) else current["text"] + text
            current["segments"].append(seg)
            for flag in seg.get("flags") or []:
                if flag not in current["flags"]:
                    current["flags"].append(flag)
            current["speaker_confidence"] = min(current["speaker_confidence"], seg.get("speaker_confidence", 1.0))
        else:
            current = {
                "id": f"u{len(utterances) + 1:04d}",
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": text,
                "speaker_confidence": seg.get("speaker_confidence", 1.0),
                "flags": list(seg.get("flags") or []),
                "segments": [seg],
            }
            utterances.append(current)
    return utterances


def _needs_space(left: str, right: str) -> bool:
    """CJK text should not gain spaces; latin text should."""
    if not left or not right:
        return False
    last = left[-1]
    first = right[0]
    cjk = lambda ch: "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"  # noqa: E731
    return not (cjk(last) or cjk(first))


def speaker_stats(utterances: Sequence[dict]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for u in utterances:
        s = stats.setdefault(u["speaker"], {"turns": 0, "speech_sec": 0.0, "chars": 0})
        s["turns"] += 1
        s["speech_sec"] = round(s["speech_sec"] + max(0.0, float(u["end"]) - float(u["start"])), 1)
        s["chars"] += len(u.get("text") or "")
    return stats
