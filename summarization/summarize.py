"""Map → reduce → compose → verify summary pipeline with resumable chunk cache."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from app.util import hhmmss, now_iso, read_json, write_json, write_text
from summarization import prompts
from summarization.chunking import chunk_utterances, format_evidence
from summarization.ollama_client import OllamaError, chat, extract_json

PROMPT_VERSION = "2026-11-03.1"
MAX_FACTS_JSON_CHARS = 14000


class SummaryError(RuntimeError):
    pass


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _chat_json(
    *,
    model: str,
    base_url: str,
    system: str,
    user: str,
    schema: dict[str, Any] | None,
    num_ctx: int,
    timeout_sec: float,
    on_token: Callable[[int], None] | None = None,
    retries: int = 2,
    max_tokens: int = 1600,
    should_cancel: Callable[[], bool] | None = None,
) -> Any:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        if should_cancel is not None and should_cancel():
            raise SummaryError("已取消")
        try:
            result = chat(
                model,
                messages,
                base_url=base_url,
                num_ctx=num_ctx,
                json_format=schema if schema else "json",
                timeout_sec=timeout_sec,
                on_token=on_token,
                max_tokens=max_tokens,
                should_cancel=should_cancel,
            )
            return extract_json(result.content)
        except (OllamaError, json.JSONDecodeError, SummaryError) as exc:
            if isinstance(exc, SummaryError) or "已取消" in str(exc):
                raise SummaryError("已取消") from exc
            last_error = exc
            if attempt >= retries:
                break
            messages = messages[:2] + [
                {
                    "role": "user",
                    "content": "上一条回复不是有效 JSON。请只输出符合要求的 JSON，不要任何解释或代码块标记。",
                }
            ]
    raise SummaryError(f"模型输出解析失败：{last_error}")


def summarize(
    *,
    utterances: list[dict[str, Any]],
    names: dict[str, str],
    meta: dict[str, Any],
    summary_dir: Path,
    template: str,
    model: str,
    base_url: str,
    num_ctx: int = 16384,
    timeout_sec: float = 900.0,
    max_chunk_chars: int = 4500,
    verify: bool = True,
    force: bool = False,
    progress: Callable[[float, str, str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate a structured summary. Chunk results are cached on disk for resume."""
    if template not in prompts.TEMPLATES:
        raise SummaryError(f"未知的纪要模板: {template}")
    if not utterances:
        raise SummaryError("逐字稿为空，无法生成纪要")

    def report(pct: float, stage: str, message: str) -> None:
        if progress:
            progress(pct, stage, message)
        if log:
            log(message)

    def check_cancel() -> None:
        if is_cancelled and is_cancelled():
            raise SummaryError("已取消")

    t_start = time.perf_counter()
    chunks = chunk_utterances(utterances, names, max_chars=max_chunk_chars)
    if not chunks:
        raise SummaryError("逐字稿为空，无法生成纪要")

    summary_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = summary_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    total_chars = sum(len(c.text) for c in chunks)
    report(2.0, "prepare", f"共 {len(chunks)} 个分段，约 {total_chars} 字，开始逐段抽取事实")

    # ---------------------------------------------------------------- map
    facts: list[dict[str, Any]] = []
    summaries: list[str] = []
    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        check_cancel()
        key = _hash(f"{PROMPT_VERSION}|{template}|{model}|{chunk.text}")
        cache_path = chunk_dir / f"{chunk.index:03d}.json"
        cached = None if force else read_json(cache_path)
        if isinstance(cached, dict) and cached.get("key") == key:
            payload = cached.get("payload") or {}
            report(
                2 + 68 * (chunk.index + 1) / len(chunks),
                "map",
                f"分段 {chunk.index + 1}/{len(chunks)} 使用缓存",
            )
        else:
            user = prompts.MAP_USER.format(
                index=chunk.index + 1,
                total=len(chunks),
                start=hhmmss(chunk.start),
                end=hhmmss(chunk.end),
                evidence_format=prompts.EVIDENCE_FORMAT,
                transcript=chunk.text,
            )
            t0 = time.perf_counter()
            payload = _chat_json(
                model=model,
                base_url=base_url,
                system=prompts.MAP_SYSTEM,
                user=user,
                schema=prompts.MAP_SCHEMA,
                num_ctx=num_ctx,
                timeout_sec=timeout_sec,
                should_cancel=is_cancelled,
            )
            if not isinstance(payload, dict):
                raise SummaryError(f"分段 {chunk.index + 1} 返回结构异常")
            write_json(
                cache_path,
                {
                    "key": key,
                    "chunk": chunk.as_dict(),
                    "payload": payload,
                    "elapsed_sec": round(time.perf_counter() - t0, 1),
                    "created_at": now_iso(),
                },
            )
            report(
                2 + 68 * (chunk.index + 1) / len(chunks),
                "map",
                f"分段 {chunk.index + 1}/{len(chunks)} 已抽取（{round(time.perf_counter() - t0, 1)}s）",
            )
        for fact in payload.get("facts") or []:
            if isinstance(fact, dict) and fact.get("statement"):
                fact.setdefault("chunk", chunk.index)
                facts.append(fact)
        if payload.get("chunk_summary"):
            summaries.append(str(payload["chunk_summary"]))
        chunk_records.append(
            {**chunk.as_dict(), "cache": str(cache_path.relative_to(summary_dir))}
        )

    report(72.0, "reduce", f"整理 {len(facts)} 条事实")

    # -------------------------------------------------------------- reduce
    # A single chunk needs no merge; a pure dedupe is deterministic and much
    # faster than an extra LLM round-trip (this CPU runs 8B at ~3.5 tok/s).
    facts = _dedupe_facts(facts)
    if len(chunks) > 1 and len(facts) > 80:
        merged = _reduce(
            facts,
            summaries,
            model=model,
            base_url=base_url,
            num_ctx=num_ctx,
            timeout_sec=timeout_sec,
            should_cancel=is_cancelled,
        )
    else:
        merged = {"chunk_summary": " ".join(s for s in summaries if s), "facts": facts}
    check_cancel()
    merged_facts = merged.get("facts") or facts
    report(80.0, "compose", f"整理为纪要（{len(merged_facts)} 条依据）")

    # ------------------------------------------------------------- compose
    basic_info = prompts.basic_info_block(meta, names)
    facts_json = json.dumps(merged_facts, ensure_ascii=False, indent=1)
    if len(facts_json) > MAX_FACTS_JSON_CHARS:
        facts_json = _compress_facts(merged_facts, MAX_FACTS_JSON_CHARS)
    template_user = (
        prompts.FINAL_INTERVIEW_USER if template == "interview" else prompts.FINAL_MEETING_USER
    )
    user = template_user.format(basic_info=basic_info, facts_json=facts_json)
    t0 = time.perf_counter()
    result = chat(
        model,
        [
            {"role": "system", "content": "你是一名专业的中文纪要撰写者，只依据给定事实写作。\n" + prompts.RULES},
            {"role": "user", "content": user},
        ],
        base_url=base_url,
        num_ctx=num_ctx,
        json_format=None,
        timeout_sec=timeout_sec,
        max_tokens=3000,
        should_cancel=is_cancelled,
    )
    markdown = result.content.strip()
    if not markdown:
        raise SummaryError("模型未返回纪要内容")
    compose_sec = round(time.perf_counter() - t0, 1)
    check_cancel()
    report(92.0, "compose", f"纪要初稿完成（{compose_sec}s）")

    # Persist the composed draft separately: a crash during verification must not
    # lose the (expensive) markdown, and must never overwrite a human-edited
    # final.json from an earlier run.
    partial = {
        "schema_version": 1,
        "template": template,
        "template_label": prompts.TEMPLATES[template]["label"],
        "model": model,
        "generated_at": now_iso(),
        "human_edited": False,
        "markdown": markdown,
        "facts": merged_facts,
        "verification": None,
        "source": meta.get("transcript_source", "unknown"),
        "stats": {"chunks": len(chunks), "facts": len(merged_facts), "chars": len(markdown)},
        "partial": True,
    }
    write_json(summary_dir / "final.partial.json", partial)
    write_text(summary_dir / "final.partial.md", markdown + "\n")

    # -------------------------------------------------------------- verify
    verification: dict[str, Any] | None = None
    if verify:
        report(94.0, "verify", "对照逐字稿核查结论依据…")
        try:
            verification = _chat_json(
                model=model,
                base_url=base_url,
                system=prompts.VERIFY_SYSTEM,
                user=prompts.VERIFY_USER.format(summary=markdown, facts_json=facts_json),
                schema=prompts.VERIFY_SCHEMA,
                num_ctx=num_ctx,
                timeout_sec=timeout_sec,
                should_cancel=is_cancelled,
            )
        except (SummaryError, OllamaError) as exc:
            verification = {"verdict": "unknown", "unsupported": [], "notes": f"核查失败：{exc}"}
    report(100.0, "done", "纪要生成完成")

    effective_speakers = {
        s: names.get(s, s) for s in {u.get("speaker") for u in utterances if u.get("speaker")}
    }
    final = {
        "schema_version": 1,
        "template": template,
        "template_label": prompts.TEMPLATES[template]["label"],
        "model": model,
        "generated_at": now_iso(),
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
        "compose_sec": compose_sec,
        "speakers": effective_speakers,
        "basic_info": basic_info,
        "markdown": markdown,
        "facts": merged_facts,
        "chunk_summaries": summaries,
        "chunks": chunk_records,
        "verification": verification,
        "stats": {
            "chunks": len(chunks),
            "chars": total_chars,
            "facts": len(merged_facts),
            "prompt_eval_count": result.prompt_eval_count,
            "eval_count": result.eval_count,
            "model_total_ms": result.total_duration_ms,
        },
        "source": meta.get("transcript_source", "unknown"),
    }
    write_json(summary_dir / "final.json", final)
    write_text(summary_dir / "final.md", markdown + "\n")
    for leftover in ("final.partial.json", "final.partial.md"):
        (summary_dir / leftover).unlink(missing_ok=True)
    write_json(
        summary_dir / "state.json",
        {
            "template": template,
            "model": model,
            "generated_at": final["generated_at"],
            "prompt_version": PROMPT_VERSION,
            "chunks": len(chunks),
            "force_next": False,
        },
    )
    return final


def _dedupe_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic near-duplicate removal (same type + same statement prefix)."""
    import re

    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for fact in facts:
        statement = re.sub(r"[\s，,。．.、；;：:]+", "", str(fact.get("statement") or ""))
        key = (str(fact.get("type") or ""), statement[:30])
        if key in seen:
            continue
        seen.add(key)
        out.append(fact)
    return out


def _reduce(
    facts: list[dict[str, Any]],
    summaries: list[str],
    *,
    model: str,
    base_url: str,
    num_ctx: int,
    timeout_sec: float,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Merge extracted facts, chunking the JSON so the prompt always fits."""
    if len(facts) <= 1:
        return {"chunk_summary": " ".join(summaries), "facts": facts}

    payload = json.dumps(facts, ensure_ascii=False)
    if len(payload) <= MAX_FACTS_JSON_CHARS:
        try:
            out = _chat_json(
                model=model,
                base_url=base_url,
                system=prompts.REDUCE_SYSTEM,
                user=prompts.REDUCE_USER.format(facts_json=payload),
                schema=prompts.MAP_SCHEMA,
                num_ctx=num_ctx,
                timeout_sec=timeout_sec,
                max_tokens=2600,
                should_cancel=should_cancel,
            )
            if isinstance(out, dict) and out.get("facts"):
                return out
        except (SummaryError, OllamaError):
            return {"chunk_summary": " ".join(summaries), "facts": facts}

    # group by size, merge each group, then recurse
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for fact in facts:
        item_size = len(json.dumps(fact, ensure_ascii=False)) + 2
        if current and size + item_size > MAX_FACTS_JSON_CHARS:
            groups.append(current)
            current = []
            size = 0
        current.append(fact)
        size += item_size
    if current:
        groups.append(current)

    merged: list[dict[str, Any]] = []
    summaries_out: list[str] = []
    for group in groups:
        try:
            out = _chat_json(
                model=model,
                base_url=base_url,
                system=prompts.REDUCE_SYSTEM,
                user=prompts.REDUCE_USER.format(facts_json=json.dumps(group, ensure_ascii=False)),
                schema=prompts.MAP_SCHEMA,
                num_ctx=num_ctx,
                timeout_sec=timeout_sec,
                should_cancel=should_cancel,
            )
            if isinstance(out, dict) and out.get("facts"):
                merged.extend(out["facts"])
                summaries_out.append(str(out.get("chunk_summary") or ""))
            else:
                merged.extend(group)
        except (SummaryError, OllamaError):
            merged.extend(group)
    if len(merged) > 1 and len(json.dumps(merged, ensure_ascii=False)) > MAX_FACTS_JSON_CHARS:
        return _reduce(
            merged,
            summaries_out,
            model=model,
            base_url=base_url,
            num_ctx=num_ctx,
            timeout_sec=timeout_sec,
            should_cancel=should_cancel,
        )
    return {"chunk_summary": " ".join(s for s in summaries_out if s), "facts": merged}


def _compress_facts(facts: list[dict[str, Any]], limit: int) -> str:
    """Trim evidence fields, then drop the least critical facts, to fit the prompt."""
    slim = []
    for fact in facts:
        item = {k: v for k, v in fact.items() if k in ("type", "statement", "time", "speaker", "quote", "quote_zh", "owner", "deadline")}
        if item.get("quote") and len(str(item["quote"])) > 160:
            item["quote"] = str(item["quote"])[:160] + "…"
        slim.append(item)
    text = json.dumps(slim, ensure_ascii=False)
    if len(text) <= limit:
        return text
    priority = {
        "decision": 0,
        "action": 1,
        "price": 2,
        "pain_point": 3,
        "unmet_need": 4,
        "quote": 5,
        "preference": 6,
        "buying_behavior": 7,
        "current_solution": 8,
        "background": 9,
        "scenario": 10,
        "viewpoint": 11,
        "topic": 12,
        "question": 13,
    }
    slim.sort(key=lambda f: priority.get(str(f.get("type")), 20))
    out: list[dict[str, Any]] = []
    size = 2
    for fact in slim:
        s = len(json.dumps(fact, ensure_ascii=False)) + 2
        if size + s > limit:
            break
        out.append(fact)
        size += s
    return json.dumps(out, ensure_ascii=False)


def effective_transcript_evidence(
    utterances: list[dict[str, Any]], names: dict[str, str]
) -> str:
    return format_evidence(utterances, names)
