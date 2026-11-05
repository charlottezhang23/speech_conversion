"""Minimal Ollama HTTP client (localhost only, streaming, no SDK needed)."""
from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import requests

logger = logging.getLogger("interviewai.ollama")

DEFAULT_URL = "http://127.0.0.1:11434"


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatResult:
    content: str
    model: str
    total_duration_ms: float = 0.0
    eval_count: int = 0
    prompt_eval_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def _url(base_url: str, path: str) -> str:
    return (base_url or DEFAULT_URL).rstrip("/") + path


def ping(base_url: str = DEFAULT_URL, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(_url(base_url, "/api/version"), timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def version(base_url: str = DEFAULT_URL) -> str | None:
    try:
        r = requests.get(_url(base_url, "/api/version"), timeout=3.0)
        if r.ok:
            return r.json().get("version")
    except requests.RequestException:
        return None
    return None


def list_models(base_url: str = DEFAULT_URL) -> list[dict[str, Any]]:
    try:
        r = requests.get(_url(base_url, "/api/tags"), timeout=10)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaError(f"无法连接本地 Ollama（{base_url}）：{exc}") from exc
    models = r.json().get("models", [])
    out = []
    for m in models:
        out.append(
            {
                "name": m.get("name") or m.get("model"),
                "size": m.get("size"),
                "modified_at": m.get("modified_at"),
                "family": (m.get("details") or {}).get("family"),
                "parameter_size": (m.get("details") or {}).get("parameter_size"),
            }
        )
    out.sort(key=lambda m: m["name"] or "")
    return out


def loaded_models(base_url: str = DEFAULT_URL) -> list[str]:
    try:
        r = requests.get(_url(base_url, "/api/ps"), timeout=5)
        r.raise_for_status()
        return [m.get("name") or m.get("model") for m in r.json().get("models", [])]
    except requests.RequestException:
        return []


def unload(model: str, base_url: str = DEFAULT_URL) -> None:
    """Ask Ollama to evict a model from RAM (keep_alive=0)."""
    try:
        requests.post(
            _url(base_url, "/api/generate"),
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=30,
        )
    except requests.RequestException:
        pass


def chat(
    model: str,
    messages: Iterable[dict[str, str]],
    *,
    base_url: str = DEFAULT_URL,
    num_ctx: int = 16384,
    temperature: float = 0.2,
    top_p: float = 0.9,
    json_format: str | dict[str, Any] | None = "json",
    timeout_sec: float = 900.0,
    on_token: Callable[[int], None] | None = None,
    think: bool | None = False,
    seed: int = 42,
    max_tokens: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> ChatResult:
    """Stream a chat completion. `json_format` may be "json" or a JSON schema dict.

    `max_tokens` maps to Ollama's `num_predict` and caps runaway generations
    (a small model can otherwise keep emitting JSON array items forever).
    `should_cancel` is polled per streamed chunk so the request can be aborted.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": True,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": int(num_ctx),
            "seed": int(seed),
        },
        "keep_alive": "10m",
    }
    if max_tokens:
        payload["options"]["num_predict"] = int(max_tokens)
    if json_format:
        payload["format"] = json_format
    if think is not None:
        payload["think"] = think

    chunks: list[str] = []
    final: dict[str, Any] = {}
    events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            with requests.post(
                _url(base_url, "/api/chat"),
                json=payload,
                stream=True,
                timeout=(10, max(120.0, min(timeout_sec, 300.0))),
            ) as resp:
                holder["resp"] = resp
                if resp.status_code != 200:
                    events.put(("error", f"Ollama 返回 {resp.status_code}: {resp.text[:500]}"))
                    return
                for line in resp.iter_lines(decode_unicode=False):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode("utf-8", "replace"))
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        events.put(("error", f"Ollama 错误: {obj['error']}"))
                        return
                    piece = (obj.get("message") or {}).get("content") or ""
                    if piece:
                        events.put(("chunk", piece))
                    if obj.get("done"):
                        events.put(("done", obj))
                        return
            events.put(("done", {}))
        except requests.Timeout as exc:
            events.put(("timeout", exc))
        except requests.ConnectionError as exc:
            events.put(("connection", exc))
        except Exception as exc:  # noqa: BLE001 - surfaced to caller
            events.put(("exception", exc))
        finally:
            holder.pop("resp", None)

    thread = threading.Thread(target=worker, daemon=True, name="ollama-stream")
    thread.start()
    n_tokens = 0
    try:
        while True:
            if should_cancel is not None and should_cancel():
                resp = holder.get("resp")
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                raise OllamaError("已取消")
            try:
                kind, data = events.get(timeout=0.5)
            except queue.Empty:
                if not thread.is_alive() and events.empty():
                    break
                continue
            if kind == "chunk":
                chunks.append(data)
                n_tokens += 1
                if on_token and n_tokens % 20 == 0:
                    on_token(n_tokens)
            elif kind == "done":
                final = data
                break
            elif kind == "error":
                raise OllamaError(str(data))
            elif kind == "timeout":
                raise OllamaError(f"Ollama 请求超时（>{timeout_sec}s）：{data}") from data
            elif kind == "connection":
                raise OllamaError(f"无法连接本地 Ollama（{base_url}）。请确认 `ollama serve` 已启动。") from data
            elif kind == "exception":
                raise OllamaError(f"Ollama 请求失败：{data}") from data
    finally:
        thread.join(timeout=1.0)

    content = "".join(chunks)
    return ChatResult(
        content=content,
        model=model,
        total_duration_ms=float(final.get("total_duration", 0)) / 1e6,
        eval_count=int(final.get("eval_count", 0)),
        prompt_eval_count=int(final.get("prompt_eval_count", 0)),
        raw={k: v for k, v in final.items() if k not in ("message",)},
    )


def extract_json(text: str) -> Any:
    """Best-effort JSON parse of model output (tolerates code fences / prose)."""
    text = (text or "").strip()
    if not text:
        raise OllamaError("模型返回为空")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if text.startswith("```"):
        body = text.split("```")[1] if text.count("```") >= 2 else text.strip("`")
        body = body.strip()
        if body.lower().startswith("json"):
            body = body[4:]
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    raise OllamaError("模型未返回有效 JSON")
