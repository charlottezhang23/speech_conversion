"""Measure Ollama latency for the three call shapes used by the summariser."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from summarization import prompts  # noqa: E402
from summarization.ollama_client import chat  # noqa: E402

LINES = [
    "[00:00:01] 访谈者: 你好，谢谢你今天接受采访。先请你简单介绍一下自己。",
    "[00:00:09] 受访者: 好的。我今年二十九岁，在杭州做产品设计，家里养了两只猫。",
    "[00:00:18] 访谈者: 平时养猫最让你困扰的事情是什么？",
    "[00:00:23] 受访者: 最麻烦的是掉毛，每天都要吸地，不然沙发上全是毛。",
]


def main() -> int:
    s = load_settings()
    model = s["ollama_model"]
    url = s["ollama_url"]
    out: list[str] = [f"model={model} url={url}"]

    def run(label: str, messages, **kw) -> None:
        t0 = time.perf_counter()
        res = chat(model, messages, base_url=url, num_ctx=8192, timeout_sec=600, **kw)
        dt = time.perf_counter() - t0
        out.append(
            f"{label}: {dt:.1f}s  prompt={res.prompt_eval_count} eval={res.eval_count} "
            f"chars={len(res.content)} 内容首行={res.content.strip().splitlines()[0][:80] if res.content.strip() else '<空>'}"
        )

    run("plain-40tok", [{"role": "user", "content": "用一句话介绍你自己。"}], json_format=None, max_tokens=40)
    run("json-200tok", [{"role": "user", "content": "输出 JSON：{\"a\": 1, \"b\": \"中文\"}"}], json_format="json", max_tokens=200)
    run(
        "map-schema",
        [
            {"role": "system", "content": prompts.MAP_SYSTEM},
            {
                "role": "user",
                "content": prompts.MAP_USER.format(
                    index=1, total=1, start="00:00:00", end="00:00:30",
                    evidence_format=prompts.EVIDENCE_FORMAT, transcript="\n".join(LINES),
                ),
            },
        ],
        json_format=prompts.MAP_SCHEMA,
        max_tokens=800,
    )
    report = ROOT / "logs" / "llm_probe.txt"
    report.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
