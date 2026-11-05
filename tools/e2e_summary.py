"""End-to-end summary generation against a real local Ollama model."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.projects import get_project, list_projects  # noqa: E402
from app.transcript import effective_speaker_names, load_effective, summary_source_meta  # noqa: E402
from summarization.summarize import summarize  # noqa: E402


def pick_project(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [p for p in list_projects() if p.get("has_draft")]
    if not candidates:
        raise SystemExit("没有可用的逐字稿项目，请先运行 tools/e2e_run.py")
    return candidates[0]["id"]


def main() -> int:
    project_id = pick_project(sys.argv[1] if len(sys.argv) > 1 else None)
    project = get_project(project_id)
    transcript, source = load_effective(project)
    names = effective_speaker_names(project, transcript)
    settings = load_settings()
    model = settings["ollama_model"]
    lines = [f"项目：{project_id}（{project.data['name']}）", f"模型：{model}", f"逐字稿来源：{source}"]

    for template in ("interview", "meeting"):
        t0 = time.perf_counter()
        result = summarize(
            utterances=transcript["utterances"],
            names=names,
            meta=summary_source_meta(project, transcript, source),
            summary_dir=project.summary_dir,
            template=template,
            model=model,
            base_url=settings["ollama_url"],
            num_ctx=int(settings["ollama_num_ctx"]),
            timeout_sec=float(settings["ollama_timeout_sec"]),
            max_chunk_chars=int(settings["summary_chunk_chars"]),
            verify=True,
        )
        elapsed = time.perf_counter() - t0
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"[{template}] 用时 {elapsed:.1f}s，分段 {result['stats']['chunks']}，事实 {result['stats']['facts']}，字数 {len(result['markdown'])}")
        lines.append(f"核查结论：{result['verification'].get('verdict') if result['verification'] else '未启用'}")
        lines.append("-" * 70)
        lines.append(result["markdown"])

    out = ROOT / "logs" / "e2e_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\n报告：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
