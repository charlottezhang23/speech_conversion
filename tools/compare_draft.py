"""Compare a snapshot draft with the current draft (utterance counts and long lines)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
before = json.loads((ROOT / "tmp" / "draft_before.json").read_text(encoding="utf-8"))["utterances"]
after = json.loads(
    (ROOT / "projects" / "20261103-133124-b6812e" / "transcript" / "draft.json").read_text(encoding="utf-8")
)["utterances"]

lines = []
for name, utts in (("before", before), ("after", after)):
    durs = [float(u["end"]) - float(u["start"]) for u in utts]
    long_lines = sum(1 for d in durs if d > 15)
    lines.append(
        f"{name}: 段数={len(utts)}  平均时长={sum(durs) / len(durs):.1f}s  中位数={sorted(durs)[len(durs) // 2]:.1f}s  "
        f"最长={max(durs):.1f}s  超过15s 的行={long_lines}  "
        f"说话人分布={dict(Counter(u['speaker'] for u in utts))}"
    )

print("\n".join(lines))
(ROOT / "logs" / "draft_compare.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
