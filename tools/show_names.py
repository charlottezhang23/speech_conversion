"""Print the speaker label -> display name mapping of a project's draft."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pid = sys.argv[1] if len(sys.argv) > 1 else "20261103-133124-b6812e"
draft = json.loads((ROOT / "projects" / pid / "transcript" / "draft.json").read_text(encoding="utf-8"))
print("speaker_names:", json.dumps(draft.get("speaker_names"), ensure_ascii=False))
print("review:", json.dumps(draft.get("review"), ensure_ascii=False))
