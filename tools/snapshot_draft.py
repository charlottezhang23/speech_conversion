"""Snapshot the current draft transcript so a rebuild can be compared against it."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pid = sys.argv[1]
src = ROOT / "projects" / pid / "transcript" / "draft.json"
dst = ROOT / "tmp" / "draft_before.json"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)
data = json.loads(dst.read_text(encoding="utf-8"))
print(f"saved {dst} ({len(data['utterances'])} utterances)")
