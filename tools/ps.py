"""Print relevant processes (ollama / llama / python) — reliable replacement for tasklist in PS."""
from __future__ import annotations

import csv
import io
import subprocess


def main() -> int:
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(out.stdout)))
    for row in rows:
        if not row:
            continue
        name = row[0].lower()
        if any(k in name for k in ("ollama", "llama", "python")):
            print(",".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
