"""Diff the cached pyannote pipeline configs (which components each model uses)."""
from __future__ import annotations

import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
base = ROOT / "models" / "pyannote"


def configs(repo: str) -> list[Path]:
    return sorted((base / f"models--{repo.replace('/', '--')}" / "snapshots").glob("*/config.yaml"))


def main() -> int:
    a = configs("pyannote/speaker-diarization-community-1")
    b = configs("pyannote/speaker-diarization-3.1")
    if not a or not b:
        print(f"缺少缓存配置：community-1={len(a)} 3.1={len(b)}")
        return 1
    left = a[-1].read_text(encoding="utf-8").splitlines()
    right = b[-1].read_text(encoding="utf-8").splitlines()
    diff = list(difflib.unified_diff(left, right, fromfile="community-1", tofile="3.1", lineterm=""))
    out = ROOT / "logs" / "pyannote_config_diff.txt"
    out.write_text("\n".join(diff) + "\n", encoding="utf-8")
    if not diff:
        print("两个 pipeline 的 config.yaml 完全一致")
    else:
        print("\n".join(diff[:80]))
    print(f"\n写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
