"""Check which gated pyannote repos the saved Hugging Face token can read."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import hf  # noqa: E402
from app.config import get_hf_token  # noqa: E402


def main() -> int:
    token = get_hf_token()
    who = hf.whoami(token) if token else {"ok": False, "user": None, "reason": "missing"}
    print(f"token: {'已配置' if token else '缺失'}  whoami={who.get('user')} ok={who.get('ok')} reason={who.get('reason')}")
    if not token:
        return 1
    for repo in hf.KNOWN_REPOS:
        result = hf.repo_access(token, repo)
        flag = "OK  " if result["ok"] else "FAIL"
        print(f"  {flag} {repo}  ({result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
