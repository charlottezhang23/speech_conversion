"""Hugging Face token / gated-repo checks (network calls, short timeouts)."""
from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger("interviewai.hf")

WHOAMI = "https://huggingface.co/api/whoami-v2"
DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
DIARIZATION_CONFIG = f"https://huggingface.co/{DIARIZATION_REPO}/resolve/main/config.yaml"

# pyannote/speaker-diarization-3.1 is a *wrapper*: it needs its own gated repo plus
# pyannote/segmentation-3.0. Accepting one does not unlock the other.
KNOWN_REPOS = (
    "pyannote/speaker-diarization-community-1",
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
)
REQUIRED_FOR_DEFAULT = ("pyannote/speaker-diarization-community-1",)


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def whoami(token: str, timeout: float = 8.0) -> dict[str, Any]:
    """Return {'ok': bool, 'user': str|None, 'reason': str|None}."""
    if not token:
        return {"ok": False, "user": None, "reason": "missing"}
    try:
        res = requests.get(WHOAMI, headers=_headers(token), timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "user": None, "reason": "network", "detail": str(exc)}
    if res.status_code == 401:
        return {"ok": False, "user": None, "reason": "invalid"}
    if not res.ok:
        return {"ok": False, "user": None, "reason": "http", "detail": f"HTTP {res.status_code}"}
    try:
        data = res.json()
    except ValueError:
        return {"ok": False, "user": None, "reason": "http", "detail": "响应不是 JSON"}
    return {"ok": True, "user": data.get("name") or data.get("user"), "reason": None}


def gated_access(token: str, url: str = DIARIZATION_CONFIG, timeout: float = 15.0) -> dict[str, Any]:
    """Try to fetch the gated model's config.yaml.

    This is the same request pyannote makes, so it distinguishes
    'token invalid' from 'terms not accepted / not authorised'.
    """
    if not token:
        return {"ok": False, "reason": "missing"}
    try:
        res = requests.get(url, headers=_headers(token), timeout=timeout, stream=True)
        status = res.status_code
        res.close()
    except requests.RequestException as exc:
        return {"ok": False, "reason": "network", "detail": str(exc)}
    if status == 200:
        return {"ok": True, "reason": None}
    if status in (401, 403):
        return {"ok": False, "reason": "gated"}
    if status == 404:
        return {"ok": False, "reason": "not_found"}
    return {"ok": False, "reason": "http", "detail": f"HTTP {status}"}


def repo_access(token: str, repo: str, timeout: float = 12.0) -> dict[str, Any]:
    """Ask the Hub whether this token may actually download files from `repo`.

    `GET /api/models/{repo}` returns 200 even for gated repos the token cannot read
    (it only marks `gated`), so we probe real files instead: gated uploads answer
    401/403 while public ones answer 200/302.
    """
    if not token:
        return {"ok": False, "reason": "missing"}
    seen_404 = True
    for filename in ("config.yaml", "pytorch_model.bin", "model.safetensors"):
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        try:
            res = requests.head(url, headers=_headers(token), timeout=timeout, allow_redirects=True)
            status = res.status_code
            res.close()
        except requests.RequestException as exc:
            return {"ok": False, "reason": "network", "detail": str(exc)}
        if status in (200, 302):
            return {"ok": True, "reason": None, "file": filename}
        if status in (401, 403):
            return {"ok": False, "reason": "gated", "file": filename}
        if status != 404:
            return {"ok": False, "reason": "http", "detail": f"HTTP {status}"}
    return {"ok": False, "reason": "not_found"} if seen_404 else {"ok": False, "reason": "http"}


def check(token: str | None) -> dict[str, Any]:
    """One-shot check used by the settings page."""
    if not token:
        return {
            "ok": False,
            "reason": "missing",
            "message": "尚未配置 Hugging Face Token。",
        }
    who = whoami(token)
    if not who["ok"]:
        messages = {
            "invalid": "Token 无效或已被撤销（401）。请到 Hugging Face → Settings → Access Tokens 重新创建。",
            "network": "无法连接 huggingface.co，请检查网络或代理。",
        }
        return {
            "ok": False,
            "reason": who["reason"],
            "message": messages.get(who["reason"], f"Token 校验失败：{who.get('detail') or who['reason']}"),
        }

    access = gated_access(token)
    repos = []
    for repo in KNOWN_REPOS:
        result = repo_access(token, repo)
        repos.append({"repo": repo, "ok": bool(result["ok"]), "reason": result["reason"]})
    missing = [r["repo"] for r in repos if not r["ok"]]
    extra_note = ""
    if missing:
        extra_note = "\n\n未授权的仓库（如需 3.1 模型请逐个点「Agree」）：\n" + "\n".join(
            f"  - https://huggingface.co/{repo}" for repo in missing
        )

    if access["ok"]:
        return {
            "ok": True,
            "reason": None,
            "user": who["user"],
            "repo": DIARIZATION_REPO,
            "repos": repos,
            "message": (
                f"Token 有效（账号 {who['user']}），已具备 {DIARIZATION_REPO} 的访问权限。" + extra_note
            ),
        }
    if access["reason"] == "gated":
        return {
            "ok": False,
            "reason": "gated",
            "user": who["user"],
            "repo": DIARIZATION_REPO,
            "repos": repos,
            "message": (
                f"Token 有效（账号 {who['user']}），但该账号未获得 {DIARIZATION_REPO} 的访问权限。\n"
                "请用同一个账号打开模型页面 → 接受条款（Agree / Request access），状态变为 authorized 后重试。\n"
                f"https://huggingface.co/{DIARIZATION_REPO}" + extra_note
            ),
        }
    return {
        "ok": False,
        "reason": access["reason"],
        "user": who["user"],
        "message": f"无法读取模型配置：{access.get('detail') or access['reason']}",
    }
