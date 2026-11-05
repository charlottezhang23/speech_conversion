"""Server entry point: `python -m app.main`."""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def _open_later(url: str, delay: float = 1.8) -> None:
    def _run() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地访谈转录与会议纪要系统")
    parser.add_argument("--host", default=os.environ.get("INTERVIEWAI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("INTERVIEWAI_PORT", "7860")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true", help="开发用：代码变更自动重载")
    args = parser.parse_args(argv)

    from app.config import ensure_ffmpeg_on_path, find_ffmpeg

    ensure_ffmpeg_on_path()
    if not find_ffmpeg():
        print("⚠ 未找到 FFmpeg。请运行 install.ps1，或把 ffmpeg.exe 放到 tools/ffmpeg/bin/ 下。")

    url = f"http://{args.host}:{args.port}"
    if not _port_free(args.host, args.port):
        print(f"端口 {args.port} 已被占用——可能应用已经在运行：{url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    import uvicorn

    if not args.no_browser:
        _open_later(url)
    print(f"服务已启动：{url}（仅监听本机，按 Ctrl+C 停止）")
    uvicorn.run(
        "app.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
