@echo off
REM 简单的 Ollama 速度探测：生成 40 个 token，输出 eval_count / 耗时
curl.exe -s http://127.0.0.1:11434/api/generate -d "{\"model\":\"qwen3:8b\",\"prompt\":\"请用一句话介绍你自己。\",\"stream\":false,\"think\":false,\"options\":{\"num_predict\":40,\"num_ctx\":4096}}" -o "%~dp0..\logs\ollama_speed.json"
echo done
