#!/usr/bin/env bash
# start_parts.sh —— 带锁启动 4 个蒸馏分片（防 Hermes 重复派生）
# 锁机制：mkdir parts.lock，重复实例直接退出
cd /e/skill/test
LOCKDIR=parts.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[lock] 已有 start_parts.sh 实例在运行，本实例退出"
  exit 0
fi
trap 'rm -rf "$LOCKDIR"' EXIT

# 杀残留 api_distill 进程（防重复写文件/重复计费）
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*api_distill*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 1

DK=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2)
mkdir -p output/full-{1,2,3,4}
for i in 1 2 3 4; do
  DISTILL_CONCURRENCY=8 python -u api_distill.py --task cot --input "data/us4_part$i.jsonl" --k 1 --api-key "$DK" --base-url https://api.deepseek.com/v1 --model deepseek-v4-flash --out-dir "output/full-$i" >> "output/full-$i/run.log" 2>&1 &
  echo "[start] 分片 $i 已启动 (PID $!)"
done
wait
echo "[start] 全部分片执行完毕"
