#!/usr/bin/env bash
# watchdog.sh —— 蒸馏分片看护：检测停滞/进程死亡，自动重启未完成分片
# 用法: bash watchdog.sh （后台运行）；mkdir 锁防 Hermes 重复派生
cd /e/skill/test
LOCKDIR=watchdog.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[lock] 已有 watchdog 实例在运行，本实例退出"
  exit 0
fi
trap 'rm -rf "$LOCKDIR"' EXIT

DK=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2)
TARGET1=2279; TARGET2=2279; TARGET3=2279; TARGET4=2277
TOTAL_TARGET=9114

is_done() {
  local n=$(wc -l < "output/full-$1/cot.done" 2>/dev/null || echo 0)
  [ "$n" -ge $(eval echo \$TARGET$1) ]
}

count_all() {
  local t=0
  for i in 1 2 3 4; do
    local n=$(wc -l < "output/full-$i/cot.done" 2>/dev/null || echo 0)
    t=$((t+n))
  done
  echo $t
}

run_part() {
  local i=$1
  if ! is_done $i; then
    echo "[watchdog] $(date +%H:%M:%S) 启动分片 $i"
    mkdir -p "output/full-$i"
    (cd /e/skill/test && DISTILL_CONCURRENCY=8 python -u api_distill.py --task cot --input "data/us4_part$i.jsonl" --k 1 --api-key "$DK" --base-url https://api.deepseek.com/v1 --model deepseek-v4-flash --out-dir "output/full-$i" >> "output/full-$i/run.log" 2>&1 &)
  fi
}

# 初始启动全部未完成分片
for i in 1 2 3 4; do run_part $i; done

last=-1
stall=0
while true; do
  sleep 180
  total=$(count_all)
  # 分片进程死亡但未完成 → 重启
  for i in 1 2 3 4; do
    if ! is_done $i; then
      alive=$(powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*us4_part$i*' -and \$_.CommandLine -like '*api_distill*' } | Measure-Object | Select-Object -ExpandProperty Count" 2>/dev/null | tr -d '\r')
      if [ "${alive:-0}" -eq 0 ]; then
        echo "[watchdog] $(date +%H:%M:%S) 分片 $i 进程死亡但未完成 (done=$(wc -l < "output/full-$i/cot.done" 2>/dev/null || echo 0))，重启"
        run_part $i
      fi
    fi
  done
  # 全局停滞检测（6 分钟无进展 → 杀全部重启）
  if [ "$total" -eq "$last" ] && [ "$total" -lt "$TOTAL_TARGET" ]; then
    stall=$((stall+1))
    if [ "$stall" -ge 2 ]; then
      echo "[watchdog] $(date +%H:%M:%S) 全局停滞 6 分钟 (total=$total)，杀全部并重启未完成分片"
      powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*api_distill*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
      sleep 3
      for i in 1 2 3 4; do run_part $i; done
      stall=0
    fi
  else
    stall=0
  fi
  echo "[watchdog] $(date +%H:%M:%S) 总进度: $total/$TOTAL_TARGET"
  if [ "$total" -ge "$TOTAL_TARGET" ]; then
    echo "[watchdog] $(date +%H:%M:%S) 全部完成，退出"
    break
  fi
  last=$total
done
