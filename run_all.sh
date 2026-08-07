#!/usr/bin/env bash
# run_all.sh —— 挨个跑完所有可用 provider 的 CoT 蒸馏试水（200 条 k=1）
# 防重复机制：
#   1) 启动前杀掉所有残留 api_distill 进程
#   2) mkdir 单实例锁：即使本脚本被重复执行，也只有第一个实例能跑
#   3) .done 断点续传 + 每轮结束后去重，保证数据不重复
set -u
cd "$(dirname "$0")"

# --- 单实例锁 ---
LOCKDIR=run.lock
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "[lock] 已有 run_all.sh 实例在运行，本实例退出"
  exit 1
fi
trap 'rm -rf "$LOCKDIR"' EXIT

# --- 杀残留（防重复写文件/重复计费）---
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*api_distill*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 1

# 并发调高（429 则调回 8）
export DISTILL_CONCURRENCY=16

DK=$(grep '^OPENAI_API_KEY=' .env | cut -d= -f2)
NK=$(grep '^NVIDIA_API_KEY=' .env | cut -d= -f2)
AK=$(grep '^AGNES_CN_API_KEY=' .env | cut -d= -f2)

INPUT=data/us4_train.jsonl

run_provider() {
  local name="$1"; shift
  echo "================ [$name] 冒烟测试 3 条 ================"
  "$@" --limit 3 --out-dir "output/smoke-$name" 2>&1 | tail -2
  local n
  n=$(wc -l < "output/smoke-$name/cot.jsonl" 2>/dev/null || echo 0)
  if [ "${n:-0}" -ge 1 ]; then
    echo "================ [$name] 冒烟 OK（${n} 条），正式跑 200 条 ================"
    "$@" --limit 200 --out-dir "output/$name" 2>&1 | tail -3
    # 保险：去重（若发生重复写）
    python api_distill.py --dedup --input "output/$name/cot.jsonl" \
      --out "output/$name/cot_dedup.jsonl" >/dev/null 2>&1 && \
      mv -f "output/$name/cot_dedup.jsonl" "output/$name/cot.jsonl"
    echo "================ [$name] 完成: $(wc -l < "output/$name/cot.jsonl" 2>/dev/null || echo 0) 条 ================"
  else
    echo "================ [$name] 冒烟失败，跳过正式跑 ================"
  fi
}

echo "########## [1/3] DeepSeek 直连 (deepseek-v4-flash) ##########"
run_provider deepseek-flash \
  python api_distill.py --task cot --input "$INPUT" --k 1 \
    --api-key "$DK" --base-url https://api.deepseek.com/v1 --model deepseek-v4-flash

echo "########## [2/3] NVIDIA (deepseek-ai/deepseek-v4-flash, TLS1.2) ##########"
run_provider nvidia \
  env DISTILL_TLS12=1 python api_distill.py --task cot --input "$INPUT" --k 1 \
    --api-key "$NK" --base-url https://integrate.api.nvidia.com/v1 --model deepseek-ai/deepseek-v4-flash \
    --limit 100

echo "########## [3/3] Agnes-CN (agnes-2.5-flash) ##########"
run_provider agnes-cn \
  python api_distill.py --task cot --input "$INPUT" --k 1 \
    --api-key "$AK" --base-url https://api.agnes-ai.cn/v1 --model agnes-2.5-flash

echo "########## 全部完成 ##########"
