#!/bin/bash
# token_distill_watchdog.sh —— 监控 token 蒸馏进度，跑完后自动合并并输出通知
# 用法：cron 每 10 分钟执行一次
# 逻辑：
#   1) 蒸馏进程在跑 → 删除 .token_merge.done（合并标记失效，防漏合并）+ 首次报一次进度
#   2) 已合并过且无进程 → 静默退出
#   3) 进程已结束且未合并 → 冷却确认 → 执行合并，输出统计结果（cron 自动发送）
set -u
cd /e/skill/test || exit 1

# --- 1. 检查蒸馏进程是否还在跑 ---
# 用 wmic 查 api_distill 进程数；查询失败时保守退出（不合并）
PROC_OUT=$(wmic process where "name='python.exe'" get CommandLine 2>/dev/null || echo "ERR")
if echo "$PROC_OUT" | grep -qi "ERR"; then
    exit 0  # 查询失败，保守跳过
fi
RUNNING=$(echo "$PROC_OUT" | grep -c "api_distill")
RUNNING=${RUNNING:-0}

if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
    # 有进程在跑：合并标记失效（防止旧标记导致新轮次漏合并）
    rm -f .token_merge.done
    # 仅首次输出一次进度，之后静默
    if [ ! -f .progress_notified ]; then
        US=$(wc -l < output/token-us/cot.jsonl 2>/dev/null || echo 0)
        ZH=$(wc -l < output/token-zh/cot.jsonl 2>/dev/null || echo 0)
        echo "⏳ token 蒸馏进行中：us=${US} / zh=${ZH} 条，完成后会自动合并并通知你"
        touch .progress_notified
    fi
    exit 0
fi

# --- 2. 已合并过 → 静默 ---
if [ -f .token_merge.done ]; then
    exit 0
fi

# --- 3. 进程已结束 → 冷却确认（输出文件 5 分钟内无更新才动手）---
US_MTIME=$(stat -c %Y output/token-us/cot.jsonl 2>/dev/null || echo 0)
ZH_MTIME=$(stat -c %Y output/token-zh/cot.jsonl 2>/dev/null || echo 0)
NOW=$(date +%s)
if [ $((NOW - US_MTIME)) -lt 300 ] || [ $((NOW - ZH_MTIME)) -lt 300 ]; then
    exit 0  # 还在写文件，等等
fi

echo "✅ token 蒸馏进程已结束，开始合并数据..."
rm -f .progress_notified

if python merge_with_token.py --shuffle 2>&1; then
    touch .token_merge.done
    echo ""
    echo "✅ 合并完成！训练集已更新：data/sft/train.jsonl + eval.jsonl"
    echo "（merge_with_token.py 合并 deepseek/nvidia/agnes/tokenrhythm 全部来源，自动去重）"
else
    echo "❌ 合并失败，请检查上面的错误信息"
fi
