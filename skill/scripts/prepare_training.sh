#!/bin/bash
# prepare_training.sh —— 一键准备训练数据

set -e
echo "=== 准备训练数据 ==="
echo ""

# 1. 合并多 provider 数据
echo "[1/3] 合并蒸馏数据..."
python merge_data.py \
    --output data/sft/merged.jsonl \
    --split data/sft/train.jsonl data/sft/eval.jsonl

# 2. 生成统计
echo ""
echo "[2/3] 生成统计..."
python -c "
import json
from collections import Counter

rows = [json.loads(l) for l in open('data/sft/train.jsonl', encoding='utf-8')]
print(f'训练集: {len(rows)} 条')
print(f'答案分布: {dict(Counter(r[\"answer\"] for r in rows))}')
print(f'平均输出长度: {sum(len(r[\"output\"]) for r in rows) // len(rows)} 字符')

eval_rows = [json.loads(l) for l in open('data/sft/eval.jsonl', encoding='utf-8')]
print(f'验证集: {len(eval_rows)} 条')
"

# 3. 验证文件格式
echo ""
echo "[3/3] 验证格式..."
python -c "
import json
row = json.loads(open('data/sft/train.jsonl', encoding='utf-8').readline())
print(f'字段: {list(row.keys())}')
print(f'样例输出长度: {len(row[\"output\"])} 字符')
print('格式验证通过 ✓')
"

echo ""
echo "=== 训练数据准备完成 ==="
echo "路径: data/sft/"
