#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_token_data.py —— 将 tokenrhythm 新蒸馏数据并入训练集，重建 train/eval

逻辑：
1. 读取现有 merged_all.jsonl（已含 source/answer 字段的旧数据）
2. 读取新数据 output/token_merged.jsonl（30,755 条，无 source/answer）
   - 按 instruction 语言区分 us/zh：英文题 → tokenrhythm-us，中文题 → tokenrhythm-zh
   - 从 output 提取 answer 字母
3. 合并 + 按 instruction 去重
4. 随机打乱，按比例拆分为 train/eval，写回 data/sft/
"""
import argparse
import json
import os
import random
import re
from collections import Counter


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_answer(output):
    """从 output 末尾提取答案字母。"""
    lines = output.strip().split("\n")
    for line in reversed(lines):
        if "答案是" in line or "答案：" in line or "答案:" in line:
            for c in "ABCDE":
                if c in line:
                    return c
        if '"answer"' in line:
            m = re.search(r'"answer"\s*:\s*"([ABCDE])"', line)
            if m:
                return m.group(1)
    return None


def dedup(rows):
    """按 instruction 全文本去重（同题不同选项版本保留）。"""
    seen = set()
    kept = []
    for r in rows:
        k = r.get("instruction", "")
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="data/sft/merged_all.jsonl", help="现有合并数据")
    ap.add_argument("--new", default="output/token_merged.jsonl", help="新 tokenrhythm 数据")
    ap.add_argument("--output", default="data/sft", help="输出目录")
    ap.add_argument("--split-ratio", type=float, default=0.9, help="训练集比例")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. 旧数据（保留原有 source）
    old_rows = []
    if os.path.exists(args.old):
        old_rows = load_jsonl(args.old)
        print(f"旧数据: {len(old_rows)} 条 ({args.old})")

    # 2. 新数据（补 source/answer）
    new_rows = []
    for r in load_jsonl(args.new):
        instr = r.get("instruction", "")
        # 判断 us/zh：题目以英文开头 → us；否则 → zh
        first_ch = instr[3:].strip()[:1] if instr.startswith("题目：") else instr[:1]
        src = "tokenrhythm-us" if first_ch and ord(first_ch) < 128 else "tokenrhythm-zh"
        r["source"] = src
        r["answer"] = extract_answer(r.get("output", ""))
        new_rows.append(r)
    print(f"新数据: {len(new_rows)} 条 ({args.new})")

    # 3. 合并去重
    all_rows = dedup(old_rows + new_rows)
    print(f"合并去重后: {len(all_rows)} 条")
    print(f"来源分布: {dict(Counter(r['source'] for r in all_rows))}")

    # 4. 打乱 + 拆分
    random.seed(args.seed)
    random.shuffle(all_rows)
    split = int(len(all_rows) * args.split_ratio)
    train_rows = all_rows[:split]
    eval_rows = all_rows[split:]

    os.makedirs(args.output, exist_ok=True)
    # 写 merged_all（全量）
    with open(os.path.join(args.output, "merged_all.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 写 train/eval
    with open(os.path.join(args.output, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n输出:")
    print(f"  merged_all: {len(all_rows)} 条")
    print(f"  train: {len(train_rows)} 条")
    print(f"  eval: {len(eval_rows)} 条")
    print(f"\n统计:")
    print(f"  答案分布: {dict(Counter(r.get('answer') or '?' for r in train_rows))}")


if __name__ == "__main__":
    main()
