#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_with_token.py —— 合并所有蒸馏数据（含 tokenrhythm 新数据），生成训练集

用法：
  python merge_with_token.py [--shuffle] [--split-ratio 0.9]

输出：
  data/sft/merged_all.jsonl
  data/sft/train.jsonl
  data/sft/eval.jsonl
"""
import argparse
import json
import os
import random
import re
from collections import Counter


SOURCES = [
    ("output/us4-full/cot.jsonl", "deepseek-us4"),
    ("output/nvidia-full/cot.jsonl", "nvidia"),
    ("output/agnes-cn-full/cot.jsonl", "agnes-cn"),
    ("output/zh-full/cot.jsonl", "deepseek-zh"),
    ("output/token-us/cot.jsonl", "tokenrhythm-us"),
    ("output/token-zh/cot.jsonl", "tokenrhythm-zh"),
]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 跳过损坏行（多进程并发写可能产生截断行）
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


def dedup(rows, fingerprint_len=100):
    """按 instruction 前 N 字符指纹去重。"""
    seen = set()
    kept = []
    for r in rows:
        k = r.get("instruction", "")[:fingerprint_len]
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="总条数上限")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--split-ratio", type=float, default=0.9, help="训练集比例")
    ap.add_argument("--output", default="data/sft", help="输出目录")
    args = ap.parse_args()

    all_rows = []
    for path, source in SOURCES:
        if os.path.exists(path):
            rows = load_jsonl(path)
            # 先对每个源去重（token 文件因多进程并发写有大量重复）
            rows = dedup(rows)
            for r in rows:
                r["source"] = source
                r["answer"] = extract_answer(r.get("output", ""))
            all_rows.extend(rows)
            print(f"  {source:16} {len(rows):5} 条  ({path})")

    print(f"\n合并: {len(all_rows)} 条")
    print(f"来源分布: {dict(Counter(r['source'] for r in all_rows))}")

    all_rows = dedup(all_rows)
    print(f"跨源去重: {len(all_rows)} 条")

    if args.limit:
        all_rows = all_rows[:args.limit]
        print(f"截断: {len(all_rows)} 条")

    if args.shuffle:
        random.seed(42)
        random.shuffle(all_rows)

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "merged_all.jsonl"), "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    split = int(len(all_rows) * args.split_ratio)
    train_rows = all_rows[:split]
    eval_rows = all_rows[split:]

    with open(os.path.join(args.output, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(args.output, "eval.jsonl"), "w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n输出:")
    print(f"  合并: {args.output}/merged_all.jsonl ({len(all_rows)} 条)")
    print(f"  训练: {args.output}/train.jsonl ({len(train_rows)} 条)")
    print(f"  验证: {args.output}/eval.jsonl ({len(eval_rows)} 条)")

    print(f"\n统计:")
    print(f"  平均 output 长度: {sum(len(r['output']) for r in train_rows) // max(len(train_rows), 1)} 字符")
    print(f"  答案分布: {dict(Counter(r.get('answer', 'A') for r in train_rows))}")


if __name__ == "__main__":
    main()
