#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_data.py —— 合并多 provider 蒸馏数据 + 生成训练集

用法：
  python merge_data.py --output data/sft/merged.jsonl --split data/sft/train.jsonl data/sft/eval.jsonl
"""
import argparse
import json
import os
from collections import Counter

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dedup(rows, key="instruction"):
    """按 instruction 前缀去重（保留第一条）。"""
    seen = set()
    kept = []
    for r in rows:
        k = r.get(key, "")[:80]
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept


def to_alpaca(row):
    """CoT 蒸馏数据 → Alpaca 格式（数据已结构化，直接提取）。"""
    # 提取 answer 字母（从 output 末尾）
    output = row.get("output", "")
    lines = output.strip().split("\n")
    answer = None
    for line in reversed(lines):
        if line.strip().startswith("答案是") or "答案：" in line:
            for c in "ABCDE":
                if c in line:
                    answer = c
                    break
        elif '"answer":"' in line or '"answer": "' in line:
            for c in "ABCDE":
                if f'"answer": "{c}"' in line or f'"answer":"{c}"' in line:
                    answer = c
                    break
        if answer:
            break
    if not answer:
        answer = row.get("answer", "A")
    
    # 数据已结构化，直接使用 instruction
    instruction = row.get("instruction", "")
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": row.get("source", "medqa"),
        "id": row.get("id", ""),
        "answer": answer
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="合并输出路径")
    ap.add_argument("--split", nargs=2, help="训练/验证集路径")
    ap.add_argument("--eval-ratio", type=float, default=0.1)
    args = ap.parse_args()

    # 加载所有数据
    all_rows = []
    sources = Counter()
    for path in ["output/us4-full/cot.jsonl", "output/nvidia-full/cot.jsonl", "output/agnes-cn-full/cot.jsonl"]:
        if os.path.exists(path):
            rows = load_jsonl(path)
            source = path.split("/")[1]
            for r in rows:
                r["source"] = source
            all_rows.extend(rows)
            sources[source] += len(rows)
            print(f"  {path:40} {len(rows):5} 条")

    print(f"\n合并后: {len(all_rows)} 条")
    print(f"来源分布: {dict(sources)}")

    # 去重
    all_rows = dedup(all_rows)
    print(f"去重后: {len(all_rows)} 条")

    # 保存合并文件
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"合并保存: {args.output}")

    # 生成 Alpaca 格式
    if args.split:
        train_path, eval_path = args.split
        os.makedirs(os.path.dirname(train_path) or ".", exist_ok=True)
        
        # 按来源分层采样
        from_sources = {}
        for r in all_rows:
            s = r.get("source", "unknown")
            from_sources.setdefault(s, []).append(r)
        
        train_rows, eval_rows = [], []
        import random
        random.seed(42)
        for s, rows in from_sources.items():
            random.shuffle(rows)
            split = int(len(rows) * (1 - args.eval_ratio))
            train_rows.extend(rows[:split])
            eval_rows.extend(rows[split:])
        
        # 转换为 Alpaca
        def to_alpaca_list(rows):
            return [to_alpaca(r) for r in rows]
        
        train_alpaca = to_alpaca_list(train_rows)
        eval_alpaca = to_alpaca_list(eval_rows)
        
        with open(train_path, "w", encoding="utf-8") as f:
            for r in train_alpaca:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(eval_path, "w", encoding="utf-8") as f:
            for r in eval_alpaca:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        print(f"\nAlpaca 格式:")
        print(f"  训练集: {train_path} ({len(train_alpaca)} 条)")
        print(f"  验证集: {eval_path} ({len(eval_alpaca)} 条)")

        # 统计
        print(f"\n数据统计:")
        print(f"  平均输出长度: {sum(len(r['output']) for r in train_alpaca) // len(train_alpaca)} 字符")
        print(f"  答案分布: {Counter(r['answer'] for r in train_alpaca)}")


if __name__ == "__main__":
    main()
