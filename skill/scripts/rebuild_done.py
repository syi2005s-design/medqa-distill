#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_done.py —— 重建干净的 done 文件
从 cot.jsonl 中提取真实成功的题目，重建 done 文件（剔除被余额不足污染的 skip 条目）

用法：
  python rebuild_done.py --name us --input data/us_train.jsonl --out-dir output/token-us
  python rebuild_done.py --name zh --input data/zh_train.jsonl --out-dir output/token-zh
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="重建干净 done 文件")
    ap.add_argument("--name", required=True, help="数据集名(us/zh)")
    ap.add_argument("--input", required=True, help="原始输入文件")
    ap.add_argument("--out-dir", required=True, help="输出目录")
    args = ap.parse_args()

    # 1. 读取原始输入：question -> id
    q2id = {}
    for line in open(args.input, encoding="utf-8"):
        d = json.loads(line)
        q2id[d["question"]] = d["id"]

    # 2. 从 cot.jsonl 提取成功题目的 question
    jsonl_path = Path(args.out_dir) / "cot.jsonl"
    success_questions = []
    seen = set()
    for line in open(jsonl_path, encoding="utf-8"):
        d = json.loads(line)
        instr = d.get("instruction", "")
        if instr.startswith("题目："):
            q = instr[3:].split("\n选项：")[0]
            if q not in seen:
                seen.add(q)
                success_questions.append(q)

    # 3. 重建 done：只保留成功题目对应的 id
    done_ids = set()
    matched = 0
    for q in success_questions:
        if q in q2id:
            done_ids.add(q2id[q])
            matched += 1
        else:
            print(f"[warn] 未匹配到原始题目: {q[:60]}...")

    # 4. 写入干净 done 文件
    done_path = Path(args.out_dir) / "cot.done"
    done_path.write_text("\n".join(sorted(done_ids)) + "\n", encoding="utf-8")
    print(f"[done] {args.name}: 成功题目 {len(success_questions)} 条, 匹配原始id {matched} 条")
    print(f"      干净 done 文件: {done_path} ({len(done_ids)} 条) -> 续传将跳过这些")


if __name__ == "__main__":
    main()
