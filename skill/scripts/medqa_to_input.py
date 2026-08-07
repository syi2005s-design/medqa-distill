#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medqa_to_input.py —— 把 MedQA data_clean 原始 JSONL 转为 api_distill.py 的输入格式

原始格式:  {"question": "...", "options": {"A": "...", "B": "..."}, "answer": "...", "answer_idx": "D", "meta_info": "..."}
输出格式:  {"id": "...", "question": "...", "A": "...", ..., "E": "...", "answer": "D"}

用法: python medqa_to_input.py [--src E:\\skill\\data_clean\\questions] [--dst E:\\skill\\test\\data]
"""
import argparse
import json
from pathlib import Path

DEFAULT_SRC = Path(r"E:\skill\data_clean\questions")
DEFAULT_DST = Path(r"E:\skill\test\data")

# (源文件相对 questions/ 的路径, 输出文件名)
PAIRS = [
    ("US/train.jsonl",                       "us_train.jsonl"),
    ("US/dev.jsonl",                         "us_dev.jsonl"),
    ("US/test.jsonl",                        "us_test.jsonl"),
    ("US/4_options/phrases_no_exclude_train.jsonl", "us4_train.jsonl"),
    ("US/4_options/phrases_no_exclude_dev.jsonl",   "us4_dev.jsonl"),
    ("US/4_options/phrases_no_exclude_test.jsonl",  "us4_test.jsonl"),
    ("Mainland/train.jsonl",                 "zh_train.jsonl"),
    ("Mainland/dev.jsonl",                   "zh_dev.jsonl"),
    ("Mainland/test.jsonl",                  "zh_test.jsonl"),
]


def convert(src_file: Path, out_file: Path):
    items = []
    with open(src_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            opts = d.get("options", {}) or {}
            item = {
                "id": f"{out_file.stem}-{i:06d}",
                "question": d.get("question", ""),
                "answer": str(d.get("answer_idx", "")).strip().upper(),
            }
            for k in "ABCDE":
                if k in opts and opts[k]:
                    item[k] = opts[k]
            # 保留题目来源信息，方便溯源
            item["meta_info"] = d.get("meta_info", "")
            items.append(item)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_opt = {k: sum(1 for x in items if k in x) for k in "ABCDE"}
    print(f"[ok] {out_file.name}: {len(items)} 条, 选项分布 {n_opt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    for rel, name in PAIRS:
        s = src / rel
        if s.exists():
            convert(s, dst / name)
        else:
            print(f"[skip] 不存在: {s}")


if __name__ == "__main__":
    main()
