#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedQA 数据转换工具

将原始 MedQA 数据转换为蒸馏格式。
"""
import json
import argparse
import os


def convert_us(raw_path, output_path):
    """转换 US MedQA 格式（A-E 五选项）。"""
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "id": d.get("question_concept_id", ""),
                "question": d.get("question", ""),
                "A": d.get("option_a", ""),
                "B": d.get("option_b", ""),
                "C": d.get("option_c", ""),
                "D": d.get("option_d", ""),
                "E": d.get("option_e", ""),
                "answer": d.get("correct_option", "").strip()
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"US MedQA: {len(rows)} 条 -> {output_path}")
    return rows


def convert_zh(raw_path, output_path):
    """转换 Mainland MedQA 格式（A-E 五选项）。"""
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                "id": d.get("question_concept_id", ""),
                "question": d.get("question", ""),
                "A": d.get("option_a", ""),
                "B": d.get("option_b", ""),
                "C": d.get("option_c", ""),
                "D": d.get("option_d", ""),
                "E": d.get("option_e", ""),
                "answer": d.get("correct_option", "").strip()
            })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Zh MedQA: {len(rows)} 条 -> {output_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="MedQA 数据转换")
    parser.add_argument("--input-dir", required=True, help="原始 MedQA 数据目录")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换 US 数据
    if os.path.exists(f"{args.input_dir}/questions/US/train.jsonl"):
        convert_us(f"{args.input_dir}/questions/US/train.jsonl", f"{args.output_dir}/us4_train.jsonl")
        convert_us(f"{args.input_dir}/questions/US/dev.jsonl", f"{args.output_dir}/us4_dev.jsonl")
        convert_us(f"{args.input_dir}/questions/US/test.jsonl", f"{args.output_dir}/us4_test.jsonl")
    
    # 转换中国数据
    if os.path.exists(f"{args.input_dir}/questions/Mainland/train.jsonl"):
        convert_zh(f"{args.input_dir}/questions/Mainland/train.jsonl", f"{args.output_dir}/zh_train.jsonl")
        convert_zh(f"{args.input_dir}/questions/Mainland/dev.jsonl", f"{args.output_dir}/zh_dev.jsonl")
        convert_zh(f"{args.input_dir}/questions/Mainland/test.jsonl", f"{args.output_dir}/zh_test.jsonl")


if __name__ == "__main__":
    main()
