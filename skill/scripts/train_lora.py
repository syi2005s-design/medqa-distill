#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_lora.py —— 用 API 蒸馏数据微调医学 LLM（LoRA SFT + MedQA 评测）

用法（Linux GPU / Windows CPU 均可）：
  pip install torch transformers peft accelerate datasets
  python train_lora.py \
      --data output/us4-full/cot.jsonl \
      --model Qwen/Qwen2.5-7B-Instruct \
      --eval-file data/us4_dev.jsonl \
      --output output/model

评测：MedQA 多选题准确率（选项字母 next-token 概率 argmax，无生成开销）
训练完转 GGUF：llama.cpp 的 convert_hf_to_gguf.py 对 merge 后的模型执行
"""
import argparse
import json
import os
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

OPTION_LETTERS = "ABCDE"
IGNORE = -100


def build_user_prompt(item):
    """必须与蒸馏时 instruction 拼装完全一致（模板一致性是训练效果的关键）。"""
    opts = "\n".join(f"{k}. {item[k]}" for k in OPTION_LETTERS if item.get(k))
    return f"题目：{item['question']}\n选项：\n{opts}"


def load_data(path, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    return rows


def tokenize_row(row, tokenizer, max_len):
    """模板无关的 label masking：user 部分 -100，assistant 输出参与 loss。"""
    user_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_user_prompt(row)}],
        tokenize=False, add_generation_prompt=True)
    full_text = user_text + row["output"]
    enc = tokenizer(full_text, truncation=True, max_length=max_len)
    n_user = len(tokenizer(user_text, truncation=True, max_length=max_len).input_ids)
    labels = [IGNORE] * n_user + enc.input_ids[n_user:]
    if len(labels) < len(enc.input_ids):      # 截断保护
        labels = labels[:len(enc.input_ids)]
    labels = labels + [IGNORE] * (len(enc.input_ids) - len(labels))
    return {"input_ids": enc.input_ids, "attention_mask": enc.attention_mask, "labels": labels}


@torch.no_grad()
def eval_medqa(model, tokenizer, eval_file, limit=200, max_len=1024):
    """MedQA 多选准确率：比较各选项字母作为 next-token 的概率。"""
    device = next(model.parameters()).device
    items = load_data(eval_file, limit)
    letter_ids = {L: tokenizer.convert_tokens_to_ids(L) for L in OPTION_LETTERS}
    correct = total = 0
    model.eval()
    for it in items:
        prompt = build_user_prompt(it)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=max_len).to(device)
        logits = model(**enc).logits[:, -1, :].float()          # next-token logits
        probs = torch.softmax(logits[0], dim=-1)
        scores = {L: probs[letter_ids[L]].item()
                  for L in OPTION_LETTERS if it.get(L) and letter_ids[L] not in (0, -1)}
        if not scores:
            continue
        pred = max(scores, key=scores.get)
        correct += (pred == it["answer"])
        total += 1
    return correct / max(total, 1), correct, total


def main():
    ap = argparse.ArgumentParser(description="医学蒸馏数据 LoRA 微调 + MedQA 评测")
    ap.add_argument("--data", required=True, help="Alpaca 格式蒸馏数据 JSONL")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--eval-file", default=None, help="MedQA 格式评测集（medqa_to_input 输出）")
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--output", default="output/model")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="训练数据上限（试跑用）")
    ap.add_argument("--merge", action="store_true", help="训练后 merge LoRA 保存全量权重")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_data(args.data, args.limit)
    print(f"[data] 训练样本 {len(rows)} 条")
    ds = Dataset.from_list(rows)
    ds = ds.map(lambda r: tokenize_row(r, tokenizer, args.max_len),
                remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None)
    model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
                      lora_alpha=args.lora_alpha,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.05)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_num_workers=0)
    trainer = Trainer(
        model=model, args=train_args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"[save] LoRA 权重已保存: {args.output}")

    if args.merge:
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output, "merged"))
        tokenizer.save_pretrained(os.path.join(args.output, "merged"))
        print(f"[merge] 全量权重已保存: {os.path.join(args.output, 'merged')}（可转 GGUF）")

    if not args.no_eval and args.eval_file and os.path.exists(args.eval_file):
        model.eval()
        acc, c, t = eval_medqa(model, tokenizer, args.eval_file, args.eval_limit)
        print(f"[eval] MedQA 准确率: {acc:.4f} ({c}/{t})")
    elif args.eval_file:
        print(f"[eval] 跳过：评测文件不存在 {args.eval_file}")


if __name__ == "__main__":
    main()
