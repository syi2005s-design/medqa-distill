#!/bin/bash
# 云端训练部署脚本（AutoDL）
# 用法：上传到 AutoDL，运行 bash start_training.sh

set -e

echo "=== 配置训练环境 ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft accelerate datasets wandb

echo "=== 准备数据 ==="
cd /workspace
python merge_data.py --output data/sft/merged.jsonl --split data/sft/train.jsonl data/sft/eval.jsonl

echo "=== 开始训练 ==="
python train_lora.py \
    --data data/sft/train.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --eval-file data/us4_dev.jsonl \
    --output output/model \
    --epochs 3 \
    --lora-r 64 \
    --lora-alpha 128 \
    --batch 4 \
    --grad-accum 8 \
    --merge

echo "=== 训练完成 ==="
echo "模型路径: output/model/"
echo "合并权重: output/model/merged/"
