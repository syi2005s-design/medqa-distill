#!/bin/bash
# run_train.sh —— 本地/云端训练启动脚本

set -e

OUTPUT_DIR="${1:-output/model}"
DATA_DIR="${2:-data/sft}"

echo "=== 医学 LoRA 训练 ==="
echo "数据: $DATA_DIR"
echo "输出: $OUTPUT_DIR"
echo ""

# 检查数据
if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "错误: 找不到训练数据 $DATA_DIR/train.jsonl"
    echo "请先运行: python merge_all_data.py"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATA_DIR/train.jsonl")
echo "训练集: $TRAIN_COUNT 条"

# 安装依赖（如需）
# pip install torch transformers peft accelerate datasets

# 运行训练
python train_lora.py \
    --data "$DATA_DIR/train.jsonl" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output "$OUTPUT_DIR" \
    --epochs 3 \
    --lora-r 64 \
    --lora-alpha 128 \
    --batch 2 \
    --grad-accum 8

echo ""
echo "=== 训练完成 ==="
echo "模型: $OUTPUT_DIR/"
