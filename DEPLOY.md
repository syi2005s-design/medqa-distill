# 云端训练部署包
# 用于 AutoDL / RunPod / 其他云 GPU 平台

## 上传文件清单

```
E:\skill\test\
├── data/sft/
│   ├── train.jsonl      # 训练集（7,891 条）
│   ├── eval.jsonl       # 验证集（878 条）
│   └── merged.jsonl     # 合并数据（8,769 条）
├── output/us4-full/cot.jsonl  # 原始蒸馏数据（8,714 条）
├── api_distill.py       # 蒸馏脚本
├── train_lora.py        # 训练脚本
├── merge_data.py        # 合并脚本
├── start_training.sh    # 启动脚本
└── README.md
```

## AutoDL 部署步骤

1. **登录 AutoDL** (https://www.autodl.com)
2. **租用 GPU**：推荐 RTX 3090 / A100（12GB+ 显存）
3. **上传文件**：
   ```bash
   # 在 AutoDL 终端执行
   mkdir -p /workspace
   # 上传本地文件到 /workspace/
   ```
4. **运行训练**：
   ```bash
   cd /workspace
   bash start_training.sh
   ```

## 预期结果

- 训练时间：约 2-3 小时（RTX 3090）
- 输出模型：`output/model/`（LoRA 权重）
- 合并模型：`output/model/merged/`（可转 GGUF）
- 评测结果：MedQA 准确率

## 成本预估

- AutoDL 租金：约 ¥15-30/小时
- 训练时长：2-3 小时
- 总成本：约 ¥30-90

## 备选方案：本地 CPU 推理

如果不想用云端，可以用 `llama.cpp` 推理现有模型：

```bash
# 1. 转 GGUF（需要已训练好的模型）
python convert_hf_to_gguf.py output/model/merged/

# 2. 推理
./main -m model.gguf -p "题目：..."
```
