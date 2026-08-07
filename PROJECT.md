# 医学数据 API 蒸馏项目

## 一句话介绍

用大模型 API 把医学题库（MedQA）蒸馏成高质量推理数据，训练出能给出中文临床推理的医学模型。

---

## 为什么做这个

医学大模型训练数据稀缺，公开题库只有题目和答案，缺少"为什么选这个"的推理过程。本项目把题库"翻译"成带完整临床推理的 SFT 数据：

```
题目：...
选项：A. ... B. ... C. ... D. ... E. ...
↓ 蒸馏 ↓
推理：该患者为孕22周孕妇，主诉排尿烧灼感...
      符合急性单纯性膀胱炎表现。孕期尿路感染首选...
      呋喃妥因对孕期膀胱炎有效且安全性较好...
      因此最佳治疗为呋喃妥因。
答案：D
```

---

## 技术架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MedQA 题库     │────▶│   API 蒸馏       │────▶│  训练数据        │
│  (US + Mainland)│     │  DeepSeek/NVIDIA/│     │  12,000+ 条     │
│   37,578 题     │     │  Agnes-CN       │     │  (Alpaca 格式)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   LoRA 微调      │◀────│   训练脚本       │◀────│  Qwen2.5-7B     │
│  医学模型        │     │  transformers   │     │  (基座模型)      │
│  (可选)          │     └─────────────────┘     └─────────────────┘
└─────────────────┘
```

---

## 核心能力

### 1. 多 Provider 蒸馏
- **DeepSeek**：主力，性价比高（¥0.003/条）
- **NVIDIA**：托管 DeepSeek 模型，需 TLS1.2
- **Agnes-CN**：国内可用，免费额度有限

### 2. 智能解析
- 4 级答案提取回退（整体 JSON → 围栏块 → `"answer":"X"` → "答案是 X"）
- 自动过滤答案不一致的条目

### 3. 断点续传
- `.done` 文件记录已完成 ID
- 重跑自动跳过，不重复计费
- 末尾去重兜底

### 4. 训练数据生成
- 合并多 provider 数据
- 自动去重 + 训练/验证拆分
- Alpaca 格式输出（instruction + output）

---

## 当前成果

### 数据产出
| 来源 | 条数 | 说明 |
|---|---|---|
| DeepSeek（英文） | 8,714 | us4_train，主力 |
| DeepSeek（中文） | 5,120+ | zh_train，运行中 |
| NVIDIA | 170 | 试水 |
| Agnes-CN | 978 | 试水 |
| **合计** | **~14,000** | |

### 训练数据
- **训练集**: 12,033 条
- **验证集**: 1,337 条
- **格式**: Alpaca（instruction + input + output）
- **平均输出**: 208 字符

---

## 使用场景

1. **医学模型微调**：训练能给出临床推理的医学 LLM
2. **数据扩充**：从有限题库生成大量推理数据
3. **多语言适配**：英文题也生成中文推理（目标中文医学模型）
4. **快速验证**：单机可运行，云端可训练

---

## 项目亮点

- ✅ **端到端**：从题库 → 蒸馏 → 训练 → 评测
- ✅ **低成本**：¥20-90 可完成 1-2 万条数据
- ✅ **高复用**：脚本支持任意题库（只需改输入格式）
- ✅ **开源**：MIT License，可自由使用
- ✅ **文档全**：README、部署指南、API 说明

---

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/medical-distill.git
cd medical-distill

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 Key

# 3. 转换数据
python medqa_to_input.py --input-dir /path/to/MedQA --output-dir data

# 4. 蒸馏
python api_distill.py --task cot --input data/us4_train.jsonl \
    --out-dir output/us4-full --api-key "$OPENAI_API_KEY"

# 5. 合并数据
python merge_all_data.py

# 6. 训练（需 GPU）
python train_lora.py --data data/sft/train.jsonl --model Qwen/Qwen2.5-7B-Instruct
```

---

## 技术栈

- **Python 3.11+**
- **transformers** + **PEFT**（LoRA）
- **OpenAI SDK**（多 Provider 兼容）
- **datasets**（数据处理）

---

## License

MIT License
