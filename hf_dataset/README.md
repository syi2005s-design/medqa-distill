---
license: mit
language:
  - zh
  - en
task_categories:
  - multiple-choice
  - question-answering
tags:
  - medical
  - MedQA
  - distillation
  - SFT
  - reasoning
  - chinese
pretty_name: MedQA-Distill-SFT
---

# MedQA-Distill-SFT

**医学多选题 → 中文推理 SFT 数据**（由 MedQA 题库经 LLM API 蒸馏生成）

从 MedQA（美国 USMLE / 中国执业医师考试）多选题蒸馏而来：每题包含题目、选项、**逐步临床推理（中文）** 和答案。用于微调中文医学大模型（SFT / LoRA）。

## 数据统计

| 字段 | 数值 |
|------|------|
| 总条数 | 39,618 |
| 训练集 | 35,656 |
| 验证集 | 3,962 |
| 来源 Provider | DeepSeek / TokenRhythm / Agnes-CN / NVIDIA |
| 推理语言 | 中文（英文题也生成中文推理） |

## 数据格式（Alpaca）

```json
{
  "instruction": "题目：A 23-year-old pregnant woman...\n选项：\nA. ...\nB. ...\nC. ...\nD. ...\nE. ...",
  "input": "",
  "output": "患者为孕22周孕妇，主诉排尿烧灼感，符合急性单纯性膀胱炎...\n答案：A",
  "source": "tokenrhythm-us",
  "answer": "A"
}
```

## 来源分布

| source | 条数 | 说明 |
|--------|------|------|
| deepseek-us4 | 8,714 | 美国 MedQA（DeepSeek 蒸馏） |
| deepseek-zh | 4,613 | 中国 MedQA（DeepSeek 蒸馏） |
| tokenrhythm-us | 10,421 | 美国 MedQA（TokenRhythm 补充） |
| tokenrhythm-zh | 15,751 | 中国 MedQA（TokenRhythm 补充） |
| agnes-cn / nvidia | 119 | 补充 |

## 使用

```python
from datasets import load_dataset

ds = load_dataset("syi2005s-design/medqa-distill-sft")
print(ds["train"][0])
```

## 生成工具

配套开源工具链：[medqa-distill](https://github.com/syi2005s-design/medqa-distill)（MIT License）

## 免责声明

- 仅用于**研究/教育**目的，禁止用于临床决策
- 数据由 LLM 生成，可能存在医学错误，使用前请人工抽检
- MedQA 原始数据集仅限研究用途
