# 🩺 MedQA-Distill — 医学数据 API 蒸馏工具链

> 用大模型 API 把 MedQA 医学题库蒸馏成 **高质量中文推理 SFT 数据**，再用 LoRA 微调医学模型。
> 从原始题库 → 蒸馏 → 合并 → 训练 → 评测，全链路自动化。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🌟 项目亮点

- **✅ 39,618 条高质量中文推理训练数据**（已开源，见下方数据集）
- **🔬 数据级知识蒸馏**：把"只有答案"的多选题，变成"逐步推理 + 答案"的思维链数据
- **🌏 中文推理**：英文 USMLE 题目也生成中文推理（目标中文医学模型）
- **💸 成本极低**：10,178 题仅 ~¥20（DeepSeek flash 模型）
- **🛡️ 工程健壮**：断点续传、单实例锁、心跳假死保护、自动重试、分片并行
- **📦 多 Provider 支持**：DeepSeek / NVIDIA / Agnes / TokenRhythm 等任意 OpenAI 兼容端点

---

## 📊 数据集

| 数据集 | 条数 | 内容 |
|--------|------|------|
| **train.jsonl** | 35,656 | 题目 + 中文推理 + 答案（Alpaca 格式） |
| **eval.jsonl** | 3,962 | 同格式验证集 |
| **merged_all.jsonl** | 39,618 | 全量合并（含 source/answer 标注） |

**数据来源分布**：

| 来源 | 条数 | 说明 |
|------|------|------|
| DeepSeek (us4) | 8,714 | 美国 MedQA，4 选项 |
| DeepSeek (zh) | 4,661 | 中国 MedQA，5 选项 |
| TokenRhythm (us) | ~10,400 | 美国 MedQA 补充，5 选项 |
| TokenRhythm (zh) | ~15,700 | 中国 MedQA 补充，5 选项 |
| Agnes-CN / NVIDIA | ~120 | 补充（免费额度） |

**数据格式**（Alpaca）：

```json
{
  "instruction": "题目：A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination...\n选项：\nA. ...\nB. ...\nC. ...\nD. ...\nE. ...",
  "input": "",
  "output": "患者为孕22周孕妇，主诉排尿烧灼感，符合急性单纯性膀胱炎的临床表现...\n答案：A",
  "source": "tokenrhythm-us",
  "answer": "A"
}
```

> 📦 **数据集下载**：https://huggingface.co/datasets/rewrewrv343/medqa-distill-sft

---

## 🚀 快速开始

### 1. 安装

```bash
git clone https://github.com/syi2005s-design/medqa-distill.git
cd medqa-distill
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key（任意 OpenAI 兼容端点）
```

### 3. 数据转换（MedQA 原始数据 → 蒸馏输入）

```bash
python medqa_to_input.py --input-dir /path/to/MedQA --output-dir data
```

### 4. 蒸馏

```bash
# 单 Provider 蒸馏
python api_distill.py --task cot --input data/us4_train.jsonl --k 1 \
    --out-dir output/us4-full \
    --api-key "$DEEPSEEK_API_KEY" \
    --base-url https://api.deepseek.com/v1 \
    --model deepseek-v4-flash

# 分片并行（提速 4 倍）
python api_distill.py --task cot --input data/zh_train.jsonl --k 1 \
    --out-dir output/zh-p1 --limit 6850 \
    # ... 每个分片独立 out-dir 并行跑
```

### 5. 合并数据

```bash
# 合并所有蒸馏结果，自动去重、拆 train/eval
python merge_all_data.py
# 或合并新 provider 数据（自动识别 us/zh 来源）
python merge_token_data.py
```

### 6. LoRA 训练

```bash
# 本地 GPU
python train_lora.py \
    --data data/sft/train.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output output/model \
    --epochs 3 --lora-r 64 --lora-alpha 128 --merge

# 云端 GPU（AutoDL 等）
bash run_train.sh
```

### 7. 评测

```bash
python train_lora.py --eval-only \
    --model output/model \
    --eval-file data/us4_dev.jsonl
```

---

## 🧠 蒸馏原理

MedQA 等医学题库只有**题目 + 选项 + 答案字母**，没有推理过程。直接微调只能学到"选答案"的格式，学不到临床推理。

**API 蒸馏**是数据层面的知识迁移：

```
原始题库(题+答案) → 教师模型 API（DeepSeek/GPT-4o）→ 逐步推理+答案 → SFT 训练数据
```

### 核心特性

| 特性 | 说明 |
|------|------|
| **CoT 蒸馏** | 每题生成病理机制、鉴别诊断、排除理由 |
| **中文推理** | System prompt 强制"始终中文回答"，英文题也出中文推理 |
| **答案自洽过滤** | 答案字母必须与官方答案一致，矛盾样本直接丢弃 |
| **k=3 自洽投票** | 同题采样 3 次多数投票，意见分裂丢弃（可选，成本×3） |
| **断点续传** | `.done` 文件记录已完成 id，重跑不重复计费 |
| **单实例锁** | mkdir 原子锁 + 心跳，防镜像进程双写 |
| **假死保护** | 90s 硬超时 + 心跳检查，网络卡死自动退出重启 |
| **分片并行** | 大任务按题分片，多进程并行（4 倍提速） |

---

## 📁 项目结构

```
medqa-distill/
├── api_distill.py              # 蒸馏主脚本（并发/重试/续传/锁/解析/去重/统计）
├── clinical-assistant/         # 临床医学助手 🩺
│   ├── SKILL.md                # 完整技能说明
│   └── clinical_assistant.py   # CLI 命令行工具
├── medqa_to_input.py           # MedQA → 蒸馏输入格式
├── merge_all_data.py           # 合并所有蒸馏数据 → train/eval
├── merge_token_data.py         # 合并新 provider 数据（自动识别 us/zh、补 answer）
├── rebuild_done.py             # 从产出重建干净 done（剔除被污染条目）
├── train_lora.py               # LoRA 微调 + MedQA 评测
├── run_all.sh                  # 多 Provider 依次试水
├── run_train.sh                # 训练启动脚本
├── requirements.txt            # 依赖
├── .env.example                # API Key 配置模板
├── data/                       # 输入数据（MedQA，不入库）
├── data/sft/                   # 训练数据（train/eval/merged_all，托管 HF）
└── output/                     # 蒸馏产出（不入库）
```

---

## 💰 成本参考（实测）

| 任务 | 题数 | 模型 | 成本 | 产出 |
|------|------|------|------|------|
| us4 CoT 蒸馏 | 10,178 | DeepSeek v4-flash | ~¥20 | ~8,700 条 |
| zh CoT 蒸馏（全量） | 27,400 | DeepSeek v4-flash | ~¥60-90 | ~15,000 条 |
| 自洽性 k=3（可选） | — | — | ×3 | 质量更高 |

**省钱策略**：
1. 先 `--limit 200` 试跑，确认格式和质量再全量
2. 批量走 DeepSeek/Agnes flash，GPT-4o/Claude 只做精标抽检
3. 自洽性 k=3 只抽 10% 测一致性率，再决定是否全量
4. 用分片并行 + 断点续传，失败重跑零成本

---

## ⚠️ 已知坑（实测经验）

1. **推理模型空回复**：deepseek-v4-flash 等会把 token 全花在思考上，`content` 为空——脚本已自动从 `reasoning_content` 兜底提取，全空则重试
2. **高并发打爆 API**：16 并发会导致 503"服务繁忙"，稳定配置总并发 ≤8
3. **NVIDIA API 强制 TLS1.2**：设 `DISTILL_TLS12=1`
4. **锁竞态**：镜像进程同时启动会双写——脚本已修复（写 hb 失败即让路）
5. **数据合规**：只蒸馏公开数据集，**不要上传真实患者数据**（HIPAA/个保法红线）
6. **模板一致性铁律**：蒸馏/训练/推理的 prompt 拼装必须完全一致

---

## 🧪 评测基准

| 模型 | MedQA 准确率 |
|------|-------------|
| 7B 基座（参考基线） | ~40% |
| 7B + LoRA 微调 | ~45-55% |
| 0.5B 小模型 | ~30% |

---

## 🤝 贡献

欢迎 PR！建议方向：
- 新 Provider 适配
- Self-Instruct 新题生成增强
- DPO 偏好对蒸馏
- 更多数据集支持（MedMCQA、PubMedQA、CMB）

---

## 📄 License

[MIT](LICENSE) — 自由使用、修改、分发（请遵守 MedQA 数据集的原始许可，仅限研究用途）

## 📚 引用

```bibtex
@software{medqa-distill,
  title = {MedQA-Distill: Medical QA Distillation Toolkit},
  author = {Syi2005s Design},
  year = {2026},
  url = {https://github.com/syi2005s-design/medqa-distill}
}
```

---

## 📦 数据集 (HuggingFace)

训练数据托管在 HuggingFace Hub，GitHub 仓库只含代码：

```
https://huggingface.co/datasets/rewrewrv343/medqa-distill-sft
```

```python
from datasets import load_dataset
ds = load_dataset("rewrewrv343/medqa-distill-sft")
print(ds["train"][0]["instruction"])
```
