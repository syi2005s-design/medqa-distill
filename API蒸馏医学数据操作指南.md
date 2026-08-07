# API 蒸馏医学数据：完整操作指南

> 版本 1.0 · 适用场景：用大模型 API（教师模型）把 MedQA 等原始医学数据蒸馏成高质量 SFT/DPO 训练数据，用于本地微调医学小模型（transformers → GGUF 全链路）。
> 配套脚本：`api_distill.py`（同目录，可直接运行）。

---

## 1. 什么是 API 蒸馏（先读这节）

**知识蒸馏**是把大模型的"知识"迁移给小模型（软标签/logits）；**API 蒸馏**（本指南所指）是数据层面的蒸馏：**调用教师模型 API，把原始数据改写成更高质量的训练样本**——补上推理过程、扩充题目数量、生成偏好对，再用这批数据微调学生模型。

为什么医学数据尤其需要它：

- MedQA 训练集只有 **1.2 万道**多选题 + 答案，没有推理过程；直接微调只能学会"选答案"的格式，学不到临床推理。
- 医学专家标注一条带推理的问答要 10~30 元；API 生成一条只要几厘到几分钱。
- 教师模型（GPT-4o / DeepSeek / Qwen / GLM）在 USMLE 类题目上准确率已超过多数人类考生，生成内容可信度足够做训练数据（**但仍需过滤+抽检**，见第 5 节）。

### 三条蒸馏路线，先选型

| 路线 | 输入 | 输出 | 扩量倍数 | 提升方向 | 成本 |
|---|---|---|---|---|---|
| **A. CoT 响应蒸馏** | 现有题+答案 | 逐步推理+答案 | 1×（改质） | 临床推理能力 | 低 |
| **B. Self-Instruct 新题** | 种子题 | 新题+推理+答案 | 5~20× | 知识覆盖面 | 中 |
| **C. Evol-Instruct 进化** | 简单题 | 更难的新题 | 2~5× | 推理难度 | 中 |
| **D. 偏好蒸馏 (DPO)** | 题目 | 好答/坏答对 | 1× | 对齐、抑制幻觉 | 中高 |

**推荐组合**：先跑 A（跑通管线）→ 再跑 B 扩量 → 数据够多后用 D 做一轮偏好对齐。C 可选。

---

## 2. 前置准备

### 2.1 数据源

| 数据 | 规模 | 语言 | 用途 |
|---|---|---|---|
| MedQA (train) | 12,723 题 | 英/中 | 种子题 + 评测（你已有） |
| MedMCQA | 183k 题 | 英 | CoT 蒸馏主力 |
| PubMedQA | 1k 专家标注 | 英 | 开放问答蒸馏 |
| CMB / 中文 MedQA | ~25 万条 | 中 | 中文模型 |
| 教材/临床指南段落 | 自备 | 任 | 知识点问答生成（**改写**，勿整段复制） |

### 2.2 红线（先记住）

- ⛔ **真实患者数据（EHR/病历）一律不上第三方 API**——HIPAA / 个人信息保护法红线。要用必须先去标识化，或只使用公开数据。
- ⚠️ 教材/指南受版权保护：只提取**事实性知识点**并改写，不复制原文段落。
- ⚠️ MedQA 等数据集仅限**研究用途**。

### 2.3 环境搭建

```bash
cd /e/skill/test
python -m venv .venv
source .venv/Scripts/activate        # Windows git-bash；PowerShell 用 .venv\Scripts\Activate.ps1
pip install openai tiktoken pandas
```

### 2.1b 本机数据（已就绪）

MedQA `data_clean` 已解压到 `E:\skill\data_clean`，转换后的蒸馏输入已生成在 `E:\skill\test\data\`：

| 文件 | 规模 | 说明 |
|---|---|---|
| `us_train/dev/test.jsonl` | 10,178 / 1,272 / 1,273 | 美国题，**A–E 五选项** |
| `us4_train/dev/test.jsonl` | 同上 | 美国题 4 选项版（推荐做 SFT） |
| `zh_train/dev/test.jsonl` | 27,400 / 3,425 / 3,426 | 中文题（大陆），A–E 五选项 |

重新转换用：`python medqa_to_input.py`（输出 A–E 全字段，脚本已支持）。

### 2.4 选择 API Provider（OpenAI 兼容端点）

| Provider | base_url | 性价比 | 建议用途 |
|---|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | ⭐⭐⭐⭐⭐ | 批量主力 |
| 阿里百炼 Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | ⭐⭐⭐⭐ | 批量主力 |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | ⭐⭐⭐⭐ | 批量/中文 |
| SiliconFlow | `https://api.siliconflow.cn/v1` | ⭐⭐⭐⭐ | 便宜走量 |
| OpenAI GPT-4o | `https://api.openai.com/v1` | ⭐⭐ | **只做精标/抽检** |
| 本地 Ollama | `http://localhost:11434/v1` | 免费 | 先跑通流程再花钱 |

创建 `.env`：

```bash
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
DISTILL_MODEL=deepseek-chat
```

---

## 3. 总体流程

```
原始数据(JSONL) → 清洗 → 提示词模板 → API 批量生成(并发+重试+断点续传)
   → 质量过滤(格式/自洽性/去重/抽检) → 格式化(Alpaca JSONL)
   → SFT 微调(transformers+wandb) → MedQA 评测 → 迭代
```

**黄金法则**：任何一步都用脚本做，输出全部落盘 JSONL；生成脚本必须支持断点续传（失败重跑不重复花钱）。

---

## 4. 生成阶段

### 4.1 异步 API 客户端（并发 + 重试 + 限速）

核心要点：

- `AsyncOpenAI` + `Semaphore(8)` 控制并发（多数 provider 免费额度 RPM=60，付费档 1000+）。
- 指数退避重试（429/5xx/超时）。
- 每完成一条立即 `flush()` 写盘；ID 记入 `.done` 文件，重跑自动跳过。

完整代码见 `api_distill.py`，这里贴核心：

```python
import asyncio, os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_BASE_URL"))
sem = asyncio.Semaphore(8)   # 并发上限

async def call_llm(messages, temperature=0.7, max_tokens=2048, retries=4):
    for attempt in range(retries):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=os.getenv("DISTILL_MODEL"),
                    messages=messages, temperature=temperature,
                    max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            await asyncio.sleep(2 ** attempt + 0.5)  # 指数退避
    raise RuntimeError(f"API 失败: {messages[1]['content'][:50]}...")
```

### 4.2 任务 A：CoT 推理蒸馏（必做）

给每道多选题生成"逐步推理 + 答案"，同时要求 **JSON 输出**方便解析：

```
system: 你是一名严谨的医学专家。回答医学问题时：1) 先给出逐步推理（病理机制、
鉴别诊断、排除理由）；2) 必须输出 JSON：{"reasoning": "...", "answer": "D"}，
answer 为选项字母。只输出 JSON。
user: 题目：{question}
选项：
A. ...
B. ...
C. ...
D. ...
```

解析时先试 `json.loads`（剥掉 ```json 围栏），失败再用正则 `答案[:：]\s*([A-D])` 兜底。

### 4.3 任务 B：Self-Instruct 新题生成（两轮）

第一轮让模型**命题**（同考点新表述），第二轮让模型**解题**：

```
system: 你是医学题库命题专家。根据原题生成一道考点相同、表述不同的新题
（难度相近或略高），只输出新题目和 A-D 四个选项，不要答案。
user: 原题：{question} ...（含原选项）
```

拿到新题后，用任务 A 的模板让模型解题。**可选增强**：把新题再喂给另一家 provider 解题，两家答案一致才保留（跨模型交叉验证，成本翻倍但质量最高）。

### 4.4 任务 C：Evol-Instruct 复杂度进化

```
请在保持考点不变的前提下，把题目改得更难：
- 增加一个干扰选项（症状相似但机制不同的疾病）；
- 要求结合实验室检查结果判断；
- 把单一问题改成"先诊断、再选下一步处理"的两步问题。
只输出新题目和选项。
```

### 4.5 任务 D：DPO 偏好对

```
system: 你是一名医学教育专家。针对下面这道题，生成一对回答：
"chosen": 严谨、循证、结构清晰的满分回答；
"rejected": 含一个常见临床错误或过于武断的低分回答；
"reason": 说明 rejected 错在哪里。
输出 JSON：{"chosen": "...", "rejected": "...", "reason": "..."}
```

---

## 5. 质量过滤阶段（决定数据上限的一步）

### 5.1 格式与合法性过滤（零成本，必做）

- JSON 能解析、`answer` ∈ {A,B,C,D}；
- 输出长度 50~2000 字符（过短=敷衍，过长=离题）；
- 无 "作为AI/我不能" 等残留；
- 推理文本与标准答案**字母一致**才保留（蒸馏数据里不能出现"推理选 A、答案是 B"的矛盾样本）。

### 5.2 自洽性过滤（核心，强烈建议）

同一道题采样 **k=3 次**（temperature 0.7），取答案字母多数投票：

- 3 次全一致 → 高质量，保留多数派第一条推理；
- 2:1 → 保留；
- 意见分裂（1:1:1）→ **丢弃**（模型都不确定，别教给学生）。

代价是生成成本 ×3，所以可以**全量 k=1 先跑，随机 10% 做 k=3 校验**，测得一致性率：

- ≥85% → 模型对这个数据集很稳，直接 k=1 全量跑；
- 70~85% → 全量 k=3；
- <70% → 换更强/更便宜的模型再测。

### 5.3 去重

Self-Instruct 生成的新题高度重复。无需 embedding，字符 3-gram Jaccard 相似度 > 0.85 即判重：

```python
import re
def ngrams(s, n=3):
    s = re.sub(r"\s+", "", s)
    return set(s[i:i+n] for i in range(len(s)-n+1))

def jaccard(a, b):
    if not a or not b: return 0
    return len(a & b) / len(a | b)
```

顺序扫描，新样本与已保留样本相似度 > 0.85 就丢。10 万条级别 O(n²) 太慢，可先按首字符分桶再桶内比较。

### 5.4 Round-trip 校验（可选，成本翻倍）

让模型根据生成的"推理"反推题目，与原文相似度 < 0.5 则丢弃（防止模型自说自话跑题）。

### 5.5 人工抽检（不可省略）

随机抽 **5%（至少 200 条）**，按表打分：

| 维度 | 标准 |
|---|---|
| 医学正确性 | 推理无原则性错误（病理/药理/剂量） |
| 推理质量 | 有步骤、有鉴别诊断，不是空话 |
| 格式 | JSON 合法、答案与推理一致 |
| 重复率 | 与已见样本不重复 |

正确率 < 90% → 调整提示词或换模型，重新生成。

---

## 6. 格式化与统计

统一 Alpaca 格式（与你的 transformers 训练模板一致）：

```json
{"instruction": "题目：...\n选项：\nA. ...\nB. ...\nC. ...\nD. ...", "input": "", "output": "逐步推理...\n答案：D"}
```

**⚠️ 模板一致性是最大坑**：蒸馏时 instruction 的拼装方式，必须与训练/推理时完全一致，否则微调后模型答非所问。

token 统计（估成本/设 max_seq_len）：

```python
import tiktoken, json
enc = tiktoken.get_encoding("cl100k_base")
n = total = 0
for l in open("cot.jsonl", encoding="utf-8"):
    d = json.loads(l); n += 1
    total += len(enc.encode(d["instruction"] + "\n" + d["output"]))
print(f"{n} 条, {total} token, 平均 {total//n}/条")
```

---

## 7. 成本控制

参考价（2026-08 约价，**以官网为准**）：

| 模型 | 输入 / 百万 token | 输出 / 百万 token |
|---|---|---|
| DeepSeek-chat | ~$0.27 | ~$1.1 |
| Qwen-Plus (百炼) | ~$0.11 | ~$0.28 |
| GLM-4.5-Air | ~$0.14 | ~$0.28 |
| GPT-4o | ~$2.5 | ~$10 |
| Claude Sonnet 4 | ~$3 | ~$15 |

**估算公式**：`总成本 ≈ 条数 × (输入token×输入价 + 输出token×输出价) / 1e6`

例：10 万条 CoT，每条输入 ~300 token + 输出 ~600 token：

- DeepSeek：100k×(300×0.27 + 600×1.1)/1e6 ≈ **$74**
- GPT-4o：同样算法 ≈ **$675**（k=3 自洽再 ×3）

**省钱策略**：

1. 批量生成用 DeepSeek/Qwen/GLM，**GPT-4o/Claude 只用来精标 1~2 万条 + 抽检**；
2. 自洽性 k=3 只做抽样（10%），测得一致性率再决定全量策略；
3. 先拿 200 条跑通全流程，确认格式和质量再全量；
4. 输出限长（`max_tokens=2048` 足够，别浪费）。

---

## 8. 微调集成（衔接你现有 transformers 流程）

- **数据混合比例**：原题(格式) 30~50% + CoT 蒸馏 30~50% + 通用指令 10~20%（防止灾难性遗忘，保留基本对话能力）。
- **超参**（7B 量级）：epochs 2~3、lr 1e-5~2e-5、`max_seq_len 2048`、batch 按显存调（你的 4×8 梯度累积配置即可）、`report_to="wandb"` 不变。
- **评测集固定**：MedQA 验证集 500 题，每个 epoch 后都测，对比基线。
- 蒸馏数据质量 > 数量：10 万条好数据 > 50 万条脏数据。宁可过滤狠一点。
- 训练完成后照旧转 GGUF（`convert.py`）+ `llama-server` 本地推理。

---

## 9. 评测与迭代

**Ablation 对照**（回答"蒸馏数据到底有没有用"）：

| 实验 | 训练数据 | MedQA acc |
|---|---|---|
| 基线 | 原题 100% | X% |
| +CoT | 原题 50% + CoT 50% | ? |
| +新题 | 原题 30% + CoT 30% + 新题 40% | ? |
| +DPO | 上面 + DPO 一轮 | ? |

每轮只改一个变量。医学推理提升看 MedQA acc，额外看长回答质量（人工打分）。

**常见失败模式**：

| 现象 | 原因 | 对策 |
|---|---|---|
| 微调后乱答 | 训练模板 ≠ 蒸馏模板 | 统一模板 |
| 答案分布偏 A | 数据不平衡 | 按选项字母分层采样 |
| 输出全是套话 | CoT 质量差 | 换模型/加 few-shot/降 k 门槛 |
| 推理对但答案错 | 教师模型幻觉 | 自洽性过滤 + 与标准答案比对 |

---

## 10. 合规与伦理

1. 不上传真实患者数据（见 2.2）；
2. 生成内容可能含医学错误——**禁止直接用于临床决策**，训练/演示需带免责声明；
3. 发布模型/数据前检查数据源许可（MedQA 仅研究用途）；
4. 记录生成批次、prompt 版本、过滤参数（ reproducibility 需要）。

---

## 11. FAQ / 排错

| 问题 | 解决 |
|---|---|
| 429 Too Many Requests | 降 `CONCURRENCY` 到 2~4；确认 RPM/TPM 配额 |
| 断点续传 | `.done` 文件记录已完成 ID，重跑自动跳过（脚本已实现） |
| 输出 JSON 解析失败率高 | 提示词里给 1 个 JSON 示例（few-shot）；用正则兜底 |
| 中文数据 | 换中文 system prompt + CMB/中文 MedQA 子集；检查输出语言 |
| 想用 Claude | base_url 走 Anthropic 兼容层或中转；成本高只做精标 |
| 本地先试 | Ollama 起个 14B 模型，`base_url=http://localhost:11434/v1` 零成本跑通 |

---

## 附录 A：配套脚本（本目录 `E:\skill\test`）

- `api_distill.py` —— CoT 蒸馏（JSON 解析+正则兜底）、k=3 自洽性过滤、Self-Instruct 新题生成、n-gram 去重、token 统计、断点续传。**支持 A–E 五选项**（MedQA 美/中原始题都是 5 选）。自动读 `.env`，可用 `--api-key/--base-url/--model/--out-dir` 覆盖。
- `medqa_to_input.py` —— MedQA 原始 JSONL → 蒸馏输入格式转换器。
- `verify_providers.py` / `deep_probe.py` —— provider 体检（模型列表 + chat 测试）。
- `run_all.sh` —— 一键挨个跑所有 provider（单实例锁 + 断点续传 + 自动去重）。
- `data/` —— 转换好的输入数据（`us4_train.jsonl` 4 选项版优先用于 SFT）。
- `.env` —— 4 组 API 配置（deepseek / nvidia / agnes-cn / agnes-com）。

已知坑（实测踩过）：

- **NVIDIA 端点要求 TLS1.2**：`DISTILL_TLS12=1` 环境变量（Python 默认 TLS1.3 握手会被网络重置，curl/浏览器正常）。
- **Agnes-COM（apihub.agnes-ai.com）从国内网络不可达**（443 连接超时）。
- **推理类模型**（deepseek-v4 / agnes-2.5 系）回复有 `reasoning_content`，`content` 才是最终答案；max_tokens 太小会只剩 thinking 导致空回复。
- 后台批量任务注意防重复执行：单实例锁（mkdir）+ `.done` 断点续传 + 结束后去重。

用法：

```bash
source .venv/Scripts/activate
# 先试跑 200 条（推荐）：
python api_distill.py --task cot --input data/us4_train.jsonl --k 3 --limit 200
# 全量：
python api_distill.py --task cot --input data/us4_train.jsonl --k 3
python api_distill.py --task self-instruct --input data/us4_train.jsonl --limit 5000
python api_distill.py --dedup --input output/cot.jsonl
python api_distill.py --stats --input output/cot.jsonl
```

输入格式（一行一个 JSON）：`{"id": "...", "question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "answer": "D"}`

## 附录 C：试水实测结果（2026-08，同一批 200 题，k=1，全中文推理）

| Provider | 模型 | 成功/尝试 | 成功率 | 平均长度 | 备注 |
|---|---|---|---|---|---|
| DeepSeek 直连 | deepseek-v4-flash | 177/200 | 88.5% | 493 token | 性价比最高，主力推荐 |
| NVIDIA | deepseek-ai/deepseek-v4-flash | 89/100 | 89% | 637 token | 需 `DISTILL_TLS12=1`；偶发 529 过载；延迟较高 |
| Agnes-CN | agnes-2.5-flash | 178/200 | 89% | 437 token | 免费额度有 RPM 限流；2.5-pro 需 `--max-tokens 4096+` |
| Agnes-COM | — | — | — | — | 网络不可达 |

三家过滤掉的 ~11% 主要为"模型答案与官方答案不一致"（同一道题三家都错，多为题目本身有争议或选项重复），少量为格式解析失败。输出均为完整中文临床推理（病理机制+鉴别诊断+排除理由）。

## 附录 B：模板速查

- CoT 蒸馏：`{"instruction": 题目+选项, "input": "", "output": 推理+"答案：X"}`
- 训练模板（transformers chat template）必须与 instruction 拼装一致
- DPO 数据：`{"prompt": 题, "chosen": 好回答, "rejected": 坏回答}`
