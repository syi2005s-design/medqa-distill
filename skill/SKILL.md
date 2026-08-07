---
name: medical-data-distillation
description: "Distill medical QA via LLM APIs into SFT data; train LoRA."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [medical, distillation, SFT, LoRA, MedQA, API, data-engineering]
    related_skills: [medical-model-training, huggingface-hub, weights-and-biases]
---

# 医学数据 API 蒸馏 + LoRA 训练

用大模型 API（教师模型：DeepSeek/NVIDIA/Agnes 等）把 MedQA 等医学题库蒸馏成高质量中文推理 SFT 数据，再用 transformers + LoRA 微调医学模型（数据 → 训练 → 评测全链路）。

## 触发场景
- 手上有 MedQA 等医学题库（多选，只有答案没有推理），想生成"逐步推理+答案"的训练数据
- 需要扩充医学训练数据（Self-Instruct 新题、CoT 蒸馏、DPO 偏好对）
- 蒸馏完成后要衔接 LoRA 训练并评测 MedQA 准确率

## 工作流（4 步）

### 1. 数据转换（MedQA data_clean → 蒸馏输入）
```bash
python medqa_to_input.py   # 输出 {"id","question","A".."E","answer"} JSONL 到 data/
```
注意：MedQA 美/中原始题均为 **A–E 五选项**；`4_options/` 下有 4 选项版（推荐 SFT 用）。

### 2. API 蒸馏（核心）
```bash
export DISTILL_CONCURRENCY=24          # 并发；48 会被限流，24 稳
python api_distill.py --task cot --input data/us4_train.jsonl --k 1 \
    --out-dir output/us4-full \
    --api-key "$KEY" --base-url https://api.deepseek.com/v1 --model deepseek-v4-flash
# 其他子命令：--task self-instruct（新题）、--dedup（去重）、--stats（统计）
```
- 全中文推理：system prompt 里"始终使用中文回答（无论题目语言）"（目标中文医学模型时）
- k=3 自洽性过滤：同一题采样 3 次多数投票，不一致丢弃（成本 ×3，先抽样 10% 测一致性率再决定）
- 断点续传：`.done` 文件记录已完成 id，重跑自动跳过，不重复计费
- 过滤规则：答案字母必须与官方答案一致；JSON 解析 4 级回退（整体JSON→围栏块→"answer":"X"→答案是X）

### 3. 训练（云端 GPU 或本机小模型）
```bash
pip install torch transformers peft accelerate datasets
python train_lora.py --data output/us4-full/cot.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --eval-file data/us4_dev.jsonl --output output/model --merge
```
- 模板一致性铁律：训练/评测的 prompt 拼装必须与蒸馏时 instruction 完全一致（`build_user_prompt`）
- 评测：MedQA 多选准确率（选项字母 next-token 概率 argmax，无生成开销）
- 混合比例建议：原题 30-50% + 蒸馏 CoT 30-50% + 通用指令 10-20%（防灾难性遗忘）
- 转 GGUF：对 `--merge` 输出的 merged 权重用 llama.cpp convert_hf_to_gguf.py

### 4. 成本与预算
- 10,178 题 k=1（DeepSeek v4-flash）：约 ¥20-25，产出 ~8,700 条，耗时 ~2 小时（24 并发）
- 27,400 题中文：约 ¥60-90
- 省钱：批量用 DeepSeek/Agnes flash，GPT-4o/Claude 只做精标抽检；先 --limit 200 试跑

## 已知坑（实测）
1. **NVIDIA API 强制 TLS1.2**：`DISTILL_TLS12=1`，否则 Python TLS1.3 握手被网络重置（curl 正常）
2. **Hermes 桌面后台任务双执行**（同一命令两个进程）：脚本内置 mkdir 原子锁（out-dir/.lockdir + 心跳 hb 文件，10 分钟过期可抢占），重复派生会被拒；进程被杀后 1 分钟内 Hermes 会重新派生，属正常
3. **客户端必须设硬超时**：`asyncio.wait_for(..., timeout=90)` 包裹每次 API 调用——无超时→连接挂起→占死并发槽位→整批 gather 永不落盘；超时后走重试
4. **推理类模型**（deepseek-v4-flash/agnes-2.5 系）：`reasoning_content` 是思考、`content` 才是答案；max_tokens 太小（<2048）会只剩思考导致空回复；**content 与 reasoning 全空 = API 空响应（限流/截断），应 raise 触发重试，不要当"格式错"永久 skip**
5. **BATCH 不宜过大**：脚本内 gather 批次默认 16（原来 64）——批次内最慢请求拖累整批落盘；16 条一批，慢请求只影响 16 条
6. **高并发会打爆 provider**：4 片×4 并发=16 并发导致 API 503"服务繁忙"全体假死；稳定配置是**总并发 ≤8**（如 2 片×4）
7. **锁竞态窗口**：mkdir 成功与写 hb 之间若被镜像进程抢占，旧进程会误以为持锁继续跑（双写）。修复：mkdir 后写 hb 失败即 `sys.exit(0)` 让路；touch_lock 失败（锁丢失）也自杀（code=4）
8. **tiktoken 会从 Hermes venv 丢失**（Hermes 更新清包）：`uv pip install tiktoken --python <hermes venv python>` 重装
9. **done 文件会被污染**：skip（含 API 失败）也写 done，余额不足时大量未成功条目被误标完成。用 `rebuild_done.py` 从 cot.jsonl 重建干净 done（只保留真实成功题目）
10. **agnes 输出格式**："正文 + 尾部```json 块"，解析器已支持；agnes-com（apihub.agnes-ai.com）国内网络不可达
11. **锁残留**：前台跑过测试后 `.lockdir` 会残留，重启前 `rm -rf output/*/.lockdir`
12. 失败重跑不要慌：`.done` 续传 + 末尾去重（`--dedup`）保证数据最终正确
13. **dedup 只用于 Self-Instruct 新题**：CoT 蒸馏（一题一条）天然唯一无需去重；旧版 dedup 对 8000+ 条是 O(n²) 会跑 1 小时+，已加长度粗筛优化（~40 倍提速）
14. **分片并行提速**：单进程慢时按题目分片（`data/split/zh_partN.jsonl`），每片独立 out-dir 并行跑；完成后按 instruction 全文去重合并

## 脚本清单（本 skill scripts/ 目录，共 18 个）
- `medqa_to_input.py`：MedQA → 蒸馏输入格式
- `api_distill.py`：蒸馏主脚本（并发/重试/续传/锁/解析/去重/统计/wait_for 硬超时）
- `rebuild_done.py`：从 cot.jsonl 重建干净 done（剔除被余额不足污染的 skip 条目）
- `merge_all_data.py`：合并所有 provider 蒸馏数据 → train/eval（含 answer 提取）
- `merge_token_data.py`：合并新 provider 蒸馏数据（自动识别 us/zh 来源、补 answer、去重、拆 train/eval）
- `merge_with_token.py`：合并全部来源（deepseek/nvidia/agnes/tokenrhythm）→ 训练集（推荐用这个）
- `merge_data.py`：基础合并工具
- `diag_dedup.py`：去重性能诊断
- `train_lora.py`：LoRA SFT + MedQA 评测
- `run_all.sh`：多 provider 依次试水（冒烟+正式+去重）
- `run_train.sh` / `start_training.sh`：本地/云端训练启动
- `prepare_training.sh`：一键准备训练数据
- `start_parts.sh`：分片并行蒸馏启动
- `watchdog.sh` / `token_distill_watchdog.sh`：蒸馏监控（假死检测 + 自动重启/合并）
- `verify_providers.py` / `deep_probe.py`：API provider 体检

## 验证
- 蒸馏后：`python api_distill.py --stats --input output/us4-full/cot.jsonl` 应显示 ~500 token/条
- 训练后：`[eval] MedQA 准确率: 0.4x`（0.5B 小模型 ~0.3，7B LoRA ~0.45-0.55，基座 7B ~0.4 为参考基线）
