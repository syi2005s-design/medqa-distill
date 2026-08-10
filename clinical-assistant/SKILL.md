---
name: clinical-medical-assistant
description: "临床医学助手 — 基于 LLM API 的中文医学问答。支持症状诊断、用药建议、检查解读、鉴别诊断。"
version: 1.0.0
author: medqa-distill
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [medical, clinical, diagnosis, consultation, chinese]
    related_skills: [medical-data-distillation, medical-model-training]
---

# 临床医学助手 🩺

基于大语言模型 API（DeepSeek/TokenRhythm 等）的中文临床医学问答助手。输入症状/检查结果，输出专业的诊断推理、鉴别诊断和治疗建议。

## 触发场景

- 患者描述症状 → 需要鉴别诊断和下一步建议
- 拿到化验/影像检查结果 → 需要解读
- 用药咨询 → 需要药理分析和注意事项
- 病历摘要 → 需要诊断思路和治疗方案

## 使用方法

### 直接问答

```bash
hermes clinical-medical-assistant "患者，男，45岁，突发胸痛2小时，向左臂放射，伴大汗，舌下含服硝酸甘油不缓解，心电图示V1-V4导联ST段抬高。"
```

### 检查解读

```bash
hermes clinical-medical-assistant "解读以下化验结果：WBC 12.5×10^9/L，中性粒细胞85%，CRP 45mg/L，降钙素原0.8ng/mL"
```

### 用药咨询

```bash
hermes clinical-medical-assistant "阿莫西林克拉维酸和头孢曲松在社区获得性肺炎中的选择依据是什么？"
```

## 实现原理

通过 API 调用医学大模型（deepseek-v4-flash / qwen3.8-max 等），使用精心设计的医学系统提示词，输出结构化的中文临床推理。

### 系统提示词

```
你是中国三甲医院副主任医师，拥有20年临床经验，精通内科、外科、妇产科、儿科等各科室。
请根据以下结构回答临床问题：

1. **初步判断**：基于症状/检查结果，给出最可能的诊断方向
2. **鉴别诊断**：列出其他可能的诊断及其排除依据
3. **建议检查**：建议进一步检查以明确诊断
4. **治疗方案**：推荐治疗方案（药物、剂量、疗程）
5. **注意事项**：需要警惕的危险信号和随访建议

注意：
- 回答必须严谨、循证，基于最新临床指南
- 明确标注不确定之处，不夸大诊断信心
- 紧急情况需建议立即就医
- 禁止开具体处方（仅提供参考信息）
- 免责声明：本回答仅供参考，不能替代执业医师面对面诊疗
```

## 配置

在 `.env` 中配置 API Key：

```bash
# 推荐使用 TokenRhythm（国内直连，模型丰富）
TOKENRHYTHM_API_KEY=sk_tr_xxx
TOKENRHYTHM_BASE_URL=https://tokenrhythm.studio/v1

# 或使用 DeepSeek 官方
DEEPSEEK_API_KEY=sk_xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

## 可用模型（推荐优先级）

| 优先级 | 模型 | 适用场景 |
|--------|------|---------|
| 1 | deepseek-v4-flash | 日常快速问答，性价比最高 |
| 2 | qwen3.8-max | 复杂病例分析，最高质量 |
| 3 | kimi-k2.6 | 长文献/病历分析 |
| 4 | deepseek-v4-pro | 需要深度推理 |

## 与训练数据的关系

本 skill 使用的 API 模型与 `medical-data-distillation` 蒸馏的教师模型相同。通过 39,618 条 MedQA 蒸馏数据的验证，这些模型在医学多选题上的准确率约 85-90%，具备可靠的临床推理能力。

## 免责声明

⚠️ **重要**：本工具仅供医学教育和参考，**不能替代执业医师的面对面诊疗**。遇到紧急情况请立即拨打 120 或前往最近的医院就诊。本工具不提供处方，所有用药建议仅供参考。

## 数据来源

- 蒸馏训练数据：39,618 条 MedQA 中文推理 SFT 数据
- 数据集：https://huggingface.co/datasets/rewrewrv343/medqa-distill-sft
- 蒸馏工具链：https://github.com/syi2005s-design/medqa-distill