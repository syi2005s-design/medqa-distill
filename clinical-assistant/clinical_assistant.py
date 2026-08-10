#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clinical_assistant.py —— 临床医学助手 CLI

用法：
  python clinical_assistant.py "患者发烧39度，咳嗽3天，有黄痰"
  python clinical_assistant.py --case "WBC 15×10^9/L, 中性粒细胞90%"
  python clinical_assistant.py --interactive
"""
import argparse
import json
import os
import sys
import urllib.request

# 默认配置
DEFAULT_API_KEY = os.getenv("TOKENRHYTHM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
DEFAULT_BASE_URL = os.getenv("TOKENRHYTHM_BASE_URL", "https://tokenrhythm.studio/v1")
DEFAULT_MODEL = os.getenv("CLINICAL_MODEL", "deepseek-v4-flash")

SYSTEM_PROMPT = """你是中国三甲医院副主任医师，拥有20年临床经验，精通内科、外科、妇产科、儿科等各科室。
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
- 回答使用中文，医学术语可保留英文
- 免责声明：本回答仅供参考，不能替代执业医师面对面诊疗"""


def call_llm(messages, model=DEFAULT_MODEL, max_tokens=4096, temperature=0.3):
    """调用 LLM API 获取临床回答。"""
    if not DEFAULT_API_KEY:
        print("错误: 请设置 TOKENRHYTHM_API_KEY 或 OPENAI_API_KEY 环境变量", file=sys.stderr)
        sys.exit(1)

    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        f"{DEFAULT_BASE_URL}/chat/completions",
        body,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEFAULT_API_KEY}",
        },
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        msg = resp["choices"][0]["message"]
        content = msg.get("content", "")
        # 推理模型可能把内容放在 reasoning_content
        if not content.strip():
            content = getattr(msg, "reasoning_content", "") or ""
        return content
    except Exception as e:
        return f"API 调用失败: {e}"


def main():
    ap = argparse.ArgumentParser(description="临床医学助手 CLI")
    ap.add_argument("query", nargs="?", help="临床问题（如：发烧39度咳嗽3天）")
    ap.add_argument("--case", "-c", help="病历摘要/检查结果")
    ap.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    ap.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"模型（默认: {DEFAULT_MODEL}）")
    args = ap.parse_args()

    if args.interactive:
        print("🩺 临床医学助手（输入 exit 退出）\n")
        while True:
            q = input("患者症状/问题: ").strip()
            if not q or q.lower() in ("exit", "quit", "退出"):
                break
            print("\n⏳ 正在分析...\n")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            result = call_llm(messages, model=args.model)
            print(result)
            print("\n" + "=" * 60 + "\n")
    elif args.query:
        query = args.query
        if args.case:
            query = f"患者情况：{args.query}\n\n检查结果：{args.case}"
        print(f"\n🩺 临床分析（{args.model}）\n")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]
        result = call_llm(messages, model=args.model)
        print(result)
        print("\n" + "=" * 60)
        print("⚠️ 免责声明：本回答仅供参考，不能替代执业医师面对面诊疗")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()