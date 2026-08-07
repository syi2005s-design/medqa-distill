#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""deep_probe.py —— 深挖 4 个 provider：完整模型名 + 完整响应结构"""
import os
from openai import OpenAI

PROVIDERS = [
    ("deepseek",  "OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("nvidia",    "NVIDIA_API_KEY", "NVIDIA_BASE_URL"),
    ("agnes-cn",  "AGNES_CN_API_KEY", "AGNES_CN_BASE_URL"),
    ("agnes-com", "AGNES_COM_API_KEY", "AGNES_COM_BASE_URL"),
]


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def main():
    load_dotenv()
    for name, key_env, url_env in PROVIDERS:
        key, url = os.getenv(key_env, ""), os.getenv(url_env, "")
        print(f"\n{'=' * 70}\n[{name}] {url}")
        if not key:
            print("  缺少 key"); continue
        try:
            c = OpenAI(api_key=key, base_url=url, timeout=120)
            models = [m.id for m in c.models.list().data]
            print(f"  模型列表 OK，共 {len(models)} 个：")
            for m in models:
                print(f"    - {m}")
            # 用第一个模型做深度 chat 测试
            if models:
                model = models[0]
                try:
                    r = c.chat.completions.create(
                        model=model,
                        messages=[{"role": "user",
                                   "content": "请用一句话回答：1+1=？"}],
                        max_tokens=512, temperature=0)
                    msg = r.choices[0].message
                    print(f"  ✅ chat 测试 {model}:")
                    print(f"     content        = {str(msg.content)[:120]!r}")
                    rc = getattr(msg, "reasoning_content", None)
                    if rc:
                        print(f"     reasoning_content = {str(rc)[:120]!r}")
                    print(f"     finish_reason  = {r.choices[0].finish_reason}")
                    print(f"     usage          = {r.usage}")
                except Exception as e:
                    print(f"  ❌ chat 测试失败: {type(e).__name__}: {str(e)[:200]}")
        except Exception as e:
            print(f"  ❌ 模型列表失败: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
