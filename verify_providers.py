#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_providers.py —— 验证 .env 中所有 API provider 的可用性
对每个 provider：1) 拉取模型列表  2) 挑候选模型  3) 发一个最小 chat 请求测试
用法: python verify_providers.py
"""
import os
from openai import OpenAI

PROVIDERS = [
    ("deepseek",  "OPENAI_API_KEY", "OPENAI_BASE_URL"),
    ("nvidia",    "NVIDIA_API_KEY", "NVIDIA_BASE_URL"),
    ("agnes-cn",  "AGNES_CN_API_KEY", "AGNES_CN_BASE_URL"),
    ("agnes-com", "AGNES_COM_API_KEY", "AGNES_COM_BASE_URL"),
]

# 医学/通用能力较强的模型关键词，用于从模型列表里挑候选
PREFERRED = ("deepseek", "qwen", "llama", "glm", "medical", "med", "nemotron",
             "mistral", "phi", "gpt", "kimi", "minimax", "moonshot", "yi", "ernie")
BLOCKED = ("embed", "whisper", "tts", "asr", "stable", "flux", "sdxl", "image",
           "video", "rerank", "guard", "moderation", "stt", "audio", "nano-llama")


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def test_provider(name, key, url):
    print(f"\n{'=' * 60}\n[1/2] {name}  模型列表拉取中...")
    if not key or not url:
        print("  ❌ 缺少 key 或 base_url")
        return []
    try:
        c = OpenAI(api_key=key, base_url=url, timeout=60)
        models = [m.id for m in c.models.list().data]
        print(f"  ✅ 模型列表 OK：共 {len(models)} 个")
        cands = [m for m in models
                 if any(k in m.lower() for k in PREFERRED)
                 and not any(b in m.lower() for b in BLOCKED)]
        print(f"  📋 候选模型（{len(cands)} 个）: {cands[:12]}")
        return [(name, key, url, m) for m in cands[:6]]
    except Exception as e:
        print(f"  ❌ 模型列表失败: {type(e).__name__}: {str(e)[:200]}")
        return []


def ping(c, model):
    r = c.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "只回复两个字：正常"}],
        max_tokens=32, temperature=0)
    return r.choices[0].message.content.strip()


def main():
    load_dotenv()
    pool = []
    for name, key_env, url_env in PROVIDERS:
        pool += test_provider(name, os.getenv(key_env, ""), os.getenv(url_env, ""))

    print(f"\n{'=' * 60}\n[2/2] 逐个发最小 chat 测试（每个候选模型）...")
    for name, key, url, model in pool:
        try:
            c = OpenAI(api_key=key, base_url=url, timeout=90)
            out = ping(c, model)
            print(f"  ✅ {name:10s} {model:45s} -> {out!r}")
        except Exception as e:
            print(f"  ❌ {name:10s} {model:45s} -> {type(e).__name__}: {str(e)[:150]}")

    print(f"\n共 {len(pool)} 个候选（provider×模型），上面 ✅ 的即可用于蒸馏。")


if __name__ == "__main__":
    main()
