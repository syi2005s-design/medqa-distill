#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api_distill.py —— MedQA 医学数据 API 蒸馏工具
功能：CoT 推理蒸馏 / Self-Instruct 新题生成 / k=3 自洽性过滤 / n-gram 去重 / token 统计 / 断点续传

用法：
  python api_distill.py --task cot            --input data/medqa_train.jsonl --k 3
  python api_distill.py --task self-instruct  --input data/medqa_train.jsonl --k 1
  python api_distill.py --dedup  --input output/cot.jsonl --out output/cot_dedup.jsonl
  python api_distill.py --stats  --input output/cot.jsonl

输入格式（JSONL，一行一个对象）：
  {"id": "...", "question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "answer": "D"}

环境变量（.env 或 export）：
  OPENAI_API_KEY=sk-xxx
  OPENAI_BASE_URL=https://api.deepseek.com/v1
  DISTILL_MODEL=deepseek-chat
"""
import argparse
import asyncio
import json
import os
import random
import re
import ssl
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import tiktoken
from openai import AsyncOpenAI

# ---------------- 配置（.env 自动加载，CLI 参数可覆盖） ----------------
def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("DISTILL_MODEL", "deepseek-chat")
API_KEY = os.getenv("OPENAI_API_KEY", "")
CONCURRENCY = int(os.getenv("DISTILL_CONCURRENCY", "4"))  # 并发数：429/限流就降到 2~4；镜像进程叠加时也要控制
TEMPERATURE = 0.7
MAX_TOKENS = 2048  # 提示词已限制思考长度；2048 足够（4096 会让推理模型思考更久）
OUT_DIR = Path(os.getenv("DISTILL_OUT_DIR", "output"))
OUT_DIR.mkdir(exist_ok=True)

_client = None
sem = asyncio.Semaphore(CONCURRENCY)


def build_http_client():
    """部分网络会重置 TLS1.3 握手（如 NVIDIA 端点），设 DISTILL_TLS12=1 强制 TLS1.2。
    timeout=120：防止连接挂起导致假死（heartbeat 检查点在循环里，卡在 await 时不会触发）。"""
    if os.getenv("DISTILL_TLS12", "").lower() in ("1", "true", "yes"):
        ctx = ssl.create_default_context()
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        return httpx.AsyncClient(verify=ctx, timeout=120)
    return httpx.AsyncClient(timeout=120)


def get_client():
    """惰性初始化，让 CLI 参数（--api-key/--base-url/--model）生效。"""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL,
                              http_client=build_http_client(), timeout=120)
    return _client

# ---------------- 提示词模板 ----------------
COT_SYSTEM = (
    "你是一名严谨的医学专家。请始终使用中文回答（无论题目是什么语言）。"
    "回答医学问题时：1) 先给出逐步推理（病理机制、"
    "鉴别诊断、排除理由）；2) 必须输出 JSON：{\"reasoning\": \"...\", \"answer\": \"D\"}，"
    "answer 为选项字母，与推理结论一致。只输出 JSON，不要多余内容。"
)
COT_USER = "题目：{question}\n选项：\n{options}"

SI_SYSTEM = (
    "你是医学题库命题专家。请始终使用中文回答（无论原题是什么语言）。"
    "根据原题生成一道考点相同、表述不同的新题"
    "（难度相近或略高），只输出新题目和 A-D 四个选项，不要答案。"
    "必须输出 JSON：{\"question\": \"...\", \"A\": \"...\", \"B\": \"...\", \"C\": \"...\", \"D\": \"...\"}"
)
SI_USER = "原题：{question}\n选项：\n{options}"


def options_block(item):
    return "\n".join(f"{k}. {item[k]}" for k in "ABCDE" if item.get(k))


def build_messages(task, item):
    opts = options_block(item)
    if task == "cot":
        return [{"role": "system", "content": COT_SYSTEM},
                {"role": "user", "content": COT_USER.format(question=item["question"], options=opts)}]
    if task == "self-instruct":
        return [{"role": "system", "content": SI_SYSTEM},
                {"role": "user", "content": SI_USER.format(question=item["question"], options=opts)}]
    raise ValueError(task)


# ---------------- API 调用（并发 + 指数退避重试） ----------------
LAST_SUCCESS = time.time()   # 心跳：最后一次 API 成功时间
HEARTBEAT_TIMEOUT = 300      # 连续 5 分钟无成功请求 → 自杀退出（网络假死保护）
LOCK_DIR = None              # 单实例锁目录（out-dir 下 .lockdir，mkdir 原子获取）


def acquire_instance_lock(out_dir):
    """单实例锁（mkdir 原子操作）：同一 out-dir 只允许一个进程跑。
    锁目录内 hb 文件的 mtime 由 API 成功心跳持续刷新；死进程的锁 10 分钟无刷新可被抢占。
    注意：mkdir 与写 hb 之间存在竞态窗口（镜像进程几乎同时启动时会双双通过）。
    解决办法：mkdir 成功后立即写 hb，若写失败说明锁已被抢占，立即退出让路。"""
    import shutil
    global LOCK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(out_dir) / ".lockdir"
    hb = lock_dir / "hb"
    try:
        lock_dir.mkdir()          # 原子操作：同一瞬间只有一个进程能成功
    except FileExistsError:
        if hb.exists() and time.time() - hb.stat().st_mtime < 600:
            print(f"[lock] {out_dir} 已有活跃实例，本进程退出", file=sys.stderr)
            sys.exit(0)
        # 锁已过期（死进程残留）：抢占
        shutil.rmtree(lock_dir, ignore_errors=True)
        lock_dir.mkdir()
    try:
        hb.write_text(str(os.getpid()))
    except OSError:
        # mkdir 成功但 hb 写失败 = 锁刚被另一镜像进程抢占，让路
        print(f"[lock] {out_dir} 锁被抢占，本进程退出", file=sys.stderr)
        sys.exit(0)
    LOCK_DIR = lock_dir


def touch_lock():
    """API 成功时刷新锁心跳（活体证明）。锁丢失（被镜像进程抢占）则自杀让路。"""
    if LOCK_DIR is not None:
        try:
            (LOCK_DIR / "hb").touch()
        except OSError:
            print("[lock] 锁丢失，疑似镜像进程抢占，退出(code=4)", file=sys.stderr)
            sys.exit(4)


def check_heartbeat():
    """网络假死保护：长时间无成功请求说明连接池已坏，自杀让看护脚本重启。"""
    if time.time() - LAST_SUCCESS > HEARTBEAT_TIMEOUT:
        print(f"[heartbeat] {HEARTBEAT_TIMEOUT}s 无成功请求，疑似假死，退出(code=3)", file=sys.stderr)
        sys.exit(3)


async def call_llm(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, retries=6):
    global LAST_SUCCESS
    for attempt in range(retries):
        try:
            async with sem:
                # 硬超时 90s：长题目推理可能久，但绝不允许无限挂起占死并发槽位
                r = await asyncio.wait_for(
                    get_client().chat.completions.create(
                        model=MODEL, messages=messages,
                        temperature=temperature, max_tokens=max_tokens),
                    timeout=90)
            LAST_SUCCESS = time.time()
            touch_lock()
            msg = r.choices[0].message
            content = msg.content or ""
            # 推理模型（deepseek-v4-flash 等）可能把 token 全花在思考上，
            # content 为空但 reasoning_content 里有完整推理+答案 JSON —— 兜底提取
            if not content.strip():
                rc = getattr(msg, "reasoning_content", None) or ""
                if rc.strip():
                    # reasoning 里通常含最终 JSON 或 答案：X
                    return rc.strip()
                # content 与 reasoning 全空 = API 异常空响应（限流/截断），
                # 抛异常触发重试，避免被上层误判为"格式错"永久 skip
                raise RuntimeError("空响应(content与reasoning均为空)")
            return content
        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"[retry {attempt + 1}] {type(e).__name__}: {e} -> 等待 {wait:.1f}s", file=sys.stderr)
            await asyncio.sleep(wait)
    return None


# ---------------- 解析 ----------------
def strip_fence(s):
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    return m.group(1).strip() if m else s


def parse_json(s):
    """先剥围栏 json.loads，失败返回 None。"""
    try:
        return json.loads(strip_fence(s))
    except Exception:
        return None


def extract_answer_cot(out):
    """从 CoT 输出提取答案字母。支持：整体 JSON / 正文+尾部```json块 / "answer":"X" / 答案：X、答案是 X。"""
    s = out or ""
    # 1) 整体 JSON
    d = parse_json(s)
    if isinstance(d, dict) and re.fullmatch(r"[A-E]", str(d.get("answer", "")).strip().upper()):
        return str(d["answer"]).strip().upper(), d
    # 2) 正文后跟 ```json 围栏块（取最后一个）
    for b in reversed(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)):
        d = parse_json(b)
        if isinstance(d, dict) and re.fullmatch(r"[A-E]", str(d.get("answer", "")).strip().upper()):
            return str(d["answer"]).strip().upper(), d
    # 3) 文本中任意位置 "answer": "X"（取最后一个）
    for m in reversed(list(re.finditer(r'"answer"\s*:\s*"([A-Ea-e])"', s))):
        return m.group(1).upper(), None
    # 4) 中文表述：答案：X / 答案是 X / 答案为 X
    m = re.search(r"答案(?:是|为|[:：])\s*([A-Ea-e])", s)
    if m:
        return m.group(1).upper(), None
    return None, None


# ---------------- 任务执行 ----------------
async def generate_one(task, item, k=1):
    """k=1: 单次生成；k>1: 自洽性多数投票（不一致返回 None）。"""
    outputs, letters = [], []
    for _ in range(k):
        out = await call_llm(build_messages(task, item))
        if not out:
            continue
        if task == "cot":
            letter, d = extract_answer_cot(out)
            if letter is None:
                continue
            # 与标准答案不一致的样本直接丢弃（矛盾数据比没有更糟）
            if letter != item.get("answer", "").strip().upper():
                continue
            letters.append(letter)
            if d is None:
                # 非整体 JSON：剥掉尾部围栏 JSON 块，只保留正文推理
                cleaned = re.sub(r"```(?:json)?\s*\{.*?\}\s*```\s*$", "", out or "", flags=re.S).strip()
                d = {"reasoning": cleaned or out, "answer": letter}
            outputs.append(d)
        else:  # self-instruct：要求输出 JSON 新题
            d = parse_json(out)
            if isinstance(d, dict) and d.get("question") and all(d.get(x) for x in "ABCD"):
                letters.append("ok")
                outputs.append(d)

    if not outputs:
        return None
    if k == 1:
        return outputs[0]
    cnt = Counter(letters)
    best, n = cnt.most_common(1)[0]
    if n < max(2, (k + 1) // 2):        # 意见分裂 -> 丢弃
        return None
    for letter, out in zip(letters, outputs):
        if letter == best:
            return out
    return None


def to_alpaca(task, item, result):
    """转 Alpaca 格式。"""
    if task == "cot":
        reasoning = result.get("reasoning", "")
        ans = result.get("answer", "")
        return {
            "instruction": f"题目：{item['question']}\n选项：\n{options_block(item)}",
            "input": "",
            "output": f"{reasoning}\n答案：{ans}",
        }
    return {
        "instruction": f"题目：{result['question']}\n选项：\n{options_block(result)}",
        "input": "",
        "output": f"（由种子题 {item['id']} 生成，答案待二次生成）",
    }


# ---------------- 主流程（断点续传） ----------------
async def run_task(task, input_path, k, limit):
    items = [json.loads(l) for l in open(input_path, encoding="utf-8")]
    if limit:
        items = items[:limit]
    done_file = OUT_DIR / f"{task}.done"
    done_ids = set(done_file.read_text().splitlines()) if done_file.exists() else set()
    out_path = OUT_DIR / f"{task}.jsonl"
    out = open(out_path, "a", encoding="utf-8")
    done = open(done_file, "a", encoding="utf-8")
    ok = skip = 0

    # 待处理列表（跳过已完成的 id）
    pending = [(str(item.get("id", i)), item) for i, item in enumerate(items)
               if str(item.get("id", i)) not in done_ids]

    async def worker(iid, item):
        result = await generate_one(task, item, k=k)
        return iid, result

    def flush_done(iid, result):
        nonlocal ok, skip
        check_heartbeat()   # 每条完成即检查假死（gather 卡住时也能触发退出让看护重启）
        if result is None:
            skip += 1
            print(f"[skip {ok + skip}/{len(items)}] {iid}（不一致或格式错）")
        else:
            out.write(json.dumps(to_alpaca(task, item_by_id[iid], result), ensure_ascii=False) + "\n")
            ok += 1
            if (ok + skip) % 20 == 0:
                print(f"[progress {ok + skip}/{len(items)}] ok={ok} skip={skip}")
        done.write(iid + "\n")
        done.flush()

    item_by_id = {iid: it for iid, it in pending}
    try:
        # 批量并发：gather 一批（并发由 call_llm 内 Semaphore 控制），结果按序落盘
        # BATCH 不宜过大：批次内最慢请求拖累整批落盘；16 条一批，慢请求只影响 16 条
        BATCH = 16
        for start in range(0, len(pending), BATCH):
            check_heartbeat()   # 假死保护：一批开始前检查
            chunk = pending[start:start + BATCH]
            results = await asyncio.gather(*[worker(iid, it) for iid, it in chunk])
            for iid, result in results:
                flush_done(iid, result)
        out.flush()
    finally:
        out.close()
        done.close()
    print(f"[done] {task}: 成功 {ok} 条, 丢弃 {skip} 条 -> {out_path}")


# ---------------- 去重 ----------------
def ngrams(s, n=3):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i + n] for i in range(len(s) - n + 1))


def jaccard(a, b):
    if not a or not b:
        return 0
    return len(a & b) / len(a | b)


def dedup(input_path, out_path, threshold=0.85):
    """按首字符分桶 + 长度粗筛 + 优化 jaccard，避免 O(n²) 全量比较。
    注意：CoT 蒸馏数据（一题一条）天然唯一，无需 dedup；此函数用于 Self-Instruct 新题。"""
    kept = []
    buckets = {}
    total = dup = 0
    for line in open(input_path, encoding="utf-8"):
        d = json.loads(line)
        total += 1
        q = d.get("instruction", "")
        g = ngrams(q)
        key = q[:1].lower() if q else "?"
        dup_found = False
        for cand_g, cand_len in buckets.get(key, []):
            # 长度粗筛：长度差 >30% 直接跳过（大提速）
            if abs(len(g) - cand_len) > 0.3 * max(len(g), cand_len):
                continue
            inter = len(g.intersection(cand_g))
            j = inter / (len(g) + cand_len - inter) if inter else 0.0
            if j > threshold:
                dup += 1
                dup_found = True
                break
        if not dup_found:
            buckets.setdefault(key, []).append((g, len(g)))
            kept.append(d)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[dedup] 输入 {total} 条, 去重后 {len(kept)} 条, 去掉 {dup} 条 -> {out_path}")


# ---------------- 统计 ----------------
def stats(input_path):
    enc = tiktoken.get_encoding("cl100k_base")
    n = total = 0
    lens = []
    for line in open(input_path, encoding="utf-8"):
        d = json.loads(line)
        n += 1
        t = len(enc.encode(d.get("instruction", "") + "\n" + d.get("output", "")))
        total += t
        lens.append(t)
    print(f"[stats] {n} 条, 共 {total} token, 平均 {total // max(n, 1)}/条, "
          f"最长 {max(lens) if lens else 0}")
    print(f"        估算输出成本(以 deepseek-chat 输出价 $1.1/M 计): ${total * 1.1 / 1e6:.2f}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="医学数据 API 蒸馏工具")
    ap.add_argument("--task", choices=["cot", "self-instruct"])
    ap.add_argument("--input", default="data/medqa_train.jsonl")
    ap.add_argument("--out", default=None, help="dedup/stats 的输出路径")
    ap.add_argument("--k", type=int, default=3, help="自洽性采样次数(1=不校验)")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 条(先试跑)")
    ap.add_argument("--dedup", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--api-key", help="覆盖 API key（默认读 .env）")
    ap.add_argument("--base-url", help="覆盖 base_url（默认读 .env）")
    ap.add_argument("--model", help="覆盖模型名（默认读 .env）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 output/）")
    ap.add_argument("--max-tokens", type=int, default=None, help="输出上限（默认 2048；深思考模型可调大）")
    args = ap.parse_args()

    global API_KEY, BASE_URL, MODEL, OUT_DIR, _client, MAX_TOKENS
    if args.max_tokens:
        MAX_TOKENS = args.max_tokens
    if args.api_key:
        API_KEY = args.api_key
    if args.base_url:
        BASE_URL = args.base_url
    if args.model:
        MODEL = args.model
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    _client = None  # 强制用新配置重建客户端

    # 单实例锁：生成任务才加锁（dedup/stats 只读不锁）
    if args.task and not args.dedup and not args.stats:
        acquire_instance_lock(OUT_DIR)

    if not API_KEY and not args.stats and not args.dedup:
        print("请先设置 OPENAI_API_KEY（.env 或环境变量）", file=sys.stderr)
        sys.exit(1)

    if args.dedup:
        dedup(args.input, args.out or args.input.replace(".jsonl", "_dedup.jsonl"))
    elif args.stats:
        stats(args.input)
    else:
        asyncio.run(run_task(args.task, args.input, k=args.k, limit=args.limit))


if __name__ == "__main__":
    main()
