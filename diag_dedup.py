#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dedup 性能诊断"""
import json, re, time

def ngrams(s, n=3):
    s = re.sub(r"\s+", "", s or "")
    return set(s[i:i+n] for i in range(len(s)-n+1))

t0 = time.time()
lines = open('output/us4-full/cot.jsonl', encoding='utf-8').readlines()
print(f"读取 {len(lines)} 行: {time.time()-t0:.2f}s")

t0 = time.time()
ngs = []
for l in lines[:2000]:
    d = json.loads(l)
    ngs.append(ngrams(d.get('instruction','')))
print(f"2000 条 ngram: {time.time()-t0:.2f}s")

t0 = time.time()
buckets = {}
dup = 0
for g in ngs:
    key = 'a'
    hit = False
    for cg in buckets.get(key, []):
        inter = len(g.intersection(cg))
        union = len(g.union(cg))
        if union and inter / union > 0.85:
            dup += 1
            hit = True
            break
    if not hit:
        buckets.setdefault(key, []).append(g)
print(f"2000 条桶内比较: {time.time()-t0:.2f}s, dup={dup}, 桶大小={len(buckets.get(key, []))}")

# 全量预估
t0 = time.time()
all_ngs = []
for l in lines:
    d = json.loads(l)
    all_ngs.append(ngrams(d.get('instruction','')))
print(f"全量 {len(all_ngs)} 条 ngram: {time.time()-t0:.2f}s")
