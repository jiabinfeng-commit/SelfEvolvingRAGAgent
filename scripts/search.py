#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索脚本（阶段 1 验收用）

完整走通这条链路，证明「切片 + 向量存储」真的可用：

    问题 → bge 编码（带查询前缀）→ Milvus 近邻搜索 → 拿 chunk_id → PG 回原文

用法：
    python scripts/search.py "FastAPI 怎么做依赖注入"
    python scripts/search.py "怎么部署" --top-k 5
    python scripts/search.py --eval            # 用评测集跑一遍，看命中情况
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore


def retrieve(question: str, emb, vec: VecStore, pg: PGStore, top_k: int = 5):
    """一次完整检索，返回带原文的结果列表"""
    qv = emb.encode_query(question)
    hits = vec.search(qv, top_k=top_k)
    if not hits:
        return []
    chunk_ids = [cid for cid, _ in hits]
    chunks = pg.get_chunks_by_ids(chunk_ids)
    out = []
    for cid, score in hits:
        c = chunks.get(cid)
        if not c:
            continue
        out.append({"score": score, **c})
    return out


def show(question: str, results, show_content: int = 160):
    print(f"\n{'=' * 70}")
    print(f"Q: {question}")
    print("=" * 70)
    if not results:
        print("  （没召回任何内容）")
        return
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r['score']:.4f}  {r['doc_name']}")
        print(f"      heading: {r['heading']}")
        body = r["content"].replace("\n", " ")
        print(f"      {body[:show_content]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="要检索的问题")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="用评测集批量跑")
    ap.add_argument("--eval-limit", type=int, default=10)
    args = ap.parse_args()

    if not args.question and not args.eval:
        ap.error("给一个问题，或者加 --eval")

    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)

    print("加载模型 ...", end="", flush=True)
    t0 = time.time()
    emb = get_embedder("bge")
    print(f" {time.time()-t0:.1f}s（{emb.dim} 维）")

    pg = PGStore()
    vec = VecStore(dim=emb.dim)
    print(f"PG: {pg.count()}   Milvus: {vec.count()} 条\n")

    if args.question:
        t0 = time.time()
        res = retrieve(args.question, emb, vec, pg, args.top_k)
        show(args.question, res)
        print(f"\n耗时 {time.time()-t0:.2f}s")
        pg.close()
        return

    # ---- 评测集模式 ----
    eval_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "eval", "questions.json")
    if not os.path.exists(eval_path):
        print(f"找不到评测集: {eval_path}")
        pg.close()
        sys.exit(1)

    with open(eval_path, encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"] if isinstance(data, dict) else data
    questions = questions[:args.eval_limit]

    hit1 = hit3 = 0
    scored = 0
    for q in questions:
        text = q["question"]
        gold = q.get("gold_doc")          # 拒答题没有 gold_doc
        res = retrieve(text, emb, vec, pg, top_k=max(args.top_k, 3))
        docs = [r["doc_name"] for r in res]
        if gold:
            scored += 1
            in1 = gold in docs[:1]
            in3 = gold in docs[:3]
            hit1 += int(in1)
            hit3 += int(in3)
            flag = "✓" if in1 else ("○" if in3 else "✗")
        else:
            flag = "·"                     # 拒答题：只看它召回了什么
        top1 = docs[0] if docs else "-"
        print(f"  {flag} {text[:38]:<40} top1={top1[:32]:<34} gold={gold or '(拒答)'}")

    print(f"\nTop-1 命中: {hit1}/{scored}    Top-3 命中: {hit3}/{scored}")
    pg.close()


if __name__ == "__main__":
    main()
