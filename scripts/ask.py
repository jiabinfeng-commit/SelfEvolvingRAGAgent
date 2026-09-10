#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2：最小 RAG 闭环（检索 → 拼 prompt → LLM 生成 → 答案落库）

完整链路（这就是"闭环"二字的由来）：
    用户问题
      → bge 编码（带查询前缀，和阶段 1 检索一致）
      → Milvus 向量检索 top-k 个 chunk_id（+ 相似度分数）
      → PG 按 chunk_id 回表拿原文（双存储的"回表"在这步用上）
      → 用 core/prompt.py 拼成 RAG prompt（上下文 + 问题）
      → LLM 生成答案
      → 答案 + 召回信息写入 PG 的 qa_log（"答案落库"）

相比阶段 1 的 search.py：
    search.py 只到"召回 + 打印"，不调用 LLM、不落库，算是验收切片/向量；
    本脚本把最后一截（生成 + 落库）补上，构成端到端可用的 RAG。

用法：
    python scripts/ask.py "FastAPI 怎么做依赖注入"
    python scripts/ask.py "怎么部署" --top-k 5
    python scripts/ask.py "测试" --dry-run        # 只召回+拼 prompt，不调 LLM、不落库
    python scripts/ask.py "测试" --no-save         # 调 LLM 但不写 qa_log

LLM 后端：读 core/config.py 的 LLM_BACKEND（默认 ollama，本地免费）。
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.prompt import build_prompt, SYSTEM_PROMPT
from core.retrieval import retrieve




def main():
    ap = argparse.ArgumentParser(description="阶段 2：RAG 问答闭环")
    ap.add_argument("question", help="要问的问题")
    ap.add_argument("--top-k", type=int, default=config.RAG_TOP_K)
    ap.add_argument("--dry-run", action="store_true",
                    help="只做召回 + 拼 prompt 并打印，不调用 LLM、也不落库")
    ap.add_argument("--no-save", action="store_true",
                    help="调用 LLM 生成答案，但不写入 qa_log")
    args = ap.parse_args()

    # 1. 配置校验（缺密码/缺语料等早失败）
    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(config.summary())

    # 2. 加载 embedding 模型 + 连接存储
    print("加载 embedding 模型 ...", end="", flush=True)
    t0 = time.time()
    emb = get_embedder("bge")
    print(f" {time.time() - t0:.1f}s（{emb.dim} 维）")

    pg = PGStore()
    vec = VecStore(dim=emb.dim)
    # 确保 qa_log 表存在（CREATE TABLE IF NOT EXISTS，幂等）
    pg.init_schema()
    print(f"PG: {pg.count()}   Milvus: {vec.count()} 条\n")

    # 3. 召回
    t0 = time.time()
    retrieved = retrieve(args.question, emb, vec, pg, args.top_k)
    print(f"召回 {len(retrieved)} 块（耗时 {time.time() - t0:.2f}s）：")
    for i, (score, c) in enumerate(retrieved, 1):
        src = c.get("doc_name", "?")
        heading = (c.get("heading") or "").strip()
        label = f"{src} / {heading}" if heading else src
        print(f"  [{i}] score={score:.4f}  {label}")

    # 4. 拼 prompt（user 部分）
    contexts = [c for _, c in retrieved]
    user_prompt = build_prompt(args.question, contexts, max_chars=config.RAG_CONTEXT_MAX_CHARS)

    # 5. dry-run：到此为止，不调 LLM、不落库
    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY-RUN：下面是会发给 LLM 的 prompt（未实际调用）：")
        print("=" * 70)
        print(f"[system]\n{SYSTEM_PROMPT}\n")
        print(f"[user]\n{user_prompt}")
        pg.close()
        return

    # 6. 调 LLM 生成 默认调用openai
    print("\n生成答案中（后端：%s / %s）..." % (config.LLM_BACKEND, config.LLM_MODEL))
    backend = llm_mod.get_llm("openai")
    t0 = time.time()
    try:
        answer = backend.generate(SYSTEM_PROMPT, user_prompt)
    except RuntimeError as e:
        # 连不上 Ollama / key 错误等：给清晰提示，不崩
        print(f"\n✗ LLM 调用失败：{e}")
        print("   ollama 后端请确认本机 Ollama 已启动且有对应模型；")
        print("   openai 后端请确认 OPENAI_API_KEY 已配置。")
        pg.close()
        sys.exit(1)
    latency = time.time() - t0

    # 7. 输出答案
    print("\n" + "=" * 70)
    print(f"Q: {args.question}")
    print("-" * 70)
    print(answer)
    print(f"\n[生成耗时 {latency:.2f}s]")

    # 8. 答案落库（阶段 2 闭环的最后一步）
    if args.no_save:
        print("（--no-save：未写入 qa_log）")
    else:
        ctx_ids = [c.get("chunk_id") for _, c in retrieved]
        ctx_scores = [float(s) for s, _ in retrieved]
        pg.save_qa(
            question=args.question,
            answer=answer,
            ctx_chunk_ids=ctx_ids,
            ctx_scores=ctx_scores,
            model=config.LLM_MODEL,
            backend=config.LLM_BACKEND,
            latency_s=latency,
        )
        print("（已写入 qa_log ✓）")

    pg.close()


if __name__ == "__main__":
    main()
