#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2：最小 RAG 闭环（命令行入口）

完整链路（这就是"闭环"二字的由来）：
    用户问题
      → bge 编码（带查询前缀）→ Milvus 向量检索 → PG 回表拿原文
      → 拼 RAG prompt → LLM 生成答案 → 答案 + 召回信息写入 PG 的 qa_log

为什么这个文件这么短？
- 之前问答主逻辑写死在 main() 里，后来抽到了 core/rag.py 的 generate_answer()，
  三个入口（CLI / HTTP 服务 / streamlit 页面）共用同一份逻辑，避免口径漂移。
  本文件现在只负责「解析命令行参数 + 把结果打印成人能看的格式」。

用法：
    python scripts/ask.py "FastAPI 怎么做依赖注入"
    python scripts/ask.py "怎么部署" --top-k 5
    python scripts/ask.py "测试" --retrieval hybrid          # 阶段4：混合检索
    python scripts/ask.py "测试" --self-heal                  # 阶段5：自愈
    python scripts/ask.py "测试" --agent-heal                # 阶段6：缺口自愈（联网补库）
    python scripts/ask.py "测试" --dry-run                   # 只召回+拼 prompt，不调 LLM、不落库
    python scripts/ask.py "测试" --no-save                   # 调 LLM 但不写 qa_log

LLM 后端：读 core/config.py 的 LLM_BACKEND（默认 openai，走阿里云 DashScope）。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.rag import generate_answer
from core.agent import run_self_heal


def main():
    ap = argparse.ArgumentParser(description="阶段 2：RAG 问答闭环（命令行）")
    ap.add_argument("question", help="要问的问题")
    ap.add_argument("--top-k", type=int, default=config.RAG_TOP_K,
                    help="召回多少块拼进 prompt（默认读 config.RAG_TOP_K）")
    ap.add_argument("--retrieval", choices=["vector", "hybrid"], default="vector",
                    help="阶段4：vector=纯向量语义检索；hybrid=向量+BM25 关键词融合")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做召回 + 拼 prompt 并打印，不调用 LLM、也不落库")
    ap.add_argument("--no-save", action="store_true",
                    help="调用 LLM 生成答案，但不写入 qa_log")
    ap.add_argument("--self-heal", action="store_true",
                    help="阶段5：拒答且召回分够高时，换宽松指令 + 扩大召回自动重试一次")
    ap.add_argument("--agent-heal", action="store_true",
                    help="阶段6：知识缺口自愈——库内答不上时联网调研→生成→门禁→写影子库→再答")
    ap.add_argument("--reflect", action="store_true",
                    help="阶段8：生成答案后做事实核查，发现无依据断言则重检索/重写，仍不过则降级拒答")
    ap.add_argument("--trace", action="store_true",
                    help="阶段8：把本次问答链路（召回/分数/prompt/耗时/token/反射）写入 trace_log")
    args = ap.parse_args()

    # 1. 配置校验（缺密码/缺语料等早失败，和 serve.py / ui.py 一致）
    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(config.summary())

    # 阶段6：Agent 自愈（联网补库）。需要常驻依赖，单独建实例并注入。
    if args.agent_heal:
        from core.embedder import get_embedder
        from core.storage.pg_store import PGStore
        from core import llm as llm_mod
        pg = PGStore()
        emb = get_embedder("bge")
        llm = llm_mod.get_llm(config.LLM_BACKEND)
        try:
            res = run_self_heal(
                args.question, top_k=args.top_k, retrieval=args.retrieval,
                pg=pg, emb=emb, backend_llm=llm, vec=None,
                save=not args.no_save,
            )
        finally:
            pg.close()
        _render_agent(res)
        return

    # 2. 调共享核心逻辑（所有入口都走这一份）
    res = generate_answer(
        args.question,
        top_k=args.top_k,
        retrieval=args.retrieval,
        self_heal=args.self_heal,
        save=not args.no_save,
        dry_run=args.dry_run,
        reflect=args.reflect,
        trace=args.trace,
    )

    # 3. 渲染结果
    if args.dry_run:
        # dry-run：打印召回 + 拼好的 prompt，不调 LLM
        print(f"\n召回 {len(res['retrieved'])} 块：")
        for i, c in enumerate(res["retrieved"], 1):
            label = f"{c['doc_name']} / {c['heading']}" if c["heading"] else c["doc_name"]
            print(f"  [{i}] score={c['score']:.4f}  {label}")
        print("\n" + "=" * 70)
        print("DRY-RUN：下面是会发给 LLM 的 prompt（未实际调用）：")
        print("=" * 70)
        print(f"[system]\n{res['system_prompt']}\n")
        print(f"[user]\n{res['user_prompt']}")
        return

    # 正常问答：打印召回概览 + 答案 + 状态
    print(f"\n召回 {len(res['retrieved'])} 块：")
    for i, c in enumerate(res["retrieved"], 1):
        label = f"{c['doc_name']} / {c['heading']}" if c["heading"] else c["doc_name"]
        print(f"  [{i}] score={c['score']:.4f}  {label}")

    print("\n" + "=" * 70)
    print(f"Q: {res['question']}")
    print("-" * 70)
    print(res["answer"])
    print(f"\n[生成耗时 {res['latency_s']}s]")

    if res.get("self_healed"):
        print("（✅ 本次触发了阶段5 Agent 自愈并成功救回）")
    elif res.get("refusal"):
        print("（⚠️ 模型判定为拒答，未触发自愈或被安全阀拦下）")

    # 阶段 8 Reflexion 信息
    if res.get("reflected"):
        rp = res.get("reflect_pass")
        tag = "通过✅" if rp else "未通过→已降级拒答⚠️"
        print(f"（🔎 阶段8 Reflexion 已触发，事实核查{tag}）")

    if args.no_save:
        print("（--no-save：未写入 qa_log）")
    else:
        print("（已写入 qa_log ✓）")


def _render_agent(res: dict):
    """渲染阶段6 自愈结果（复用 heal_knowledge 的展示口径）。"""
    print(f"\n{'=' * 70}")
    print(f"Q: {res['question']}")
    print(f"{'=' * 70}")
    print(f"缺口判定 : {'有缺口' if res.get('gap_detected') else '无缺口'} —— {res.get('gap_reason','')}")
    if res.get("gap_detected"):
        searched = res.get("searched") or []
        print(f"调研来源 : {len(searched)} 个链接")
        for u in searched[:5]:
            print(f"    - {u}")
        q = res.get("quality_detail") or {}
        print(f"质量门禁 : {res.get('quality_score', 0):.2f} "
              f"{'通过' if res.get('quality_pass') else '不通过'} —— {q.get('reason','')}")
        print(f"影子库   : 入库 {res.get('ingested_chunks', 0)} 个 chunk"
              f"（doc_id={res.get('shadow_doc_id','-')}）")
    print("-" * 70)
    print(res.get("answer", ""))
    if res.get("refusal"):
        print("（⚠️ 仍判定为拒答：本轮自愈未能补全，未写入影子库）")
    if res.get("self_healed_by_agent"):
        print("（✅ 阶段6 Agent 自愈成功：联网调研 → 过门禁 → 写影子库 → 再答）")
    # 召回标注未验证来源
    for i, c in enumerate(res.get("retrieved") or [], 1):
        tag = "  ⚠️[未验证来源]" if c.get("unverified") else ""
        label = f"{c.get('doc_name','')} / {c.get('heading','')}" if c.get("heading") else c.get("doc_name", "")
        print(f"  [{i}] score={c.get('score',0):.4f}  {label}{tag}")
    for s in res.get("steps") or []:
        print(f"  · {s}")


if __name__ == "__main__":
    main()
