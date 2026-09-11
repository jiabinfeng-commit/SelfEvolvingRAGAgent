#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 6：Agent 自愈闭环（命令行入口 / Demo 脚本）

这就是路线图里那个"灵魂 Demo"：
    问一个知识库里没有的问题
      → 看它自己联网去查
      → 自动生成结构化知识条目
      → 过 LLM 质量门禁
      → 写进影子库（staging）
      → 再问同一个问题，它就能答上来了

用法：
    # 单问题：知识库没有，让它自己去查并补库
    python scripts/heal_knowledge.py "FastAPI 的 BackgroundTasks 怎么用"
    python scripts/heal_knowledge.py "某个我库里没有的概念" --retrieval hybrid
    python scripts/heal_knowledge.py "Python dataclass 的 field 默认值是怎么工作的" --max-steps 3

    # 限定调研站点（只爬官方文档，更准也更安全）
    python scripts/heal_knowledge.py "Flask 和 FastAPI 的区别" --site docs.python.org

    # 批量：从文件读，每行一个问题（自动生成 report.md 自愈报告）
    python scripts/heal_knowledge.py --file questions.txt

    # 不想写 qa_log / 影子库（纯演示，不落库）
    python scripts/heal_knowledge.py "测试问题" --no-save

设计要点：
- 重依赖（embedding 模型）只加载一次；pg / llm 也复用，避免每条问题重连重载。
- 每条问题跑完打印：缺口原因 → 调研来源 → 门禁分数 → 入库块数 → 最终答案。
- 批量模式会把逐题结果汇总成一份 Markdown 报告（heal_report.md），方便你
  统计"自愈成功率""脏数据拦截率"这两项简历硬指标。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core import llm as llm_mod
from core.agent import run_self_heal


def _render(res: dict, idx: int = None) -> str:
    """把单条自愈结果渲染成给人看的文本。"""
    head = f"\n{'=' * 70}\n" + (f"问题 #{idx}：" if idx else "问题：") + f"{res['question']}\n{'=' * 70}"
    lines = [head]

    lines.append(f"缺口判定 : {'有缺口' if res.get('gap_detected') else '无缺口'} —— {res.get('gap_reason','')}")

    if res.get("gap_detected"):
        searched = res.get("searched") or []
        lines.append(f"调研来源 : {len(searched)} 个链接")
        for u in searched[:5]:
            lines.append(f"    - {u}")
        q = res.get("quality_detail") or {}
        lines.append(f"质量门禁 : {res.get('quality_score', 0):.2f} "
                     f"{'✅通过' if res.get('quality_pass') else '❌不通过'} —— {q.get('reason','')}")
        lines.append(f"影子库   : 入库 {res.get('ingested_chunks', 0)} 个 chunk"
                     f"（doc_id={res.get('shadow_doc_id','-')}）")

    # 最终答案
    lines.append("-" * 70)
    lines.append(res.get("answer", ""))
    if res.get("refusal"):
        lines.append("（⚠️ 仍判定为拒答：本轮自愈未能补全，未写入影子库）")
    if res.get("self_healed_by_agent"):
        lines.append("（✅ 本轮触发了阶段6 Agent 自愈：联网调研 → 过门禁 → 写影子库 → 再答成功）")

    # 召回标注：来自影子库的标"未验证来源"
    recs = res.get("retrieved") or []
    if recs:
        lines.append(f"\n召回 {len(recs)} 块：")
        for i, c in enumerate(recs, 1):
            label = f"{c.get('doc_name','')} / {c.get('heading','')}" if c.get("heading") else c.get("doc_name", "")
            tag = "  ⚠️[未验证来源]" if c.get("unverified") else ""
            lines.append(f"  [{i}] score={c.get('score',0):.4f}  {label}{tag}")

    # 可观测：步骤轨迹
    for s in res.get("steps") or []:
        lines.append(f"  · {s}")
    return "\n".join(lines)


def _load_questions(path: str) -> list:
    """从文件读问题，每行一条，空行忽略。"""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description="阶段 6：Agent 自愈闭环（联网补库）")
    ap.add_argument("question", nargs="?", help="要问的问题（知识库里没有的）")
    ap.add_argument("--file", help="批量模式：每行一个问题的文本文件")
    ap.add_argument("--top-k", type=int, default=config.RAG_TOP_K, help="召回块数")
    ap.add_argument("--retrieval", choices=["vector", "hybrid"], default="vector",
                    help="vector=纯向量；hybrid=向量+BM25 融合")
    ap.add_argument("--max-steps", type=int, default=3, help="调研重试上限（防死循环）")
    ap.add_argument("--site", default=None, help="限定调研站点域名，如 docs.python.org")
    ap.add_argument("--no-save", action="store_true", help="不写 qa_log / 不写影子库（纯演示）")
    ap.add_argument("--report", default="heal_report.md", help="批量模式报告输出路径")
    args = ap.parse_args()

    # 配置校验：缺密码/缺语料等早失败
    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    if not args.question and not args.file:
        ap.error("必须提供 question 或 --file")
    print(config.summary())

    # 重依赖只建一次（embedding 模型加载慢，pg/llm 复用连接）
    pg = PGStore()
    emb = get_embedder("bge")
    backend_llm = llm_mod.get_llm(config.LLM_BACKEND)

    try:
        if args.file:
            questions = _load_questions(args.file)
            print(f"\n批量自愈：{len(questions)} 个问题\n")
            results = []
            for i, q in enumerate(questions, 1):
                res = run_self_heal(
                    q, top_k=args.top_k, retrieval=args.retrieval,
                    pg=pg, emb=emb, backend_llm=backend_llm, vec=None,
                    max_steps=args.max_steps, site=args.site, save=not args.no_save,
                )
                print(_render(res, i))
                results.append(res)
            _write_report(results, args.report)
        else:
            res = run_self_heal(
                args.question, top_k=args.top_k, retrieval=args.retrieval,
                pg=pg, emb=emb, backend_llm=backend_llm, vec=None,
                max_steps=args.max_steps, site=args.site, save=not args.no_save,
            )
            print(_render(res))
    finally:
        pg.close()


def _write_report(results: list, path: str):
    """批量模式：把结果汇总成 Markdown 报告（自愈成功率 / 拦截率一目了然）。"""
    total = len(results)
    healed = sum(1 for r in results if r.get("self_healed_by_agent"))
    passed = sum(1 for r in results if r.get("quality_pass"))
    rejected = sum(1 for r in results if r.get("refusal"))
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 阶段 6 自愈报告\n\n")
        f.write(f"- 问题总数：{total}\n")
        f.write(f"- Agent 自愈成功（写入影子库并答出）：{healed}（{healed/total:.0%}）\n")
        f.write(f"- 质量门禁通过：{passed}\n")
        f.write(f"- 最终仍拒答：{rejected}\n\n")
        f.write("## 逐题明细\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"### #{i}. {r['question']}\n")
            f.write(f"- 缺口：{'有' if r.get('gap_detected') else '无'} —— {r.get('gap_reason','')}\n")
            f.write(f"- 门禁：{r.get('quality_score', 0):.2f} "
                    f"{'通过' if r.get('quality_pass') else '不通过'}\n")
            f.write(f"- 入库：{r.get('ingested_chunks', 0)} chunk\n")
            f.write(f"- 自愈：{'✅' if r.get('self_healed_by_agent') else '❌'}\n\n")
    print(f"\n报告已写出：{path}")


if __name__ == "__main__":
    main()
