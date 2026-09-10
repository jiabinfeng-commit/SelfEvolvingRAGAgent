#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 3：端到端评估闭环（自进化系统的「度量」）

完整链路（这就是阶段 3 的全部）：
    评测集每题 → 召回(复用阶段2口径) → 拼 prompt(复用阶段2) → LLM 生成答案(复用阶段2)
              → LLM 当裁判打分(0/1/2) → 聚合指标 → 写 eval_run/eval_result 表 → 出报告

和阶段 2 的关系：
    生成答案这一步与 ask.py 用的是**同一套** retrieve + build_prompt + llm.generate，
    阶段 3 只多出「裁判打分 + 评分聚合 + 落 eval 表」。所以评估测的就是线上真实链路。

为什么这一步是「自进化」的地基？
    你以后改检索(rerank/混合)、改 prompt、换模型，到底变好还是变坏？必须有稳定指标。
    eval_run 把每次跑分存下来，下次优化直接比 accuracy / recall_top3 的数字。

用法：
    python scripts/evaluate.py --limit 5            # 先跑 5 题自测（快、省 token）
    python scripts/evaluate.py                       # 全量 40 题评估
    python scripts/evaluate.py --no-save             # 跑但不写 eval 表（只打印+报告）
    python scripts/evaluate.py --output my_report.md # 报告写到指定路径

LLM 后端：读 core/config.py 的 LLM_BACKEND（现在已是阿里云 openai 兼容）。
"""
import os
import sys
import time
import argparse
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.evaluator import run_evaluation, JUDGE_SYSTEM_PROMPT


def load_questions(limit: int = None) -> list:
    """加载评测集 eval/questions.json 的 questions 列表。"""
    path = os.path.join(ROOT, "eval", "questions.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    qs = data["questions"]
    return qs[:limit] if limit else qs


def render_md(summary: dict, results: list, run_id, model: str, backend: str) -> str:
    """把评估结果渲染成 Markdown 报告（也方便以后直接发到 wiki / 存档）。"""
    lines = []
    lines.append(f"# 阶段 3 评估报告（run_id={run_id}）\n")
    lines.append(f"- 模型：`{model}`  后端：`{backend}`")
    lines.append(f"- 题数：`{summary['n']}`  平均耗时：`{summary['avg_latency_s']}s/题`\n")
    lines.append("## 汇总\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 正确率(裁判=2) | {summary['accuracy']*100:.1f}% ({summary['correct']}/{summary['n']}) |")
    lines.append(f"| 部分正确率 | {summary['partial_rate']*100:.1f}% |")
    lines.append(f"| 错误率 | {summary['wrong']}/{summary['n']} |")
    if summary["refusal_rate"] is not None:
        lines.append(f"| 拒答正确率 | {summary['refusal_rate']*100:.1f}% |")
    lines.append(f"| 召回 top1 | {summary['recall_top1']*100:.1f}% |")
    lines.append(f"| 召回 top3 | {summary['recall_top3']*100:.1f}% |")
    lines.append("")
    lines.append("## 逐题明细\n")
    lines.append("| # | 题型 | 裁判 | recall@3 | 理由 |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        reason = (r["judge_reason"] or "").replace("|", "/").replace("\n", " ")[:50]
        q = (r["question"] or "").replace("|", "/")[:24]
        lines.append(f"| {r['qid']} | {r['qtype']} | {r['judge_label']} | "
                     f"{'✓' if r['recall_top3'] else '✗'} | {reason} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="阶段 3：RAG 端到端评估")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题（自测用，省 token）")
    ap.add_argument("--top-k", type=int, default=config.RAG_TOP_K)
    ap.add_argument("--no-save", action="store_true", help="跑但不写 eval_run/eval_result 表")
    ap.add_argument("--output", default=None, help="Markdown 报告输出路径")
    args = ap.parse_args()

    # 1. 配置校验（缺密码/缺语料等早失败）
    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(config.summary())

    # 2. 加载模型 + 连接存储（init_schema 含阶段3新增的 eval_run/eval_result 表）
    print("加载 embedding 模型 ...", end="", flush=True)
    t0 = time.time()
    emb = get_embedder("bge")
    print(f" {time.time()-t0:.1f}s（{emb.dim} 维）")
    pg = PGStore()
    vec = VecStore(dim=emb.dim)
    pg.init_schema()
    print(f"PG: {pg.count()}   Milvus: {vec.count()} 条\n")

    # 3. 取评测集 + 拿 LLM（生成答案 + 当裁判 复用同一个）
    questions = load_questions(args.limit)
    llm = llm_mod.get_llm()
    print(f"评估 {len(questions)} 题，后端 {config.LLM_BACKEND}/{config.LLM_MODEL} ...\n")

    # 4. 跑评估（生成 + 裁判 + 聚合）
    t0 = time.time()
    results, summary = run_evaluation(
        questions, args.top_k, llm, emb, vec, pg, limit=args.limit
    )
    elapsed = time.time() - t0

    # 5. 打印汇总
    print("=" * 70)
    print(f"评估完成：{len(results)} 题，耗时 {elapsed:.1f}s（{elapsed/max(len(results),1):.2f}s/题）")
    print(f"  正确率(裁判=2): {summary['accuracy']*100:.1f}%  ({summary['correct']}/{summary['n']})")
    print(f"  部分正确: {summary['partial_rate']*100:.1f}%   错误: {summary['wrong']}/{summary['n']}")
    if summary["refusal_rate"] is not None:
        print(f"  拒答正确率: {summary['refusal_rate']*100:.1f}%")
    print(f"  召回 top1: {summary['recall_top1']*100:.1f}%   top3: {summary['recall_top3']*100:.1f}%")
    print("=" * 70)
    for r in results:
        print(f"  [{r['judge_label']}] #{r['qid']} {r['question'][:30]}")
        print(f"        recall_top3={'✓' if r['recall_top3'] else '✗'}  理由: {r['judge_reason'][:60]}")

    # 6. 落库（eval_run + eval_result）
    run_id = None
    if args.no_save:
        print("\n（--no-save：未写 eval 表）")
    else:
        run_id = pg.save_eval(
            config.LLM_MODEL, config.LLM_BACKEND, args.top_k, summary, results,
            note=f"limit={args.limit}",
        )
        print(f"\n（已写入 eval_run id={run_id} ✓）")

    # 7. 报告
    md = render_md(summary, results, run_id, config.LLM_MODEL, config.LLM_BACKEND)
    out = args.output or os.path.join(ROOT, "eval", f"report_{run_id if run_id else 'dry'}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"（报告已写: {out}）")

    pg.close()


if __name__ == "__main__":
    main()
