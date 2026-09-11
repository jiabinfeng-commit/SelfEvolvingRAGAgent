#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 7：知识库体检 Agent（命令行入口）

跑四个检测器（过时/矛盾/僵尸/重复），产出一份 Markdown 健康报告，列出"该修的内容"。
体检 Agent 只读不写：报告只给建议，绝不自动改库。人工确认后再去改。

用法：
    python scripts/health_check.py                         # 跑全套，报告写 health_report.md
    python scripts/health_check.py --report out.md         # 指定报告路径
    python scripts/health_check.py --no-conflict           # 跳过矛盾检测（省 LLM）
    python scripts/health_check.py --dup-threshold 0.97    # 调重复检测阈值
    python scripts/health_check.py --print                 # 同时把报告打印到终端

依赖说明：
  - 过时/僵尸/重复 三个检测器是纯规则/纯向量，不需要 LLM；
  - 矛盾检测需要把 chunk 两两交给 LLM 判冲突，默认开启（用 config 配的后端）。
    若不想烧 LLM，加 --no-conflict 跳过。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.health_check import run_health_check


def main():
    ap = argparse.ArgumentParser(description="阶段 7：知识库体检 Agent")
    ap.add_argument("--report", default="health_report.md",
                    help="Markdown 报告输出路径（默认 health_report.md）")
    ap.add_argument("--no-conflict", action="store_true",
                    help="跳过矛盾检测（不调用 LLM，省钱）")
    ap.add_argument("--dup-threshold", type=float, default=0.95,
                    help="重复检测余弦阈值（默认 0.95）")
    ap.add_argument("--max-conflict-pairs", type=int, default=200,
                    help="矛盾检测最多比对的对数（防 N^2 爆炸，默认 200）")
    ap.add_argument("--print", action="store_true",
                    help="同时把 Markdown 报告打印到终端")
    args = ap.parse_args()

    # 1. 配置校验
    errs = config.validate()
    if errs:
        print("✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(config.summary())

    # 2. 建依赖（重依赖常驻，只建一次）
    from core.embedder import get_embedder
    from core.storage.pg_store import PGStore
    from core import llm as llm_mod

    pg = PGStore()
    emb = get_embedder("bge")
    llm = None if args.no_conflict else llm_mod.get_llm(config.LLM_BACKEND)
    try:
        report = run_health_check(
            pg=pg, emb=emb, llm=llm,
            dup_threshold=args.dup_threshold,
            max_conflict_pairs=args.max_conflict_pairs,
            report_path=args.report,
        )
    finally:
        pg.close()

    # 3. 渲染概览
    s = report["summary"]
    print("\n" + "=" * 70)
    print("知识库体检完成")
    print("=" * 70)
    print(f"文档数 {s['documents']}　块数 {s['chunks']}　发现问题 {s['total_issues']}")
    bk = s["by_kind"]
    print(f"过时 {bk.get('stale',0)} / 矛盾 {bk.get('conflict',0)} / "
          f"僵尸 {bk.get('zombie',0)} / 重复 {bk.get('duplicate',0)}")
    bs = s["by_severity"]
    print(f"严重度 high {bs['high']} / medium {bs['medium']} / low {bs['low']}")
    for skip in report.get("skipped", []):
        print(f"  ⚠️ 跳过：{skip}")
    print(f"\n报告已写入：{args.report}")
    print(f"待审影子库文档：{len(report.get('staging_docs', []))} 篇（需人工转正）")

    if args.print:
        print("\n" + "-" * 70)
        print(report["markdown"])


if __name__ == "__main__":
    main()
