# -*- coding: utf-8 -*-
"""
阶段 7 自测：用桩对象跑四个检测器 + run_health_check 汇总，不依赖外网 / 真实 PG。

设计：检测器的输入都是"普通数据结构"（chunk 列表 / 召回集合 / 向量），所以我们可以
注入桩（假数据 + 假 LLM + 假 embedder），把每个检测器单独验证，再整体跑一遍汇总。
这样在没有 PostgreSQL / 模型 key / 外网的环境里，也能证明体检逻辑是通的。

运行：
    python scripts/selftest_stage7.py
"""
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.health_check import (
    detect_stale, detect_conflicts, detect_zombies, detect_duplicates,
    run_health_check, HealthIssue,
)


# ================================================================
# 桩对象
# ================================================================

class _FakeCursor:
    """模拟 psycopg2 的 cursor（只够 _list_staging_docs 用：execute + fetchall）。"""
    def __init__(self, rows):
        self._rows = rows
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, *a, **k):
        pass
    def fetchall(self):
        return self._rows


class StubPG:
    """假装是 PGStore：提供体检需要的几个查询方法 + 一个供 _list_staging_docs 用的 conn。"""
    def __init__(self, chunks, recalled, staging_rows=None):
        self._chunks = chunks
        self._recalled = recalled
        self._staging = staging_rows or []
        # _list_staging_docs 用 pg.conn.cursor(...)；这里给个能返回空/指定行的假 conn
        self.conn = type("C", (), {
            "cursor": lambda *a, **k: _FakeCursor(self._staging)
        })()

    def get_all_chunks_with_doc(self):
        return list(self._chunks)
    def get_recalled_chunk_ids(self):
        return set(self._recalled)
    def count(self):
        docs = {c.get("doc_id") for c in self._chunks}
        return {"documents": len(docs), "chunks": len(self._chunks)}


class StubLLM:
    """假 LLM：conflict 行为由构造参数决定（True=永远说冲突，False=永远说不冲突）。"""
    def __init__(self, say_conflict=True):
        self.say_conflict = say_conflict
    def generate(self, system, user):
        return '{"conflict": %s, "reason": "stub"}' % ("true" if self.say_conflict else "false")


class StubEmb:
    """假 Embedder：按 content 返回预设向量（已 L2 归一化，点积即余弦）。"""
    def __init__(self, vectors_by_text):
        self._v = vectors_by_text
        self.dim = 2
    def encode_docs(self, texts):
        return np.array([self._v[t] for t in texts], dtype="float32")


# ================================================================
# 测试集
# ================================================================

# 当前年用固定值，避免随系统时间波动导致断言不稳
TEST_YEAR = 2026


def _chunk(cid, doc_id, doc_name, content, created_days_ago=10):
    return {
        "chunk_id": cid, "doc_id": doc_id, "doc_name": doc_name,
        "content": content,
        "created_at": datetime.datetime.now() - datetime.timedelta(days=created_days_ago),
    }


def test_detect_stale():
    print("[1] detect_stale ...")
    chunks = [
        # 过时：时效词 + 旧年份（2023 <= 2026-2）
        _chunk("c1", "d1", "配置文档", "当前版本是 2023 年发布的最新配置说明。"),
        # 不过时：有时效词但年份很新（2025 > 2024）
        _chunk("c2", "d2", "新文档", "最新版发布于 2025 年，已适配新接口。"),
        # 不过时：纯历史叙述，无时效词
        _chunk("c3", "d3", "历史", "该项目于 2021 年立项，2023 年上线。"),
        # 过时（规则B）：时效词 + 文档很老（>365 天）
        _chunk("c4", "d4", "老文档", "目前版本支持 HTTPS。", created_days_ago=400),
    ]
    issues = detect_stale(chunks, current_year=TEST_YEAR)
    ids = {it.doc_ids[0] for it in issues}
    assert "d1" in ids, "应检出 d1（时效词+旧年份）"
    assert "d2" not in ids, "d2 年份新，不应误报"
    assert "d3" not in ids, "d3 无时效词，不应误报"
    assert "d4" in ids, "应检出 d4（时效词+文档很老）"
    print("    ✅ 过时检测 4 项断言通过")


def test_detect_conflicts():
    print("[2] detect_conflicts ...")
    # 同文档两块，StubLLM 说冲突 → 应检出
    group = [
        _chunk("c1", "d1", "同一篇", "端口应配置为 8080。"),
        _chunk("c2", "d1", "同一篇", "端口应配置为 9090。"),
    ]
    yes = detect_conflicts(group, llm=StubLLM(say_conflict=True))
    assert len(yes) == 1 and yes[0].kind == "conflict", "冲突应被检出"
    assert yes[0].severity == "high", "矛盾应为 high"

    # StubLLM 说不冲突 → 不检出
    no = detect_conflicts(group, llm=StubLLM(say_conflict=False))
    assert no == [], "非冲突不应检出"

    # llm=None → 跳过，返回空
    skip = detect_conflicts(group, llm=None)
    assert skip == [], "llm=None 应跳过（返回空）"
    print("    ✅ 矛盾检测 3 项断言通过")


def test_detect_zombies():
    print("[3] detect_zombies ...")
    chunks = [
        _chunk("c1", "d1", "A", "常被问到", created_days_ago=30),
        _chunk("c2", "d2", "B", "从没被召回", created_days_ago=30),
        _chunk("c3", "d3", "C", "刚入库", created_days_ago=0),  # 太新不判僵尸
    ]
    # 只召回过 c1
    pg = StubPG(chunks, recalled=["c1"])
    issues = detect_zombies(pg, chunks_with_doc=chunks)
    assert len(issues) == 1, "应只有 1 条僵尸汇总"
    assert "c2" in issues[0].chunk_ids, "c2 从未被召回，应为僵尸"
    assert "c1" not in issues[0].chunk_ids, "c1 被召回过，不是僵尸"
    assert "c3" not in issues[0].chunk_ids, "c3 太新，不判僵尸"
    print("    ✅ 僵尸检测断言通过")


def test_detect_duplicates():
    print("[4] detect_duplicates ...")
    vectors = {
        "X 端口是 8080": np.array([1.0, 0.0], dtype="float32"),
        "X 端口是8080":   np.array([1.0, 0.0], dtype="float32"),   # 与上一句几乎重复
        "Y 主题是认证":     np.array([0.0, 1.0], dtype="float32"),
    }
    chunks = [
        _chunk("c1", "d1", "A", "X 端口是 8080"),
        _chunk("c2", "d1", "A", "X 端口是8080"),
        _chunk("c3", "d1", "A", "Y 主题是认证"),
    ]
    emb = StubEmb(vectors)
    issues = detect_duplicates(chunks, emb, threshold=0.95)
    assert len(issues) == 1, "应只有 1 个重复簇"
    assert set(issues[0].chunk_ids) == {"c1", "c2"}, "c1/c2 应被判重复"
    assert "c3" not in issues[0].chunk_ids, "c3 与它们正交，不是重复"
    print("    ✅ 重复检测断言通过")


def test_run_health_check():
    print("[5] run_health_check 汇总 ...")
    # 构造一份能触发所有检测器的数据
    vectors = {
        "当前版本是 2023 年发布的最新配置说明。": np.array([1.0, 0.0], dtype="float32"),
        "端口应配置为 8080。": np.array([1.0, 0.0], dtype="float32"),
        "端口应配置为 9090。": np.array([1.0, 0.0], dtype="float32"),  # 与上一句重复 + 矛盾
        "从没被召回的内容": np.array([0.0, 1.0], dtype="float32"),
    }
    chunks = [
        _chunk("c1", "d1", "配置文档", "当前版本是 2023 年发布的最新配置说明。"),
        _chunk("c2", "d2", "同一篇", "端口应配置为 8080。"),
        _chunk("c3", "d2", "同一篇", "端口应配置为 9090。"),  # 与 c2 同文档矛盾
        _chunk("c4", "d3", "僵尸篇", "从没被召回的内容", created_days_ago=30),
    ]
    # c2/c3 向量相同 → 重复；c2 被召回过，c4 没被召回过
    pg = StubPG(chunks, recalled=["c2"], staging_rows=[
        ("agent_x", "[agent] 测试", datetime.datetime.now())
    ])
    emb = StubEmb(vectors)
    llm = StubLLM(say_conflict=True)   # 让矛盾被检出

    report = run_health_check(pg=pg, emb=emb, llm=llm, dup_threshold=0.95)
    assert report["summary"]["total_issues"] >= 3, "至少检出 过时/矛盾/僵尸/重复 中三类"
    md = report["markdown"]
    for title in ["过时内容", "矛盾内容", "僵尸", "重复内容", "待人工审核的影子库"]:
        assert title in md, f"报告应含章节: {title}"
    # 待审影子库应列出 agent_x
    assert "agent_x" in md, "报告应列出待审影子库文档"
    # staging_docs 应被返回
    assert any(d["doc_id"] == "agent_x" for d in report["staging_docs"]), "应返回 staging 文档"
    print(f"    ✅ 汇总通过，发现问题 {report['summary']['total_issues']} 条，报告 {len(md)} 字")


def main():
    print("=" * 70)
    print("阶段 7 自测开始（桩模式，无需外网/PG）")
    print("=" * 70)
    test_detect_stale()
    test_detect_conflicts()
    test_detect_zombies()
    test_detect_duplicates()
    test_run_health_check()
    print("=" * 70)
    print("✅ 阶段 7 自测全部通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
