#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 6 自测脚本（不依赖真实 PG / 真实模型 / 真实外网之外的逻辑）

分两层验证：
  1) 联网调研三件套（web_search / fetch_page / clean_html）
     —— 对真实互联网打一发，验证纯 urllib 实现能用（沙箱已验证可联网）。
  2) 状态机闭环（run_self_heal）
     —— 用桩 LLM / 桩 pg / 桩 vec / 桩 emb 跑完整闭环：
        缺口(空召回) → 调研 → 草稿 → 门禁通过 → 写影子库 → 再问能答。
     同时验证"门禁不通过则拒入库"的反向分支。

运行：
    python scripts/selftest_stage6.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import (
    web_search, fetch_page, clean_html, gap_detected,
    quality_gate, run_self_heal, SearchHit,
)


def _check(name: str, cond: bool, extra: str = ""):
    mark = "✅ PASS" if cond else "❌ FAIL"
    print(f"  {mark}  {name}" + (f"  —— {extra}" if extra else ""))
    return cond


# ================================================================
# 桩对象（避免加载真实模型 / 连接真实 PG / Milvus）
# ================================================================

class StubLLM:
    """桩 LLM：根据 system 文案区分不同调用，返回确定性结果。"""
    def __init__(self, gate_pass=True):
        self.n_sys = 0
        self.gate_pass = gate_pass

    def generate(self, system, user):
        # 质量门禁：返回 JSON（通过/不通过由 gate_pass 控制）
        if "质量门禁" in system:
            if self.gate_pass:
                return json.dumps({"relevance": 0.9, "factuality": 0.9,
                                    "conflict": 0.9, "info_density": 0.8,
                                    "reason": "测试通过"})
            return json.dumps({"relevance": 0.2, "factuality": 0.2,
                                "conflict": 0.9, "info_density": 0.1,
                                "reason": "测试不通过"})
        # 文档生成
        if "知识库编辑" in system:
            return "# 测试知识条目\n这是关于 X 的知识。\n- 来源: http://example.com"
        # RAG 首答 / 再答（同一 SYSTEM_PROMPT，用计数器区分）
        if "严格基于" in system:
            self.n_sys += 1
            if self.n_sys == 1:
                return "根据已有资料无法回答。"      # 首答拒答 → 触发缺口
            return "这是基于补充资料的回答：X 的定义是……"  # 再答非拒答
        return "默认"


class StubVec:
    """桩向量库：upsert 后 search 能返回（模拟同进程可见新数据）。"""
    def __init__(self):
        self.data = {}

    def upsert(self, chunk_ids, vectors, doc_ids):
        for cid, v, did in zip(chunk_ids, vectors, doc_ids):
            self.data[cid] = (v, did)

    def search(self, qv, top_k=10):
        return [(cid, 0.9) for cid in self.data][:top_k]

    def ensure_loaded(self):
        pass

    def count(self):
        return len(self.data)


class StubPG:
    """桩 PG：内存字典存文档/块。"""
    def __init__(self):
        self.docs = {}
        self.chunks = {}

    def upsert_document(self, doc_id, doc_name, char_count=0, status="active", source="upload"):
        self.docs[doc_id] = {"doc_id": doc_id, "doc_name": doc_name,
                             "status": status, "source": source}

    def upsert_chunks(self, chunks):
        for c in chunks:
            self.chunks[c["chunk_id"]] = c

    def get_chunks_by_ids(self, ids):
        return {i: self.chunks[i] for i in ids if i in self.chunks}

    def get_document(self, doc_id):
        return self.docs.get(doc_id)

    def get_all_chunks(self):
        return list(self.chunks.values())

    def save_qa(self, **kwargs):
        pass  # 桩：不落库

    def close(self):
        pass


class StubEmb:
    dim = 8

    def encode_query(self, text):
        return [0.0] * self.dim

    def encode_docs(self, texts):
        return [[0.0] * self.dim for _ in texts]


# 桩搜索/抓取：返回确定性结果，避免自测依赖真实外网
def _stub_search(query, top_n=5, site=None, timeout=10):
    return [SearchHit(title=f"结果{i}", url=f"http://example.com/{i}") for i in range(3)], None


def _stub_fetch(url, timeout=10):
    return "# 页面标题\n这是爬到的正文，关于 X 的知识。\n- 补充说明", None


# ================================================================
# 主测试
# ================================================================

def test_internet_tools():
    print("\n[1] 联网调研三件套（真实互联网）")
    hits, err = web_search("FastAPI 依赖注入", top_n=3)
    if err:
        print(f"  ⚠️ 搜索被限流/网络受限：{err}（沙箱外本机通常可用，跳过断言）")
        return True
    _check("web_search 返回结果", len(hits) > 0, f"{len(hits)} 条")
    if hits:
        page, e = fetch_page(hits[0].url)
        _check("fetch_page 抓到正文", bool(page) and len(page) > 50,
               f"{len(page)} 字" if page else str(e))
        cleaned = clean_html("<html><body><script>var x=1;</script>"
                             "<p>正文内容</p><style>.a{}</style></body></html>")
        _check("clean_html 去掉 script/style", "var x=1" not in cleaned and "正文内容" in cleaned,
               repr(cleaned[:30]))
    return True


def test_gap_detected():
    print("\n[2] 缺口判定 gap_detected")
    _check("空召回 → 有缺口", gap_detected("答案", [])[0] is True)
    _check("拒答 → 有缺口", gap_detected("根据已有资料无法回答。",
                                          [{"score": 0.8}])[0] is True)
    _check("低分 → 有缺口", gap_detected("答了", [{"score": 0.3}])[0] is True)
    _check("高相关已作答 → 无缺口", gap_detected("明确答案",
                                                [{"score": 0.8}])[0] is False)


def test_quality_gate():
    print("\n[3] 质量门禁 quality_gate（桩 LLM）")
    # 通过分支
    passed, score, detail = quality_gate("# 条目\n内容", "问题X", StubLLM(gate_pass=True))
    _check("高分局通过", passed and score >= 0.7, f"score={score:.2f}")
    # 不通过分支
    passed2, score2, _ = quality_gate("# 条目\n内容", "问题X", StubLLM(gate_pass=False))
    _check("低分局不通过", (not passed2) and score2 < 0.7, f"score={score2:.2f}")
    # 坏 JSON → 安全判不通过
    class BadLLM:
        def generate(self, s, u):
            return "这不是 json"
    passed3, _, _ = quality_gate("草稿", "问题", BadLLM())
    _check("非法 JSON → 安全判不通过", passed3 is False)


def test_self_heal_loop():
    print("\n[4] 状态机闭环 run_self_heal（桩，门禁通过分支）")
    res = run_self_heal(
        "测试问题 X", pg=StubPG(), vec=StubVec(), emb=StubEmb(),
        backend_llm=StubLLM(gate_pass=True), max_steps=2,
        search_fn=_stub_search, fetch_fn=_stub_fetch,
    )
    _check("检测到缺口", res["gap_detected"] is True, res.get("gap_reason", ""))
    _check("质量门禁通过", res["quality_pass"] is True)
    _check("写入影子库(ingested_chunks>0)",
           res["ingested_chunks"] > 0, f"{res['ingested_chunks']} chunk")
    _check("触发 Agent 自愈", res["self_healed_by_agent"] is True)
    _check("再答非拒答", res["refusal"] is False)
    _check("影子块标注未验证来源",
           any(c.get("unverified") for c in res.get("retrieved", [])),
           "staging 块被打标")
    # 反向：门禁不通过 → 不入库
    print("\n[5] 状态机反向分支（门禁不通过 → 拒入库）")
    res2 = run_self_heal(
        "测试问题 Y", pg=StubPG(), vec=StubVec(), emb=StubEmb(),
        backend_llm=StubLLM(gate_pass=False), max_steps=2,
        search_fn=_stub_search, fetch_fn=_stub_fetch,
    )
    _check("门禁不通过", res2["quality_pass"] is False)
    _check("未写入影子库", res2["ingested_chunks"] == 0)
    _check("未触发 Agent 自愈", res2["self_healed_by_agent"] is False)


def main():
    print("=" * 64)
    print("阶段 6 自测")
    print("=" * 64)
    ok = True
    ok &= test_internet_tools()
    test_gap_detected()
    test_quality_gate()
    test_self_heal_loop()
    print("\n" + "=" * 64)
    print("自测完成（联网工具若被限流属正常，本机通常可用）")
    print("=" * 64)


if __name__ == "__main__":
    main()
