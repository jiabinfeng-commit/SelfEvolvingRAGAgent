# -*- coding: utf-8 -*-
"""
阶段 8 自测：Reflexion 事实核查闭环 + 链路追踪落库（桩模式，无需外网/真实 PG/模型）。

覆盖：
  A. reflect(answer, contexts, llm)：grounded 真/假判别
  B. run_reflexion：① 一次通过 ② 重写后通过 ③ 仍不通过→降级安全拒答
  C. generate_answer(reflect=True, trace=True) 集成：返回结构含 reflected/reflect_pass/request_id，
     且 trace 真的写进了 PG（save_trace 被调用）

运行：
    python scripts/selftest_stage8.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core import config
from core.prompt import SYSTEM_PROMPT
from core.reflexion import reflect, run_reflexion, _REFLECT_SYSTEM, _REWRITE_SYSTEM
from core.rag import generate_answer


# ================================================================
# 桩
# ================================================================

class ReflexLLM:
    """一个 LLM 桩，按 system 区分三种角色：生成答案 / 事实核查 / 重写。"""
    def __init__(self, answer="初始答案", reflect_seq=None, rewrite="重写后的答案"):
        self.answer = answer
        self.reflect_seq = list(reflect_seq or [True])   # 每次 reflect 调用消费一个
        self.rewrite = rewrite
        self.calls = []

    def generate(self, system, user):
        self.calls.append(system)
        if system == _REFLECT_SYSTEM:
            grounded = self.reflect_seq.pop(0) if self.reflect_seq else False
            return json.dumps({"grounded": grounded, "issues": ["某说法无依据"], "reason": "stub"})
        if system == _REWRITE_SYSTEM:
            return self.rewrite
        return self.answer   # 默认：首轮答案生成（SYSTEM_PROMPT / RELAXED）


class StubEmb:
    dim = config.EMBED_DIM
    def encode_query(self, text):
        return np.zeros(self.dim, dtype="float32")
    def encode_docs(self, texts):
        return np.zeros((len(texts), self.dim), dtype="float32")


class StubVec:
    def __init__(self, hits):
        self._hits = hits
    def search(self, qv, top_k=10):
        return self._hits
    def ensure_loaded(self):
        pass


class StubPG:
    def __init__(self, chunks_by_id):
        self._chunks = chunks_by_id
        self.saved_qa = 0
        self.saved_trace = 0
        self.last_trace = None
    def save_qa(self, **kwargs):
        self.saved_qa += 1
    def save_trace(self, **kwargs):
        self.saved_trace += 1
        self.last_trace = kwargs
    def get_chunks_by_ids(self, ids):
        return {i: self._chunks[i] for i in ids if i in self._chunks}
    def close(self):
        pass


def _mk_chunks():
    return {
        "k1": {"chunk_id": "k1", "doc_id": "d1", "doc_name": "A", "heading": "h",
               "content": "端口应配置为 8080。"},
        "k2": {"chunk_id": "k2", "doc_id": "d1", "doc_name": "A", "heading": "h",
               "content": "HTTPS 需要证书。"},
    }


# ================================================================
# 测试
# ================================================================

def test_reflect():
    print("[A] reflect 事实核查 ...")
    llm = ReflexLLM(reflect_seq=[True])
    grounded, issues, reason = reflect("答案是 8080", [{"content": "端口8080", "doc_name": "A"}], llm)
    assert grounded is True, "应判有依据"
    llm2 = ReflexLLM(reflect_seq=[False])
    g2, iss2, _ = reflect("答案是 9999（瞎编）", [{"content": "端口8080", "doc_name": "A"}], llm2)
    assert g2 is False and iss2, "应判无依据并给出 issues"
    print("    ✅ reflect 断言通过")


def test_run_reflexion_pass_first():
    print("[B1] run_reflexion 一次通过 ...")
    chunks = _mk_chunks()
    emb, vec, pg = StubEmb(), StubVec([("k1", 0.8), ("k2", 0.6)]), StubPG(chunks)
    llm = ReflexLLM(reflect_seq=[True])
    ans, reflected, passed, detail = run_reflexion(
        "配置端口", "端口8080", [("k1", chunks["k1"])], emb, vec, pg, llm, top_k=5)
    assert reflected is False and passed is True, "一次通过不应触发反射行动"
    assert ans == "端口8080", "一次通过应原样返回答案"
    print("    ✅ 一次通过断言通过")


def test_run_reflexion_rewrite_then_pass():
    print("[B2] run_reflexion 重写后通过 ...")
    chunks = _mk_chunks()
    emb, vec, pg = StubEmb(), StubVec([("k1", 0.8), ("k2", 0.6)]), StubPG(chunks)
    llm = ReflexLLM(reflect_seq=[False, True], rewrite="重写后的答案")
    ans, reflected, passed, detail = run_reflexion(
        "配置端口", "端口9999", [("k1", chunks["k1"])], emb, vec, pg, llm, top_k=5)
    assert reflected is True, "应触发反射"
    assert passed is True, "重写后应通过"
    assert ans == "重写后的答案", "答案应被重写为新版本"
    assert detail["retries"] >= 1
    print("    ✅ 重写后通过断言通过")


def test_run_reflexion_degrade():
    print("[B3] run_reflexion 仍不通过 → 降级安全拒答 ...")
    chunks = _mk_chunks()
    emb, vec, pg = StubEmb(), StubVec([("k1", 0.8), ("k2", 0.6)]), StubPG(chunks)
    llm = ReflexLLM(reflect_seq=[False, False], rewrite="还是编的")
    ans, reflected, passed, detail = run_reflexion(
        "配置端口", "端口9999", [("k1", chunks["k1"])], emb, vec, pg, llm, top_k=5)
    assert reflected is True and passed is False, "应触发且最终不通过"
    assert "根据提供的资料无法回答" in ans, "应降级为安全拒答"
    assert detail.get("degraded") is True
    print("    ✅ 降级断言通过")


def test_generate_answer_integration():
    print("[C] generate_answer(reflect=True, trace=True) 集成 ...")
    chunks = _mk_chunks()
    emb, vec, pg = StubEmb(), StubVec([("k1", 0.8), ("k2", 0.6)]), StubPG(chunks)
    llm = ReflexLLM(reflect_seq=[False, True], rewrite="重写后的答案")
    res = generate_answer(
        "配置端口是多少", top_k=5, retrieval="vector",
        save=True, emb=emb, vec=vec, pg=pg, backend_llm=llm,
        reflect=True, trace=True,
    )
    assert res["reflected"] is True, "应触发反射"
    assert res["reflect_pass"] is True, "最终应通过"
    assert res["request_id"], "trace 应生成 request_id"
    assert res["answer"] == "重写后的答案", "答案应是重写版"
    assert pg.saved_qa == 1, "应写 qa_log"
    assert pg.saved_trace == 1, "应写 trace_log"
    # trace 记录应包含反射结果与召回
    t = pg.last_trace
    assert t["reflected"] is True and t["reflect_pass"] is True
    assert t["recalled_ids"] == ["k1", "k2"]
    assert t["est_tokens"] > 0
    print("    ✅ 集成断言通过（reflected/reflect_pass/request_id/trace 均正常）")


def main():
    print("=" * 70)
    print("阶段 8 自测开始（桩模式）")
    print("=" * 70)
    test_reflect()
    test_run_reflexion_pass_first()
    test_run_reflexion_rewrite_then_pass()
    test_run_reflexion_degrade()
    test_generate_answer_integration()
    print("=" * 70)
    print("✅ 阶段 8 自测全部通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
