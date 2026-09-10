# -*- coding: utf-8 -*-
"""
存储层验证脚本（不依赖 embedding 模型）

用随机向量代替真实 embedding，单独验证 PG 与 Milvus 是否可用。
跑通这一步，就说明"向量-原文分离存储"架构落地了。
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore


def main():
    print("=" * 66)
    print("存储层验证（向量与原文分离）")
    print("=" * 66)
    print(config.summary())
    print("=" * 66)

    # ---------------- PostgreSQL ----------------
    print("\n[1] PostgreSQL —— 存原文与元数据")
    pg = PGStore()
    pg.init_schema()
    print("    ✓ 建表成功（document / chunk）")

    test_doc = ("doc_verify", "验证文档.md")
    pg.upsert_document(test_doc[0], test_doc[1], char_count=1000)
    print("    ✓ 写入文档记录")

    chunks = [
        {
            "chunk_id": f"chunk_verify_{i}",
            "doc_id": test_doc[0],
            "content": f"这是第 {i} 个测试片段，用于验证 PostgreSQL 写入。",
            "heading": f"章节 {i}",
            "char_start": i * 100,
            "char_end": i * 100 + 50,
            "chunk_index": i,
            "meta": {"strategy": "verify"},
        }
        for i in range(5)
    ]
    pg.upsert_chunks(chunks)
    print(f"    ✓ 写入 {len(chunks)} 个 chunk")

    # 幂等性验证：重复写入不应产生重复记录
    pg.upsert_chunks(chunks)
    got = pg.get_chunks_by_ids([c["chunk_id"] for c in chunks])
    print(f"    ✓ 重复写入后仍为 {len(got)} 条（幂等性 OK）")

    # 回查验证
    one = pg.get_chunks_by_ids(["chunk_verify_2"]).get("chunk_verify_2")
    assert one and one["content"] == chunks[2]["content"], "回查内容不一致"
    print(f"    ✓ 按 ID 回查: {one['content'][:30]}...")

    # JSONB 元数据查询验证
    with pg.conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunk WHERE meta->>'strategy' = 'verify'")
        n = cur.fetchone()[0]
    print(f"    ✓ JSONB 元数据过滤命中 {n} 条")

    # ---------------- Milvus ----------------
    print("\n[2] Milvus Lite —— 只存向量")
    dim = config.EMBED_DIM
    vs = VecStore(dim=dim)
    vs.init_collection(drop_if_exists=True)
    print(f"    ✓ 建集合成功（{dim} 维, COSINE）")

    rng = np.random.default_rng(42)
    ids = [f"chunk_verify_{i}" for i in range(5)]
    vecs = rng.normal(size=(5, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # 归一化
    vs.upsert(ids, vecs, [test_doc[0]] * 5)
    print(f"    ✓ 写入 {len(ids)} 条向量，当前总数 {vs.count()}")

    # 检索验证：用第 0 条去查，最相似应该是它自己
    hits = vs.search(vecs[0], top_k=3)
    top_id, top_score = hits[0]
    print(f"    ✓ 检索 top1 = {top_id}, score = {top_score:.4f}")
    assert top_id == "chunk_verify_0", f"检索结果不对: {top_id}"
    assert top_score > 0.99, f"自己查自己相似度应接近 1，实际 {top_score}"
    print("    ✓ 自检索命中且相似度为 1.0（向量写入正确）")

    # 按文档删除
    vs.delete_by_doc(test_doc[0])
    print(f"    ✓ 按 doc_id 删除后剩余 {vs.count()} 条")

    # ---------------- 清理 + 汇总 ----------------
    with pg.conn.cursor() as cur:
        cur.execute("DELETE FROM document WHERE doc_id = 'doc_verify'")
    pg.conn.commit()
    vs.init_collection(drop_if_exists=True)

    print("\n" + "=" * 66)
    print("✅ 存储层验证全部通过")
    print("=" * 66)
    print(f"  PG 记录   : {pg.count()}")
    print(f"  向量数    : {vs.count()}（已清理测试数据）")
    pg.close()


if __name__ == "__main__":
    main()
