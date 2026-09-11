# -*- coding: utf-8 -*-
"""
召回公共逻辑（阶段 1 / 2 / 3 共用）

为什么单独成一个模块？
- search.py（阶段1 检索验收）、ask.py（阶段2 问答闭环）、evaluate.py（阶段3 评估）
  都要做同一件事：
      问题 → bge 编码（带查询前缀）→ Milvus 搜 top-k → PG 按 chunk_id 回表拿原文
- 之前三处各写一份，维护时容易飘移（比如一处加了 rerank 另一处忘了）。
  抽到这里，保证「检索口径」永远一致：改一处，三处同时生效。

设计要点：
- retrieve() 是纯函数（不持有任何状态），依赖外部传入的 emb / vec / pg 实例。
  这样它不关心这些实例怎么来的，单测也能轻松注入 stub。
- 返回 [(score, chunk_dict), ...] 按相似度从高到低；
  chunk_dict 至少含 chunk_id / content / heading / doc_name（来自 PG 回表）。
- Milvus 只吐 chunk_id + 分数，原文是 PG 给的 —— 这就是阶段1 讲的「二级索引回表」。

> 和阶段1第六节的衔接：这里的 retrieve 就是 search.py 那段「召回」的原样提取，
> 语义没变，只是挪了个地方集中管理。
"""
from typing import List, Tuple, Dict, Any


def retrieve(question: str, emb, vec, pg, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    """
    一次完整召回：问题 → 向量 → Milvus top-k → PG 回表拿原文。

    :param question: 用户问题（或评测集里的问题）
    :param emb:       Embedder 实例，需提供 encode_query(text) -> 向量
    :param vec:       VecStore 实例，需提供 search(vector, top_k) -> [(chunk_id, score), ...]
    :param pg:        PGStore 实例，需提供 get_chunks_by_ids(ids) -> {chunk_id: 原文行}
    :param top_k:     召回多少块（阶段2/3 默认 5，见 config.RAG_TOP_K）
    :return:          [(score, chunk_dict), ...] 按 score 从高到低；
                     chunk_dict 至少含 chunk_id / content / heading / doc_name
    """
    # 1) 查询侧编码：bge 官方推荐加一句前缀，能小幅提升检索（阶段1 讲过）
    qv = emb.encode_query(question)
    # 2) Milvus 近邻搜索：只返回 chunk_id + 余弦相似度，一个字原文都没有
    hits = vec.search(qv, top_k=top_k)          # [(chunk_id, score), ...]
    if not hits:
        return []
    # 3) 拿 ID 去 PG 回表取原文（双存储配合的关键一步）
    chunk_ids = [cid for cid, _ in hits]
    chunks = pg.get_chunks_by_ids(chunk_ids)    # {chunk_id: 原文行}
    # 4) 按 Milvus 的 score 顺序拼回（保证「最相关在前」）
    out = []
    for cid, score in hits:
        c = chunks.get(cid)
        if not c:
            continue                            # 极端情况：向量有 ID 但 PG 没这行，跳过
        out.append((score, c))
    return out


def hybrid_retrieve(question: str, emb, vec, pg, bm25,
                    top_k: int = 5, vector_top_n: int = 20, bm25_top_n: int = 20,
                    rrf_k: int = 60, reranker=None, rerank_top_n=None) -> List[Tuple[float, Dict[str, Any]]]:
    """
    阶段 4 混合检索：向量语义 + BM25 关键词，用 RRF 融合后取 top_k。

    为什么要混合？
    - 向量检索：懂语义（"怎么让接口依赖别的东西" → 能找到"依赖注入"），但对精确术语迟钝。
    - BM25：精确匹配关键词（"Depends"、"HTTPS"），但不懂同义改写。
    两者互补，混合后通常比单用任一个都稳。

    融合用 RRF（Reciprocal Rank Fusion，倒数排名融合）：
        fused(d) = Σ  1 / (k + rank_list(d))
    即：对每个候选块，把它在「各条召回列表里的排名」换算成分数再加总。
    - 用**排名**而不是原始分数很关键：向量分是余弦(0~1)，BM25 分是无上界的实数，
      量纲完全不同，直接加权相加会被 BM25 的数值大小带跑偏；换成排名就没有量纲问题。
    - k=60 是 RRF 论文里的经典取值（k 越大，排名靠前的优势越不明显）。

    :param bm25:        BM25 实例（已 build 过索引）；传 None 就退化成纯向量（见下）
    :param vector_top_n / bm25_top_n: 两条链路各自先召回多少（要比 top_k 大，给融合留余量）
    :param rrf_k:       RRF 公式里的平滑常数
    :return: [(fused_score, chunk_dict), ...] 按融合分降序，长度 ≤ top_k
    """
    # 1) 向量语义召回
    qv = emb.encode_query(question)
    vec_hits = vec.search(qv, top_k=vector_top_n)      # [(chunk_id, cosine), ...]
    # 2) BM25 关键词召回
    bm_hits = bm25.search(question, top_n=bm25_top_n)  # [(chunk_id, bm25_score), ...]

    # 3) RRF 融合：按排名加权
    fused = {}
    for rank, (cid, _score) in enumerate(vec_hits, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (cid, _score) in enumerate(bm_hits, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    # 4) 按融合分降序排（先不切片，给重排留候选池）
    ordered = sorted(fused.items(), key=lambda x: -x[1])

    # 5) 回表拿原文（和 retrieve() 一样，Milvus 只有 ID）
    #    开重排时多取一点候选（rerank_top_n）喂给重排器精排；
    #    没开重排就只取 top_k，行为和以前完全一致。
    if reranker is not None and getattr(reranker, "available", False):
        cand_n = rerank_top_n or max(top_k, reranker.candidate_top_n)
    else:
        cand_n = top_k
    cand_ids = [cid for cid, _ in ordered[:cand_n]]
    chunks = pg.get_chunks_by_ids(cand_ids)
    candidates = [(score, chunks[cid]) for cid, score in ordered[:cand_n] if cid in chunks]

    # 6) 重排（只改顺序，保留原融合分）；不开重排就直接截断返回
    if reranker is not None and getattr(reranker, "available", False):
        return reranker.rerank(question, candidates, top_n=top_k)
    return candidates[:top_k]
