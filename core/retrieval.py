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
