# -*- coding: utf-8 -*-
"""
Milvus Lite 存储层：只存向量，不存原文

为什么用 Lite 而不是 standalone？
- standalone 官方要求 8GB 内存，在资源受限环境下容易拖垮整机
- Lite 就是本地文件，零部署，API 与 standalone 完全一致，后续可无缝切换
"""
import os
from typing import List, Dict, Tuple

from pymilvus import MilvusClient, DataType

from core import config


class VecStore:
    def __init__(self, path: str = None, collection: str = None, dim: int = None):
        self.path = path or config.MILVUS_PATH
        self.collection = collection or config.COLLECTION_NAME
        self.dim = dim or config.EMBED_DIM
        # Milvus Lite 不会自动创建父目录，必须先建好
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        self.client = MilvusClient(uri=self.path)

    # ---------------- 建集合 ----------------
    def init_collection(self, drop_if_exists: bool = False):
        if self.client.has_collection(self.collection):
            if drop_if_exists:
                self.client.drop_collection(self.collection)
            else:
                return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",          # 数据量小时精确检索；量大可换 IVF_FLAT / HNSW
            metric_type="COSINE",
        )
        self.client.create_collection(
            collection_name=self.collection,
            schema=schema,
            index_params=index_params,
        )

    # ---------------- 写入 ----------------
    def upsert(self, chunk_ids: List[str], vectors, doc_ids: List[str]):
        if not chunk_ids:
            return
        rows = [
            {"chunk_id": cid, "vector": vec.tolist(), "doc_id": did}
            for cid, vec, did in zip(chunk_ids, vectors, doc_ids)
        ]
        # 分批写入，避免单次请求过大
        for i in range(0, len(rows), 200):
            self.client.upsert(collection_name=self.collection, data=rows[i:i + 200])

    # ---------------- 加载 ----------------
    def ensure_loaded(self):
        """
        检索前必须把集合 load 到内存。

        坑点：Milvus（含 Lite）的集合在进程结束后会回到 released 状态，
        新进程直接 search 会报 "Collection is in state 'released'"。
        所以入库和检索是两个进程时，检索侧必须显式 load。
        """
        if not self.client.has_collection(self.collection):
            return
        try:
            state = self.client.get_load_state(self.collection)
            if "Loaded" not in str(state):
                self.client.load_collection(self.collection)
        except Exception:
            # 不同版本 API 行为有差异，兜底直接 load 一次
            try:
                self.client.load_collection(self.collection)
            except Exception:
                pass

    # ---------------- 检索 ----------------
    def search(self, query_vec, top_k: int = 10) -> List[Tuple[str, float]]:
        """返回 [(chunk_id, score), ...]，score 越大越相似（COSINE）"""
        self.ensure_loaded()
        res = self.client.search(
            collection_name=self.collection,
            data=[query_vec.tolist()],
            limit=top_k,
            output_fields=["doc_id"],
        )
        if not res or not res[0]:
            return []
        # pymilvus 3.x 的 Hit 对象：主键用 .id 属性（hit["id"] 不存在，会 KeyError）
        return [(str(hit.id), float(hit.distance)) for hit in res[0]]

    # ---------------- 运维 ----------------
    def count(self) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        return self.client.get_collection_stats(self.collection).get("row_count", 0)

    def delete_by_doc(self, doc_id: str):
        """删除某文档的所有向量（文档更新/删除时用）"""
        self.client.delete(
            collection_name=self.collection,
            filter=f'doc_id == "{doc_id}"',
        )
