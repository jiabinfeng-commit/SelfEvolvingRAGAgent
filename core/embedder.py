# -*- coding: utf-8 -*-
"""
Embedding 模块

抽象出统一接口，方便阶段 3 做「换 embedding 模型」的对比实验。
现在只有一个实现（bge-small-zh），后续加 DashScope / bge-base 只要继承基类即可。
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List

from core import config


class BaseEmbedder(ABC):
    """所有 embedding 实现都要遵守的接口"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def encode_docs(self, texts: List[str]) -> np.ndarray:
        """文档侧编码（入库时用）"""
        ...

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """查询侧编码（检索时用）"""
        ...


class BGEEmbedder(BaseEmbedder):
    """
    BAAI/bge-small-zh-v1.5

    - 512 维，模型约 100MB
    - 中文语义检索效果好，CPU 上速度可接受
    - normalize_embeddings=True 后可直接用内积算余弦相似度
    """

    def __init__(self, model_name: str = None, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name or config.EMBED_MODEL
        self.device = device
        self.model = SentenceTransformer(self.name, device=device)
        # sentence-transformers 6.x 把方法改名为 get_embedding_dimension，
        # 老方法还在但会抛 FutureWarning，这里做个兼容
        getter = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(getter())

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        return self.model.encode(
            texts,
            batch_size=config.EMBED_BATCH_SIZE,
            normalize_embeddings=True,      # 归一化后内积 == 余弦相似度
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

    def encode_query(self, text: str) -> np.ndarray:
        # bge 官方建议：检索任务中，查询侧加指令前缀
        q = config.QUERY_PREFIX + text if config.QUERY_PREFIX else text
        return self.model.encode(
            [q],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")[0]


def get_embedder(name: str = "bge", **kwargs) -> BaseEmbedder:
    """工厂函数：按名字拿 embedder"""
    if name == "bge":
        return BGEEmbedder(**kwargs)
    raise ValueError(f"未知 embedder: {name}")
