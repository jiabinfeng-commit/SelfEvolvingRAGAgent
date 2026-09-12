# -*- coding: utf-8 -*-
"""
重排模块（阶段 4 扩展：混合检索融合之后、返回 top-k 之前再精排一次）

为什么需要它？
- 混合检索用 RRF 把「向量语义分」和「BM25 关键词分」融在一起，但融合只是按排名相加，
  对"查询-文档真实相关度"的刻画比较粗。块一多，融合噪声被放大，top-k 里容易混入
  不太相关的块。
- 重排（rerank）用一个更准的模型，对"融合出来的候选集"逐对打分（query, doc），
  把真正相关的顶上来。这是工业界 RAG 提召回质量最划算的一刀。

两种模式（靠配置切换，不引死依赖）：
- cross：跨编码器（cross-encoder），如 BAAI/bge-reranker-v2-m3。效果最好，但要多下一个模型。
- bi   ：复用项目已有的 bge embedding 模型，把候选块重新编码算余弦相似度来重排。
         不下载任何新东西，开箱即用，对"BM25 单独命中的块"尤其有用（向量检索没排它，
         但重排能把它顶上来）。代价：语义信号和向量召回同源，提升幅度不如 cross。
- auto ：优先 cross；cross 加载失败（比如模型没下载、没网）就自动退到 bi。
         默认 auto —— 现在就能跑（bi），将来把 cross 模型下好就自动升级，零改动。

注意：重排只改变"返回顺序"，返回的还是原来的融合分（不改分数语义），
      这样下游 should_heal / gap_detected 的阈值判断不会因为分数量纲变了而崩。
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from core import config


class Reranker:
    """对混合检索的候选集做精排。线程安全由调用方（scripts/api.py）保证。"""

    def __init__(self, embedder=None, model_name: str = None,
                 mode: str = None, enabled: bool = True,
                 candidate_top_n: int = None):
        self.enabled = enabled
        self.mode = (mode or config.RERANKER_MODE).lower()
        self.model_name = model_name or config.RERANKER_MODEL
        self.embedder = embedder
        self.candidate_top_n = candidate_top_n or config.RERANKER_CANDIDATE_TOP_N
        self.model = None          # CrossEncoder 实例（仅 cross 模式有）
        self.available = False     # 是否真的能重排（不能就原样返回，服务不崩）
        if self.enabled:
            self._load()

    def _load(self):
        # auto / cross：尝试加载跨编码器
        if self.mode in ("cross", "auto"):
            # 【必须用 local_files_only=True 禁止隐式下载 —— 这里踩过一个大坑】
            #
            # 原来这里是用"临时把 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE 置 1"来防下载的，
            # 但那个做法**完全不生效**：huggingface_hub 在 **import 时**就把 HF_HUB_OFFLINE
            # 读进了模块常量（huggingface_hub.constants.HF_HUB_OFFLINE），之后再改
            # os.environ 不会改变已固化的常量。实测：
            #     import 之后            常量 = False
            #     运行时把它设成 "1" 之后  常量 = False    ← 没变
            # 后果：CrossEncoder 会真的去下载 BAAI/bge-reranker-v2-m3（约 **2.27GB**）。
            #
            # 以前之所以没出事，纯属侥幸：当时 HF 官方源在国内连不通，下载秒速失败、
            # 自动退到 bi 模式，所以掩盖了这个 bug。一旦配了 HF 镜像
            # （见 core/config.py 的 HF_ENDPOINT=hf-mirror.com），网络通了，
            # 这 2.27GB 就会在**第一个请求里静默开下**，把服务拖死、把磁盘和带宽吃光。
            #
            # local_files_only=True 才是真正可靠的开关：只在本地缓存里找，
            # 找不到立刻抛异常 → 被下面的 except 捕获 → 退 bi 模式。
            # 想用 cross 提精度，就显式把模型下到本地（scripts/fetch_model.py），
            # 之后这里会自动用上 —— 这仍是原设计的"零改动升级"体验，只是不再偷偷下载。
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(
                    self.model_name, device="cpu", max_length=512,
                    local_files_only=True,
                )
                self.available = True
                self.mode = "cross"
                return
            except Exception as e:
                # cross 失败（模型没下到本地 / 硬件不支持）-> 记一笔，往下退 bi
                print(f"[reranker] cross 模式不可用（未预下载或加载失败），退到 bi 模式: {e}")
        # bi：复用已有 embedding 模型，无需下载
        if self.mode in ("bi", "auto") and self.embedder is not None:
            self.available = True
            self.mode = "bi"
            return
        # 都不行（比如 auto 且没给 embedder）-> 原样返回，不重排
        self.available = False

    def rerank(self, query: str,
               candidates: List[Tuple[float, Dict[str, Any]]],
               top_n: int) -> List[Tuple[float, Dict[str, Any]]]:
        """
        对候选集精排，返回 top_n。

        :param candidates: [(融合分, chunk_dict), ...]（chunk_dict 需含 "content"）
        :param top_n:      最终返回几条
        :return:           重排后的 [(原始融合分, chunk_dict), ...]（只改顺序，分不变）
        """
        if not self.available or not candidates:
            return candidates[:top_n]

        docs = [c.get("content") or "" for _, c in candidates]

        if self.mode == "cross":
            pairs = [(query, d) for d in docs]
            scores = self.model.predict(pairs, show_progress_bar=False)
            scores = np.asarray(scores, dtype="float32").reshape(-1)
        else:  # bi：用现有 embedding 重算 query-doc 余弦相似
            q = self.embedder.encode_query(query)
            doc_embs = self.embedder.encode_docs(docs)   # 已归一化 → 内积 = 余弦
            scores = doc_embs @ q

        # 按重排分降序取 top_n；保留候选原来的融合分（不改下游语义）
        order = sorted(range(len(candidates)), key=lambda i: -float(scores[i]))
        return [candidates[i] for i in order[:top_n]]
