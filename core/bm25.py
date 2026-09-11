# -*- coding: utf-8 -*-
"""
BM25 关键词检索（阶段 4：混合检索的「关键词」那一半）

为什么需要它？
- 阶段 1~3 只有**向量语义检索**：意思相近就能召回，但对「专业术语 / 精确字符串」很迟钝。
  例如问 "Depends 怎么用"、"HTTPS 配置"，向量可能召回一堆语义相关但不含该词的块。
- BM25 是经典的**关键词精确匹配**算法（Elasticsearch / Lucene 的默认打分），
  正好补上语义检索的短板。两者融合 = 混合检索（hybrid retrieval）。

为什么自己写而不装 rank_bm25 / jieba？
- 项目一贯保持依赖精简（阶段 1 就为省事用过标准库 urllib 替代 requests）。
- BM25 公式本身很短，纯 Python 三四十行就够；494 块的规模性能完全够用。
- 中文分词不用 jieba：**按字 bigram 切分**（"依赖注入" → "依赖"/"赖注"/"注入"），
  这是中文信息检索里最经典、无依赖且效果不错的做法。

BM25 公式（对单个词 t、文档 d）：
    score(t, d) = IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
其中：
    IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))   # 越稀有的词权重越高
    tf     = 词 t 在文档 d 里的出现次数
    dl     = 文档 d 的长度（词数）
    avgdl  = 全库平均文档长度
    k1=1.5, b=0.75 （Lucene/ES 的默认经验值，一般不用调）
"""
import re
import math
from collections import Counter, defaultdict
from typing import List, Tuple


# CJK 统一表意文字范围（中日韩），用于识别需要按字切分的片段
_CJK = r"\u4e00-\u9fff"
# ASCII 词：字母数字开头，允许 _ - . （技术文档里 "pydantic.BaseSettings" 这类要整体保留）
_ASCII_WORD = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")
_CJK_SEG = re.compile(f"[{_CJK}]+")


def tokenize(text: str) -> List[str]:
    """
    把文本切成词表。策略（无外部依赖）：
      1. 全部转小写（英文大小写不敏感）
      2. ASCII 部分：按正则切成「词」，如 "fastapi" / "pydantic.settings" / "0.1"
      3. CJK 部分：按字 bigram 滑动切分，如 "依赖注入" → ["依赖","赖注","注入"]
         单字片段直接作为一词（否则 "的" 这种单字会被丢掉，但影响很小）
    """
    if not text:
        return []
    text = text.lower()
    tokens = []
    # 1) ASCII / 数字词
    for w in _ASCII_WORD.findall(text):
        tokens.append(w)
    # 2) 中文按 bigram
    for seg in _CJK_SEG.findall(text):
        if len(seg) == 1:
            tokens.append(seg)
        else:
            # 滑动窗口取 2 个字
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


class BM25:
    """
    极简 BM25 实现：建索引 → 查询 → 返回 top-n。

    用法：
        bm = BM25()
        bm.build([(chunk_id, "文本"), ...])     # 只需建一次
        bm.search("查询文本", top_n=20)         # -> [(chunk_id, score), ...] 降序
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = []          # 第 i 篇文档的 id（这里是 chunk_id）
        self.tfs = []              # 第 i 篇文档的词频 Counter
        self.doc_len = []          # 第 i 篇文档的词数
        self.df = Counter()        # 词 -> 出现在多少篇文档里
        self.inverted = defaultdict(list)   # 倒排索引：词 -> [文档下标, ...]
        self.idf = {}              # 词 -> IDF 值
        self.N = 0
        self.avgdl = 0.0

    def build(self, docs: List[Tuple[str, str]]):
        """
        建索引。docs = [(doc_id, text), ...]

        建了倒排索引，查询时只遍历「命中查询词」的文档，
        而不是全库 494 块挨个算 —— 规模变大时才不会退化成 O(N*Q)。
        """
        total_len = 0
        for idx, (doc_id, text) in enumerate(docs):
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_ids.append(doc_id)
            self.tfs.append(tf)
            self.doc_len.append(len(toks))
            total_len += len(toks)
            for t in tf:
                self.df[t] += 1
                self.inverted[t].append(idx)
        self.N = len(self.doc_ids)
        self.avgdl = (total_len / self.N) if self.N else 0.0
        # 预计算 IDF（建索引时算一次，查询时直接用）
        self.idf = {
            t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for t, df in self.df.items()
        }
        return self

    def search(self, query: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        查询，返回 [(doc_id, bm25_score), ...] 按分数降序取前 top_n。
        """
        qtoks = set(tokenize(query))
        scores = defaultdict(float)
        for t in qtoks:
            if t not in self.inverted:
                continue                      # 这个词全库没有，跳过
            idf = self.idf[t]
            for idx in self.inverted[t]:
                tf = self.tfs[idx].get(t, 0)
                dl = self.doc_len[idx]
                # BM25 分母：控制「长文档不该因为词多就得分高」
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl) if self.avgdl \
                        else tf + self.k1
                scores[idx] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        return [(self.doc_ids[idx], s) for idx, s in ranked]

    def __len__(self):
        return self.N
