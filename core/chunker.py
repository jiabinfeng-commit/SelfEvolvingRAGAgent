# -*- coding: utf-8 -*-
"""
文档切片模块

设计要点：
1. 所有 chunk 都带 char_start / char_end，用于前端溯源时定位到原文位置
2. 保留标题路径（heading），让 chunk 自带上下文（"高级中间件 > HTTPSRedirectMiddleware"）
3. 两种策略可切换，为阶段 3 的对比实验做准备
"""
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from abc import ABC, abstractmethod


@dataclass
class Chunk:
    """一个文本块"""
    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    char_start: int
    char_end: int
    chunk_index: int
    heading: str = ""          # 所属标题路径，如 "高级中间件 > GZipMiddleware"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BaseChunker(ABC):
    """切片策略基类"""

    @abstractmethod
    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        ...

    @staticmethod
    def _make_id(doc_id: str, index: int) -> str:
        """生成稳定的 chunk_id：同一份文档重复入库，ID 不变（幂等）"""
        raw = f"{doc_id}::{index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


class FixedChunker(BaseChunker):
    """
    固定长度切片 + 重叠

    最朴素的策略，作为 baseline。
    缺点：可能从句子中间切断，破坏语义。
    """

    def __init__(self, size: int = 512, overlap: int = 64):
        if overlap >= size:
            raise ValueError(f"overlap({overlap}) 必须小于 size({size})")
        self.size = size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        chunks = []
        step = self.size - self.overlap
        start, idx = 0, 0

        while start < len(text):
            end = min(start + self.size, len(text))
            piece = text[start:end].strip()
            if piece:  # 跳过纯空白片段
                chunks.append(Chunk(
                    chunk_id=self._make_id(doc_id, idx),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    content=piece,
                    char_start=start,
                    char_end=end,
                    chunk_index=idx,
                    heading=self._find_heading(text, start),
                    meta={"strategy": f"fixed_{self.size}_{self.overlap}"},
                ))
                idx += 1
            if end >= len(text):
                break
            start += step

        return chunks

    @staticmethod
    def _find_heading(text: str, pos: int) -> str:
        """向前查找最近的 markdown 标题，让 chunk 自带上下文"""
        head = text[:pos]
        lines = head.split("\n")
        for line in reversed(lines):
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                return m.group(2).strip()
        return ""


class StructuralChunker(BaseChunker):
    """
    结构化切片：按 markdown 标题层级切分

    优点：尊重文档结构，一个章节不被切断，语义完整
    处理：标题层级过深时向上合并，避免产生大量碎片
    """

    def __init__(self, max_size: int = 1024, min_size: int = 80):
        self.max_size = max_size
        self.min_size = min_size

    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        # 1. 按标题切分成 (标题路径, 正文, 起始位置)
        sections = self._split_by_heading(text)

        # 2. 太小的相邻章节合并（避免碎片）
        sections = self._merge_small(sections)

        # 3. 太大的章节再按长度二次切分
        chunks, idx = [], 0
        for heading, body, start in sections:
            pieces = self._split_oversize(body) if len(body) > self.max_size else [body]
            offset = start
            for piece in pieces:
                piece = piece.strip()
                if not piece:
                    offset += len(piece)
                    continue
                # 定位真实起始位置
                real_start = text.find(piece[:30], offset)
                if real_start == -1:
                    real_start = offset
                chunks.append(Chunk(
                    chunk_id=self._make_id(doc_id, idx),
                    doc_id=doc_id,
                    doc_name=doc_name,
                    content=piece,
                    char_start=real_start,
                    char_end=real_start + len(piece),
                    chunk_index=idx,
                    heading=heading,
                    meta={"strategy": f"structural_{self.max_size}"},
                ))
                idx += 1
                offset = real_start + len(piece)

        return chunks

    def _split_by_heading(self, text: str):
        """按 markdown 标题切分，维护标题栈以生成层级路径"""
        lines = text.split("\n")
        sections, stack = [], []
        buf, start = [], 0
        pos = 0

        def flush():
            nonlocal buf, start
            if buf:
                body = "\n".join(buf).strip()
                if body:
                    sections.append((" > ".join(stack), body, start))
            buf = []

        for line in lines:
            m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                flush()
                level, title = len(m.group(1)), m.group(2).strip()
                # 维护标题栈：同级替换，更浅则回退
                stack = stack[:level - 1]
                stack.append(title)
                start = pos
                # 关键：标题行也放进正文。
                # 否则问"GZipMiddleware 是什么"时，正文里根本没有这个词，检索会漏
                buf.append(line)
            else:
                if not buf:
                    start = pos
                buf.append(line)
            pos += len(line) + 1

        flush()
        return sections

    def _merge_small(self, sections):
        """把过小的章节合并到前一个，避免产生无意义的碎片"""
        if not sections:
            return sections
        merged = []
        for heading, body, start in sections:
            if merged and len(body) < self.min_size:
                h0, b0, s0 = merged[-1]
                merged[-1] = (h0, b0 + "\n" + body, s0)
            else:
                merged.append((heading, body, start))
        return merged

    def _split_oversize(self, body: str):
        """超长章节按段落边界二次切分，尽量不在句子中间切断"""
        pieces, cur = [], ""
        for para in body.split("\n"):
            if len(cur) + len(para) + 1 > self.max_size and cur:
                pieces.append(cur)
                cur = para
            else:
                cur = cur + "\n" + para if cur else para
        if cur:
            pieces.append(cur)
        return pieces


def get_chunker(name: str = "structural", **kwargs) -> BaseChunker:
    """工厂函数：方便后续按名字切换策略"""
    registry = {
        "fixed": FixedChunker,
        "structural": StructuralChunker,
    }
    if name not in registry:
        raise ValueError(f"未知切片策略: {name}，可选: {list(registry)}")
    return registry[name](**kwargs)
