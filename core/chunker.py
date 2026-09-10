# -*- coding: utf-8 -*-
"""
文档切片模块（格式无关版）

============================================================
为什么这次要大改？
============================================================
旧版只有两种策略：
  - structural：按 markdown 标题切（强依赖"文档是 markdown 且有标题"）
  - fixed：固定长度切
问题是：structural 假设输入一定是结构化的 markdown，换成一个 PDF / Word / HTML
就直接傻眼——它根本不认那些格式。换句话说，旧版是"先看了数据长什么样，
再照着写切法"，属于把数据格式写死进实现。

生产环境里文档类型是不固定的（PDF、Word、HTML、纯文本、Markdown 混着来），
正确的做法是**把"解析格式"和"切块"彻底解耦**：
  1. parse_file()   负责把任意格式的文件抽成"纯文本"（这一步才关心格式）
  2. 切分器        只认"纯文本"，不再关心它原来是什么文件
这样无论丢什么文件进来，切块逻辑都不变——这就是"任意文件都能切块"。

============================================================
设计要点
============================================================
1. Chunk 都带 char_start / char_end，用于前端溯源时定位到原文位置
2. 默认策略改为 recursive（递归字符切分）：完全格式无关，按
   "段落 → 行 → 句末标点 → 空格 → 字符" 的优先级递归切，尽量不在句子中间切断
3. structural 保留为"有标题时"的增强项：探测到 markdown 标题才用之，否则自动退化
4. 所有 chunk_id 由 (doc_id, 序号) 稳定生成，重入库同文档 ID 不变（幂等）
"""
import os
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
from abc import ABC, abstractmethod
from html.parser import HTMLParser


# ----------------------------------------------------------------
# 支持的文件类型
# 说明：md/txt 等直接读；html 用标准库抽文本；pdf/docx 由懒加载的第三方库处理。
# 不在列表里的扩展名，parse_file 会尝试按 UTF-8 纯文本读，仍失败才报错。
# ----------------------------------------------------------------
SUPPORTED_EXTENSIONS = (
    ".md", ".markdown", ".txt", ".text", ".log",
    ".csv", ".json", ".yaml", ".yml",
    ".html", ".htm",
    ".pdf", ".docx",
)


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
    heading: str = ""          # 所属标题路径（如 "高级中间件 > GZipMiddleware"）；格式无关切片留空
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ================================================================
# 第一层：解析（把任意格式变成纯文本）—— 这一层才关心文件类型
# ================================================================

def parse_file(path: str) -> str:
    """
    把任意支持的文件抽成纯文本。

    职责边界：本函数只负责"格式 → 纯文本"的转换；转换完之后，下游切分器
    就只看文本，不再管它原本是 PDF 还是 Word。这就是解耦的关键。

    各格式处理：
      - md/txt/csv/json/yaml/log：直接按 UTF-8 读（这些本就是文本）
      - html/htm：用标准库 HTMLParser 抽正文，丢掉 <script>/<style>
      - pdf：懒加载 pdfplumber（没装就给出明确的安装提示，而非莫名其妙崩溃）
      - docx：懒加载 python-docx
      - 其它扩展名：兜底尝试按 UTF-8 读，失败则抛清晰错误
    """
    ext = os.path.splitext(path)[1].lower()

    # 1) 文本类：直接读
    if ext in (".md", ".markdown", ".txt", ".text", ".log",
               ".csv", ".json", ".yaml", ".yml"):
        return _read_text(path)

    # 2) HTML：抽正文
    if ext in (".html", ".htm"):
        return _strip_html(path)

    # 3) PDF：需要第三方库
    if ext == ".pdf":
        return _read_pdf(path)

    # 4) Word：需要第三方库
    if ext == ".docx":
        return _read_docx(path)

    # 5) 兜底：当纯文本读，读不了就明确报错
    try:
        return _read_text(path)
    except Exception as e:
        raise ValueError(
            f"不支持或无法读取的文件: {path}（扩展名 {ext}）。"
            f"如确为文本文件请检查编码；PDF/DOCX 请先安装对应库。"
            f"原始错误: {e}"
        )


def _read_text(path: str) -> str:
    """按 UTF-8 读取文本文件，并归一化换行符（\r\n / \r → \n）。"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _HtmlTextExtractor(HTMLParser):
    """
    简易 HTML → 文本提取器（仅用标准库，不引新依赖）。

    做法：遍历标签，遇到 <script>/<style> 就跳过其内容；遇到块级标签
    （p/div/li/h1~h6/tr）插入换行以保留段落感；其余标签内的文本原样收集。
    """
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0  # 进入 script/style 的嵌套层数

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        # 块级标签前补一个换行，让段落之间有分隔
        if tag in ("p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def _strip_html(path: str) -> str:
    """用 _HtmlTextExtractor 把 HTML 转成纯文本，并压缩多余空行。"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        html = f.read()
    ex = _HtmlTextExtractor()
    ex.feed(html)
    text = "".join(ex.parts)
    # 把 3 个以上连续空行压成 2 个，避免切片时被大量空白占据
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: str) -> str:
    """
    读取 PDF 正文。pdfplumber 不是项目默认依赖，这里懒加载：
    没装时明确提示安装命令，而不是抛一个看不懂的 ImportError。
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError(
            "读取 PDF 需要 pdfplumber，请先安装：pip install pdfplumber"
        )
    with pdfplumber.open(path) as pdf:
        # 逐页抽文本，页间用两个换行分隔，保留"换页"这一结构边界
        pages = [p.extract_text() or "" for p in pdf.pages]
    return "\n\n".join(pages).strip()


def _read_docx(path: str) -> str:
    """读取 Word 文档正文（懒加载 python-docx）。"""
    try:
        import docx
    except ImportError:
        raise RuntimeError(
            "读取 DOCX 需要 python-docx，请先安装：pip install python-docx"
        )
    document = docx.Document(path)
    paras = [p.text for p in document.paragraphs]
    return "\n".join(paras).strip()


# ================================================================
# 第二层：切分（只认纯文本，完全不关心原格式）
# ================================================================

class BaseChunker(ABC):
    """切片策略基类"""

    @abstractmethod
    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        ...

    @staticmethod
    def _make_id(doc_id: str, index: int) -> str:
        """
        生成稳定的 chunk_id：同一份文档、同一个序号 → 同一个 ID。

        用途：重入库时用 upsert 覆盖旧块而不是追加，避免重复（幂等）。
        注意：换切分策略会改变序号，因此换策略后旧块 ID 会变，
        需要 ingest 时加 --rebuild 清掉旧集合再建。
        """
        raw = f"{doc_id}::{index}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


class RecursiveCharSplitter(BaseChunker):
    """
    递归字符切分（格式无关，生产环境默认策略）

    核心思想（对齐 LangChain 的 RecursiveCharacterTextSplitter）：
    准备一组"分隔符优先级"，从粗到细依次尝试：
        ["\\n\\n", "\\n", "。", "！", "？", ".", "!", "?", " ", ""]
      - 先按空行(段落)切；切完还有超长段，再按换行切；
      - 还超长，按中/英文句末标点切；再按空格；最后按字符硬切。
    这样能"尽量在语义边界（段落/句子）处断开"，而不是像 fixed 那样
    可能从一句话中间劈开。它不依赖任何文档格式，所以任意文件都适用。

    两阶段：
      1) _recursive_split：把全文递归切成"原子段"，每段 <= size，并记录
         它在【原文本】中的起止偏移（char_start/char_end 用于溯源）。
      2) _merge：贪心地把相邻原子段拼成块，块长不超过 size，并制造 overlap 重叠。
    """

    # 分隔符优先级：粗 → 细。空串 "" 表示"一个字符都不剩时按字符硬切"
    SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]

    def __init__(self, size: int = 512, overlap: int = 64):
        if overlap >= size:
            raise ValueError(f"overlap({overlap}) 必须小于 size({size})")
        self.size = size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        text = text or ""
        if not text.strip():
            return []

        # 1) 递归切成原子段（每段 <= size，且带原文本偏移）
        atoms = self._recursive_split(text, 0, self.SEPARATORS)
        # 2) 合并成带重叠的块
        merged = self._merge(atoms)

        # 3) 封装成 Chunk
        chunks = []
        for idx, (piece, s, e) in enumerate(merged):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Chunk(
                chunk_id=self._make_id(doc_id, idx),
                doc_id=doc_id,
                doc_name=doc_name,
                content=piece,
                char_start=s,
                char_end=e,
                chunk_index=idx,
                heading="",  # 格式无关切片没有"标题层级"概念，留空
                meta={"strategy": f"recursive_{self.size}_{self.overlap}"},
            ))
        return chunks

    def _recursive_split(self, text: str, start: int, seps: List[str]) -> List[Tuple[str, int, int]]:
        """
        递归切分，返回 [(片段文本, 原文本起始偏移, 原文本结束偏移), ...]

        - 若整段已 <= size，直接作为原子段返回。
        - 否则按 seps 里第一个"存在于文本中"的分隔符切开，对每段递归。
        - 用 cursor 维护原文本偏移：切掉的分隔符也占长度，要跳过。
        - seps 末尾的 "" 表示字符级兜底：按 size 硬切，保证一定能切完。
        """
        if len(text) <= self.size:
            return [(text, start, start + len(text))]

        sep = seps[-1]
        for s in seps:
            if s == "":
                # 字符级兜底：每 size 个字符一段
                res = []
                for i in range(0, len(text), self.size):
                    piece = text[i:i + self.size]
                    res.append((piece, start + i, start + i + len(piece)))
                return res
            if s in text:
                sep = s
                break

        # 按 sep 切分，逐段递归；cursor 从 start 起，跳过每段及其后的分隔符
        res = []
        cursor = start
        for part in text.split(sep):
            if part:
                res.extend(self._recursive_split(part, cursor, seps))
            cursor += len(part) + len(sep)
        return res

    def _merge(self, atoms: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """
        贪心合并原子段成块，块长 <= size，并制造 overlap 重叠。

        - cur_text 累积当前块内容；拼接时用 "\\n" 代替被丢弃的原分隔符（显示够用，
          真正的原文位置由 char_start/char_end 记录）。
        - 当再加一段会超 size：先把当前块定稿，新块复用上块末尾 overlap 个字符，
          形成上下文重叠，新块的 char_start 回退 overlap 以匹配。
        """
        chunks = []
        cur_text, cur_start, cur_end = "", None, None
        for t, s, e in atoms:
            candidate = (cur_text + "\n" + t) if cur_text else t
            if not cur_text or len(candidate) <= self.size:
                cur_text = candidate
                cur_start = s if cur_start is None else cur_start
                cur_end = e
            else:
                chunks.append((cur_text, cur_start, cur_end))
                if self.overlap > 0:
                    tail = cur_text[-self.overlap:]
                    cur_text = tail + "\n" + t
                    cur_start = max(0, cur_end - self.overlap)
                else:
                    cur_text, cur_start = t, s
                cur_end = e
        if cur_text:
            chunks.append((cur_text, cur_start, cur_end))
        return chunks


class FixedChunker(BaseChunker):
    """
    固定长度切片 + 重叠（朴素 baseline，保留用于对比实验）

    缺点：可能从句子中间切断，破坏语义。仅作为对照基线存在。
    """

    def __init__(self, size: int = 512, overlap: int = 64):
        if overlap >= size:
            raise ValueError(f"overlap({overlap}) 必须小于 size({size})")
        self.size = size
        self.overlap = overlap

    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        text = text or ""
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
                    heading="",
                    meta={"strategy": f"fixed_{self.size}_{self.overlap}"},
                ))
                idx += 1
            if end >= len(text):
                break
            start += step

        return chunks


class StructuralChunker(BaseChunker):
    """
    结构化切片：按 markdown 标题层级切分（"有标题时"的增强项）

    适用前提：输入文本含 markdown 标题（# / ## / ###...）。
    对没有标题的纯文本/PDF，它退化为"整篇一个 section → 超长再按段落二次切"，
    等价于朴素的段落切分，不会报错。

    优点：尊重文档结构，一个章节不被切断，语义完整。
    """

    def __init__(self, max_size: int = 1024, min_size: int = 80):
        self.max_size = max_size
        self.min_size = min_size

    def split(self, text: str, doc_id: str, doc_name: str) -> List[Chunk]:
        text = text or ""
        if not text.strip():
            return []

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
                stack = stack[:level - 1]
                stack.append(title)
                start = pos
                # 标题行也放进正文：否则问"GZipMiddleware 是什么"时正文里没有这词，检索会漏
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


def get_chunker(name: str = "recursive", **kwargs) -> BaseChunker:
    """
    工厂函数：按名字拿到切分器。

    默认 "recursive"（格式无关）。可选：
      - recursive   ：递归字符切分，任意文本都适用（推荐默认）
      - fixed       ：固定长度，朴素 baseline
      - structural  ：markdown 标题切分（仅当文档有标题时更优）
    """
    registry = {
        "recursive": RecursiveCharSplitter,  # 默认：格式无关
        "fixed": FixedChunker,
        "structural": StructuralChunker,
    }
    if name not in registry:
        raise ValueError(f"未知切片策略: {name}，可选: {list(registry)}")
    return registry[name](**kwargs)
