# -*- coding: utf-8 -*-
"""
可复用的入库函数（阶段 4 服务化 + 新阶段 HTTP 上传接口共用）

============================================================
为什么抽这个文件？
============================================================
scripts/ingest.py 是「命令行批量跑语料目录」的入口，里面把整套流水线
（读文件 → parse_file → 切块 → embedding → 写 PG → 写 Milvus）按
"批跑整目录"的场景串好了。

新阶段要给前端做「文件上传」API，需要同样的流水线，但场景是：
  - 单个/少量文件（不是整目录）
  - 在 HTTP 请求里同步返回结果（不是异步打印进度）
  - 文件已经躺在临时目录（不是要去 corpus_dir 找）

所以把这套流水线抽成可复用函数 `ingest_text(doc_id, doc_name, text)`：
  - 接收已经抽好的纯文本（不关心文件来源）
  - 返回结构化结果（doc_id / chunk_count / chunks），便于 HTTP 接口直接 JSON 返回
  - 向量库集合只 init 一次（lazy，懒加载），由 VecStore 自身保证

scripts/ingest.py 的 main() 改为调用本文件的函数（行为完全等价，避免双实现漂移）。
scripts/api.py 的 /upload 端点也调用本文件的函数。

============================================================
幂等性
============================================================
doc_id 由调用方决定（corpus 用 md5(filename)；上传也用 md5(原文件名)），
chunk_id = md5(f"{doc_id}::{index}")[:16] 在 core.chunker 里固定生成。
同 doc_id + 同切片策略重跑，PG 走 ON CONFLICT 覆盖，Milvus 走 upsert 覆盖，
所以「重新上传同名文件」是安全的（更新而非追加）。
"""
import hashlib
import os
from typing import List, Dict, Optional

from core.chunker import get_chunker
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import config


# -----------------------------------------------------------
# doc_id 生成
# -----------------------------------------------------------
def make_doc_id(filename: str) -> str:
    """
    从原始文件名生成稳定 doc_id（16 位 md5）。

    与 scripts/ingest.py 的 corpus 入库口径一致：同一文件名 → 同一 doc_id。
    这样：
      - 重新上传同名文件 = 覆盖更新（幂等）
      - 语料目录里某文件被替换 + 重跑 ingest = 覆盖更新
    副作用：上传文件名如果撞上语料目录里的文件名，会合并。v1 接受这个语义，
    因为两条路（拖文件上传 / 把文件放语料目录）目标都是「让这份知识进库」，
    撞名时谁后跑谁说了算，符合直觉。
    """
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------
# 进度回调助手
# -----------------------------------------------------------
def _notify(progress_cb, stage: str, frac: float):
    """
    安全地回调进度。progress_cb 为 None 时啥也不做；
    回调内部抛异常也吞掉 —— 进度上报永远不能影响入库主流程。
    """
    if progress_cb is None:
        return
    try:
        progress_cb(stage, max(0.0, min(1.0, float(frac))))
    except Exception:
        pass


# -----------------------------------------------------------
# 核心：单文档入库
# -----------------------------------------------------------
def ingest_text(
    doc_id: str,
    doc_name: str,
    text: str,
    *,
    strategy: str = "recursive",
    pg: Optional[PGStore] = None,
    emb=None,
    vec: Optional[VecStore] = None,
    progress_cb=None,
) -> Dict:
    """
    把一段纯文本走完「切块 → embedding → 写 PG → 写 Milvus」全流程。

    入参：
      doc_id     稳定文档 ID（调用方用 make_doc_id(filename) 生成）
      doc_name   显示名（原始文件名）
      text       已经 parse_file() 抽好的纯文本
      strategy   切片策略：recursive（默认）/ fixed / structural
      pg/emb/vec 可选注入（测试时塞桩；不传则按默认配置新建）
                 注意：新建 PGStore / Embedder / VecStore 都是较重的操作，
                 HTTP 上传场景下应在 lifespan 里建好单例复用，避免每次请求都重连。
      progress_cb 可选进度回调 progress_cb(stage, frac)：
                 stage ∈ {"切片","向量化","入库"}（解析在 ingest_file 上报），frac 0~1。
                 供异步上传的进度条使用；不传则完全不回调（零开销）。

    返回：
      {
        "doc_id": str,
        "doc_name": str,
        "chunks": int,            # 实际写入的 chunk 数
        "char_count": int,        # 原文字符数
        "chunk_ids": List[str],   # 写入的 chunk_id 列表（供 VecStore 写入对应向量）
      }

    异常：
      - text 为空：返回 chunks=0，不报错（前端可据此提示「文件无内容」）
      - 切片/embedding/写入任意一步失败：向上抛异常，由 HTTP 层转 503 / 任务状态 failed
    """
    text = text or ""
    char_count = len(text)

    if not text.strip():
        # 空文档：仍然 upsert document 行（前端能看到「这个文件被识别了但内容为空」），
        # 但不入 chunk、不写向量
        if pg is not None:
            pg.upsert_document(doc_id, doc_name, char_count=0)
        _notify(progress_cb, "切片", 1.0)
        _notify(progress_cb, "向量化", 1.0)
        _notify(progress_cb, "入库", 1.0)
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunks": 0,
            "char_count": 0,
            "chunk_ids": [],
        }

    # 1) 切片
    _notify(progress_cb, "切片", 0.0)
    chunker = get_chunker(strategy)
    chunks = chunker.split(text, doc_id, doc_name)
    _notify(progress_cb, "切片", 1.0)
    if not chunks:
        if pg is not None:
            pg.upsert_document(doc_id, doc_name, char_count=char_count)
        _notify(progress_cb, "向量化", 1.0)
        _notify(progress_cb, "入库", 1.0)
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "chunks": 0,
            "char_count": char_count,
            "chunk_ids": [],
        }

    # 2) embedding（向量化是最耗时的一步，这里把批次进度透传给进度条）
    if emb is None:
        emb = get_embedder("bge")
    texts = [c.content for c in chunks]
    _notify(progress_cb, "向量化", 0.0)
    if progress_cb is None:
        vectors = emb.encode_docs(texts)
    else:
        vectors = emb.encode_docs(texts, progress_cb=lambda f: _notify(progress_cb, "向量化", f))
    _notify(progress_cb, "向量化", 1.0)

    # 3) 写 PG：先 upsert document 行，再批量 upsert chunks
    _notify(progress_cb, "入库", 0.0)
    if pg is not None:
        pg.upsert_document(doc_id, doc_name, char_count=char_count)
        pg.upsert_chunks([c.to_dict() for c in chunks])

    # 4) 写 Milvus：按 chunk_id 写入向量
    if vec is not None:
        vec.upsert(
            [c.chunk_id for c in chunks],
            vectors,
            [c.doc_id for c in chunks],
        )
    _notify(progress_cb, "入库", 1.0)

    return {
        "doc_id": doc_id,
        "doc_name": doc_name,
        "chunks": len(chunks),
        "char_count": char_count,
        "chunk_ids": [c.chunk_id for c in chunks],
    }


# -----------------------------------------------------------
# 文件路径入库（给 scripts/ingest.py 的 CLI 用；HTTP 上传也走它）
# -----------------------------------------------------------
def ingest_file(
    path: str,
    *,
    strategy: str = "recursive",
    pg: Optional[PGStore] = None,
    emb=None,
    vec: Optional[VecStore] = None,
    progress_cb=None,
) -> Dict:
    """
    从文件路径入库：parse_file() 抽文本 → ingest_text() 走流水线。
    doc_id 用文件名 md5 生成（与 corpus 口径一致）。
    progress_cb(stage, frac) 会先报 "解析"，再透传给 ingest_text 报"切片/向量化/入库"。
    """
    from core.chunker import parse_file  # 延迟 import，避免循环

    doc_name = os.path.basename(path)
    doc_id = make_doc_id(doc_name)
    _notify(progress_cb, "解析", 0.0)
    text = parse_file(path)
    _notify(progress_cb, "解析", 1.0)
    return ingest_text(
        doc_id, doc_name, text,
        strategy=strategy, pg=pg, emb=emb, vec=vec, progress_cb=progress_cb,
    )
