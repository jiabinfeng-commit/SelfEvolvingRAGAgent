# -*- coding: utf-8 -*-
"""
入库脚本：语料 → 切片 → embedding → 双存储

用法：
    python scripts/ingest.py                      # 默认结构化切片
    python scripts/ingest.py --strategy fixed     # 固定长度切片
    python scripts/ingest.py --limit 5            # 只处理 5 篇（调试用）
    python scripts/ingest.py --rebuild            # 清空重来
"""
import os
import sys
import time
import argparse
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config
from core.chunker import get_chunker, Chunk
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore


def load_corpus(corpus_dir: str, limit: int = None):
    """读取语料目录下所有 md 文件，返回 [(doc_id, doc_name, text), ...]"""
    docs = []
    for fn in sorted(os.listdir(corpus_dir)):
        if not fn.endswith(".md"):
            continue
        path = os.path.join(corpus_dir, fn)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        doc_id = hashlib.md5(fn.encode("utf-8")).hexdigest()[:16]
        docs.append((doc_id, fn, text))
        if limit and len(docs) >= limit:
            break
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="structural", choices=["fixed", "structural"],
                    help="切片策略")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 篇文档")
    ap.add_argument("--rebuild", action="store_true", help="清空向量集合后重建")
    ap.add_argument("--corpus", default=config.CORPUS_DIR, help="语料目录")
    args = ap.parse_args()

    print("=" * 68)
    print("阶段 1：文档切片 + 双存储入库")
    print("=" * 68)
    print(config.summary())
    errs = config.validate()
    if errs:
        print("\n✗ 配置检查未通过：")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    print(f"切片策略    : {args.strategy}")
    print("=" * 68)

    # 1. 读语料
    docs = load_corpus(args.corpus, args.limit)
    if not docs:
        print(f"✗ 语料目录为空: {args.corpus}")
        sys.exit(1)
    print(f"\n[1/5] 读取语料: {len(docs)} 篇")

    # 2. 切片
    chunker = get_chunker(args.strategy)
    all_chunks: list[Chunk] = []
    t0 = time.time()
    for doc_id, doc_name, text in docs:
        chunks = chunker.split(text, doc_id, doc_name)
        all_chunks.extend(chunks)
    total_chars = sum(len(c.content) for c in all_chunks)
    print(f"[2/5] 切片完成: {len(all_chunks)} 个 chunk"
          f"（平均 {total_chars // max(1, len(all_chunks))} 字/块），耗时 {time.time()-t0:.1f}s")

    # 3. embedding
    print(f"[3/5] 加载 embedding 模型 {config.EMBED_MODEL} ...")
    t0 = time.time()
    emb = get_embedder("bge")
    print(f"      模型就绪（{emb.dim} 维），耗时 {time.time()-t0:.1f}s")

    t0 = time.time()
    texts = [c.content for c in all_chunks]
    vectors = emb.encode_docs(texts)
    print(f"      编码 {len(texts)} 条，耗时 {time.time()-t0:.1f}s")

    # 4. 写入 PostgreSQL
    print("[4/5] 写入 PostgreSQL ...")
    pg = PGStore()
    pg.init_schema()
    for doc_id, doc_name, text in docs:
        pg.upsert_document(doc_id, doc_name, char_count=len(text))
    pg.upsert_chunks([c.to_dict() for c in all_chunks])
    print(f"      {pg.count()}")

    # 5. 写入 Milvus
    print("[5/5] 写入 Milvus ...")
    vec = VecStore(dim=emb.dim)
    vec.init_collection(drop_if_exists=args.rebuild)
    vec.upsert(
        [c.chunk_id for c in all_chunks],
        vectors,
        [c.doc_id for c in all_chunks],
    )
    print(f"      向量数: {vec.count()}")

    # 汇总
    print("\n" + "=" * 68)
    print("入库完成")
    print("=" * 68)
    print(f"  文档数    : {len(docs)}")
    print(f"  chunk 数  : {len(all_chunks)}")
    print(f"  向量维度  : {emb.dim}")
    print(f"  PG 记录   : {pg.count()}")
    print(f"  向量数    : {vec.count()}")

    # 抽样展示：看看切出来的块长什么样
    print("\n--- 切片抽样（验证 heading 与位置信息）---")
    for c in all_chunks[:3]:
        print(f"\n  [{c.doc_name}] heading={c.heading!r}")
        print(f"  pos={c.char_start}-{c.char_end}  len={len(c.content)}")
        print(f"  {c.content[:120].replace(chr(10), ' ')}...")

    pg.close()


if __name__ == "__main__":
    main()
