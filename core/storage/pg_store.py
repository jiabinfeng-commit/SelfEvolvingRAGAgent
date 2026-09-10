# -*- coding: utf-8 -*-
"""
PostgreSQL 存储层：存放 chunk 原文与元数据

职责边界（重要）：
- Milvus 只存 ID + 向量，不存原文
- 原文、标题、位置信息、业务状态全部放这里
这样做的好处：原文可更新、可按元数据过滤、支持事务，而向量库不用背负大字段
"""
import json
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional

from core import config

SCHEMA_SQL = """
-- 文档表：一份原始文档一条记录
CREATE TABLE IF NOT EXISTS document (
    doc_id      VARCHAR(64) PRIMARY KEY,
    doc_name    VARCHAR(255) NOT NULL,
    status      VARCHAR(20)  NOT NULL DEFAULT 'active',   -- active / staging / archived
    source      VARCHAR(50)  NOT NULL DEFAULT 'upload',   -- upload / agent_generated
    char_count  INTEGER      DEFAULT 0,
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- 切片表：原文在这里
CREATE TABLE IF NOT EXISTS chunk (
    chunk_id     VARCHAR(64) PRIMARY KEY,
    doc_id       VARCHAR(64) NOT NULL REFERENCES document(doc_id) ON DELETE CASCADE,
    content      TEXT        NOT NULL,
    heading      TEXT        DEFAULT '',
    char_start   INTEGER     DEFAULT 0,
    char_end     INTEGER     DEFAULT 0,
    chunk_index  INTEGER     DEFAULT 0,
    meta         JSONB       DEFAULT '{}'::jsonb,
    created_at   TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunk_doc   ON chunk(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_meta ON chunk USING GIN (meta);
"""


class PGStore:
    def __init__(self, host=None, port=None, db=None, user=None, password=None):
        self.conn = psycopg2.connect(
            host=host or config.PG_HOST,
            port=port or config.PG_PORT,
            dbname=db or config.PG_DB,
            user=user or config.PG_USER,
            password=password or config.PG_PASSWORD,
        )
        self.conn.autocommit = False

    # ---------------- 建表 ----------------
    def init_schema(self):
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        self.conn.commit()

    # ---------------- 写入 ----------------
    def upsert_document(self, doc_id: str, doc_name: str, char_count: int = 0,
                        status: str = "active", source: str = "upload"):
        sql = """
            INSERT INTO document (doc_id, doc_name, char_count, status, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE
                SET doc_name = EXCLUDED.doc_name,
                    char_count = EXCLUDED.char_count,
                    status = EXCLUDED.status,
                    source = EXCLUDED.source
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (doc_id, doc_name, char_count, status, source))
        self.conn.commit()

    def upsert_chunks(self, chunks: List[Dict]):
        """批量写入 chunk（同 chunk_id 覆盖，保证重复入库不产生脏数据）"""
        if not chunks:
            return
        sql = """
            INSERT INTO chunk
                (chunk_id, doc_id, content, heading, char_start, char_end, chunk_index, meta)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE
                SET content = EXCLUDED.content,
                    heading = EXCLUDED.heading,
                    char_start = EXCLUDED.char_start,
                    char_end = EXCLUDED.char_end,
                    chunk_index = EXCLUDED.chunk_index,
                    meta = EXCLUDED.meta
        """
        rows = [
            (
                c["chunk_id"], c["doc_id"], c["content"], c.get("heading", ""),
                c.get("char_start", 0), c.get("char_end", 0),
                c.get("chunk_index", 0), json.dumps(c.get("meta", {}), ensure_ascii=False),
            )
            for c in chunks
        ]
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
        self.conn.commit()

    # ---------------- 查询 ----------------
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict]:
        """按 ID 批量取原文（检索后回查用），保持返回顺序与传入一致"""
        if not chunk_ids:
            return {}
        sql = """
            SELECT c.chunk_id, c.doc_id, c.content, c.heading,
                   c.char_start, c.char_end, c.chunk_index, c.meta, d.doc_name
            FROM chunk c JOIN document d ON c.doc_id = d.doc_id
            WHERE c.chunk_id = ANY(%s)
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (list(chunk_ids),))
            rows = cur.fetchall()
        result = {r["chunk_id"]: dict(r) for r in rows}
        # 保持顺序
        return {cid: result[cid] for cid in chunk_ids if cid in result}

    def count(self) -> Dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document")
            docs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM chunk")
            chunks = cur.fetchone()[0]
        return {"documents": docs, "chunks": chunks}

    def get_document(self, doc_id: str) -> Optional[Dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM document WHERE doc_id = %s", (doc_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
