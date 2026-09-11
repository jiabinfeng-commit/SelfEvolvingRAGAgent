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

-- 阶段 2：问答日志。每问一次就落一条，方便复盘"召回了什么 / 模型答了什么"
CREATE TABLE IF NOT EXISTS qa_log (
    id            SERIAL PRIMARY KEY,
    question      TEXT        NOT NULL,
    answer        TEXT        NOT NULL,
    ctx_chunk_ids TEXT[]      NOT NULL DEFAULT '{}',   -- 召回的 chunk_id（按相关度从高到低）
    ctx_scores    REAL[]      NOT NULL DEFAULT '{}',   -- 对应的相似度分数
    model         VARCHAR(64) NOT NULL DEFAULT '',
    backend       VARCHAR(20) NOT NULL DEFAULT '',
    latency_s     REAL        DEFAULT 0,               -- LLM 生成耗时（秒）
    created_at    TIMESTAMP   DEFAULT NOW()
);

-- 阶段 3：评估闭环。每次跑评估落一条 run + 每题一条 result，供自进化横向对比
CREATE TABLE IF NOT EXISTS eval_run (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMP   DEFAULT NOW(),
    model         VARCHAR(64) NOT NULL DEFAULT '',
    backend       VARCHAR(20) NOT NULL DEFAULT '',
    top_k         INTEGER     NOT NULL DEFAULT 5,
    n_questions   INTEGER     NOT NULL DEFAULT 0,
    accuracy      REAL        DEFAULT 0,               -- 正确率(裁判打分=2)
    partial_rate  REAL        DEFAULT 0,               -- 部分正确率(裁判=1)
    refusal_rate  REAL        DEFAULT NULL,            -- 拒答正确率(仅 unanswerable 题)
    recall_top1   REAL        DEFAULT 0,               -- 检索 top1 命中率
    recall_top3   REAL        DEFAULT 0,               -- 检索 top3 命中率
    avg_latency_s REAL        DEFAULT 0,
    note          TEXT        DEFAULT ''
);

CREATE TABLE IF NOT EXISTS eval_result (
    id            SERIAL PRIMARY KEY,
    run_id        INTEGER     NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    qid           INTEGER,
    question      TEXT,
    qtype         VARCHAR(20),
    gold_doc      VARCHAR(255),
    generated     TEXT,
    judge_score   INTEGER,
    judge_label   VARCHAR(20),
    judge_reason  TEXT,
    recall_top1   BOOLEAN,
    recall_top3   BOOLEAN,
    latency_s     REAL
);
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

    def get_all_chunks(self) -> List[Dict]:
        """
        取出全库 chunk（阶段 4：给 BM25 建关键词索引用）。

        只在启动混合检索时调用一次，把 494 块读进内存建索引。
        生产环境块数很大时，应该换成「在数据库侧建全文索引（PG 的 tsvector）」，
        这里块数少，直接全量拉最省事。
        """
        sql = """
            SELECT c.chunk_id, c.content, c.heading, d.doc_name
            FROM chunk c JOIN document d ON c.doc_id = d.doc_id
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

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

    # ---------------- 阶段 2：问答落库 ----------------
    def save_qa(self, question: str, answer: str,
                ctx_chunk_ids: List[str], ctx_scores: List[float],
                model: str = "", backend: str = "", latency_s: float = 0.0):
        """
        把一次问答写入 qa_log（阶段 2 闭环的"答案落库"那一步）。

        存召回的 chunk_id + 分数，是为了事后能复盘：
        "用户问 X，当时召回的是哪些块、相似度多少、模型最终答了什么"，
        这对分析 RAG 效果、定位"答非所问"的根因非常关键。
        """
        sql = """
            INSERT INTO qa_log
                (question, answer, ctx_chunk_ids, ctx_scores, model, backend, latency_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                question, answer,
                list(ctx_chunk_ids), list(ctx_scores),
                model, backend, latency_s,
            ))
        self.conn.commit()

    # ---------------- 阶段 3：评估落库 ----------------
    def save_eval(self, model: str, backend: str, top_k: int,
                  summary: Dict, results: List[Dict], note: str = ""):
        """
        把一次评估写入 eval_run（汇总）+ eval_result（逐题），供自进化横向对比。

        eval_run 存这一跑的整体指标；eval_result 存每题的「生成答案 + 裁判评分 + 召回命中」，
        以后优化检索/ prompt / 模型时，直接 SELECT 两次 run 的 accuracy / recall 比高低。
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO eval_run
                   (model, backend, top_k, n_questions, accuracy, partial_rate,
                    refusal_rate, recall_top1, recall_top3, avg_latency_s, note)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (model, backend, top_k, summary["n"], summary["accuracy"],
                 summary["partial_rate"], summary["refusal_rate"],
                 summary["recall_top1"], summary["recall_top3"],
                 summary["avg_latency_s"], note),
            )
            run_id = cur.fetchone()[0]
            rows = [
                (run_id, r.get("qid"), r.get("question"), r.get("qtype"),
                 r.get("gold_doc"), r.get("generated"), r.get("judge_score"),
                 r.get("judge_label"), r.get("judge_reason"),
                 r.get("recall_top1"), r.get("recall_top3"), r.get("latency_s"))
                for r in results
            ]
            psycopg2.extras.execute_batch(
                cur,
                """INSERT INTO eval_result
                   (run_id, qid, question, qtype, gold_doc, generated,
                    judge_score, judge_label, judge_reason, recall_top1, recall_top3, latency_s)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                rows, page_size=50,
            )
        self.conn.commit()
        return run_id

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
