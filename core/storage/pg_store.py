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

-- 阶段 8：可观测链路追踪。每次问答（如开启 trace）落一条，记录"召回了哪些块 / 分数 /
-- prompt / 答案 / 耗时 / 估算 token / reflexion 结果"，供事后复盘"为什么答成这样"。
-- 注意：prompt 字段可能很长，用 TEXT；token 是估算值（len/4 启发式，见 core/tracing.py）。
CREATE TABLE IF NOT EXISTS trace_log (
    id            SERIAL PRIMARY KEY,
    request_id    VARCHAR(40) NOT NULL,
    question      TEXT        NOT NULL,
    retrieval     VARCHAR(20) NOT NULL DEFAULT 'vector',
    backend       VARCHAR(20) NOT NULL DEFAULT '',
    recalled_ids  TEXT[]      NOT NULL DEFAULT '{}',   -- 召回的 chunk_id（按相关度从高到低）
    recalled_scores REAL[]    NOT NULL DEFAULT '{}',   -- 对应相似度/融合分
    prompt        TEXT,                               -- 发给 LLM 的 user prompt（可截断）
    answer        TEXT,
    latency_s     REAL        DEFAULT 0,
    est_tokens    INTEGER     DEFAULT 0,              -- 估算 token（prompt+answer 字符数/4）
    reflected     BOOLEAN     DEFAULT FALSE,          -- 是否触发了阶段 8 Reflexion
    reflect_pass  BOOLEAN     DEFAULT NULL,           -- Reflexion 最终是否通过事实核查
    created_at    TIMESTAMP   DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trace_req ON trace_log(request_id);
CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_log(created_at);
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

    def get_all_chunks_with_doc(self) -> List[Dict]:
        """
        阶段 7 体检用：取出全库 chunk，并带上它所属文档的 doc_name / created_at。

        get_all_chunks() 只返回 doc_name（文档名），但"过时内容"检测需要文档的
        created_at（入库时间）来判断"是不是很久没更新了"。所以这里多 join 一列
        document.created_at，返回结构比 get_all_chunks 多 doc_id / doc_created_at。
        """
        sql = """
            SELECT c.chunk_id, c.doc_id, c.content, c.heading,
                   d.doc_name, d.created_at
            FROM chunk c JOIN document d ON c.doc_id = d.doc_id
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_recalled_chunk_ids(self) -> set:
        """
        阶段 7「僵尸 chunk」检测用：汇总历史上所有被召回过的 chunk_id。

        思路（避免新开 hit_log 表）：qa_log.ctx_chunk_ids 已经记录了每次问答召回了哪些块，
        把全表展开成集合，就是"曾经被用到过"的块。从未出现在其中的块 = 僵尸块
        （入库了却从没被任何一次回答用到，可能是死内容 / 检索永远够不到）。

        前提：线上跑问答时 save=True（默认）才会写 qa_log。如果用户一直 --no-save，
        这里会得到空集，体检会误报"全是僵尸"——所以文档里要讲清这个前提。
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT ctx_chunk_ids FROM qa_log")
            rows = cur.fetchall()
        recalled = set()
        for (arr,) in rows:
            if arr:
                recalled.update(arr)
        return recalled

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

    def list_documents(self) -> List[Dict]:
        """
        列出全部文档 + 每个文档的 chunk 数（前端知识库管理页用）。

        用 LEFT JOIN + GROUP BY 一次性把 chunk 数带出来，避免前端再为每行发一次 count 请求。
        没有 chunk 的文档（如刚上传还在解析）也会出现，chunk_count=0。
        按 created_at DESC 排，最新的在前面。
        """
        sql = """
            SELECT d.doc_id, d.doc_name, d.status, d.source, d.char_count, d.created_at,
                   COUNT(c.chunk_id) AS chunk_count
            FROM document d
            LEFT JOIN chunk c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id
            ORDER BY d.created_at DESC
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_chunks_by_doc(self, doc_id: str, limit: int = 50, offset: int = 0):
        """
        列出某文档的 chunk 列表（前端切片 Drawer 用），分页。

        返回 (rows, total)：rows 是当前页，total 是总数（前端算总页数用）。
        按 chunk_index 升序：保持与原文一致的阅读顺序。
        """
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS n FROM chunk WHERE doc_id = %s", (doc_id,))
            total = cur.fetchone()["n"]
            cur.execute(
                """SELECT chunk_id, doc_id, content, heading, char_start, char_end,
                          chunk_index, meta
                   FROM chunk WHERE doc_id = %s
                   ORDER BY chunk_index
                   LIMIT %s OFFSET %s""",
                (doc_id, limit, offset),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows], total

    def delete_document(self, doc_id: str) -> int:
        """
        删除文档：级联删 chunks（FK ON DELETE CASCADE），再删 document 行。
        返回被删的 chunk 数（行数为 0 也不报错，幂等）。
        向量删除不在这里做，由调用方（scripts/api.py）同步调 VecStore.delete_by_doc，
        因为本类不依赖向量库。
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunk WHERE doc_id = %s", (doc_id,))
            n = cur.fetchone()[0]
            cur.execute("DELETE FROM document WHERE doc_id = %s", (doc_id,))
        self.conn.commit()
        return n

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

    # ---------------- 阶段 8：链路追踪落库 ----------------
    def save_trace(self, request_id: str, question: str, retrieval: str,
                   backend: str, recalled_ids: List[str], recalled_scores: List[float],
                   prompt: str, answer: str, latency_s: float, est_tokens: int,
                   reflected: bool, reflect_pass: Optional[bool]):
        """
        把一次问答的链路追踪写入 trace_log（阶段 8 可观测性的数据底座）。

        记录的内容覆盖"这次回答是怎么产生的"全链路：
        - 召回了哪些块 + 分数（定位"召回质量"问题）
        - 实际发给 LLM 的 prompt（定位"提示词"问题）
        - 答案 + 耗时 + 估算 token（成本/性能）
        - 是否触发 Reflexion、最终是否通过事实核查（定位"幻觉"问题）
        """
        sql = """
            INSERT INTO trace_log
                (request_id, question, retrieval, backend, recalled_ids, recalled_scores,
                 prompt, answer, latency_s, est_tokens, reflected, reflect_pass)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        with self.conn.cursor() as cur:
            cur.execute(sql, (
                request_id, question, retrieval, backend,
                list(recalled_ids), list(recalled_scores),
                prompt, answer, latency_s, est_tokens,
                reflected, reflect_pass,
            ))
        self.conn.commit()

    def get_traces(self, limit: int = 50) -> List[Dict]:
        """取最近的 limit 条 trace（阶段 8 看板 / 复盘用）。"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM trace_log ORDER BY created_at DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
