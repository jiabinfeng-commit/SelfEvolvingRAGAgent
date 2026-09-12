#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端要打的全 API（阶段 10：知识库管理 + 问答 + 可观测）

============================================================
这个文件 vs scripts/serve.py 的关系
============================================================
scripts/serve.py 是阶段 4 服务化的最小版（/health + /ask），
面向"把 RAG 问答包成 HTTP"的演示目的。

本文件（scripts/api.py）是**前端 React 页面要打的后端**：
  - 面向 RAGFlow 风格的知识库管理（上传 / 列表 / 切片查看 / 删除）
  - 问答仍然走 generate_answer，但补齐了阶段 5/6/8 的开关
  - 加 CORS，方便 Vite dev server 跨源打过来

为什么不直接扩 serve.py？
  1. serve.py 是阶段 4 的交付物，扩它会让"阶段 4"和"阶段 10"两批代码混在一起，
     历史回看不友好。
  2. 接口路径约定不同：serve 用 /ask、/health；本文件统一前缀 /api/*，
     跟前端 axios baseURL 对齐。
所以两个文件并存：serve.py 给阶段 4 演示，api.py 给前端。

============================================================
接口清单
============================================================
GET  /api/health                       健康检查（含存储条数）
GET  /api/stats                        总览：文档/块/向量/待审影子库
GET  /api/llm-info                     返回 LLM 后端/模型名（前端状态栏展示用）
GET  /api/documents                    列出全部文档（含每文档 chunk_count）
POST /api/documents/upload             批量上传文件（multipart, field="files"）→ 默认异步，返回 job_id
GET  /api/jobs/{job_id}                查询异步入库任务进度（前端进度条轮询用）
GET  /api/jobs                         列出进程内全部入库任务
GET  /api/documents/{doc_id}           单个文档详情
GET  /api/documents/{doc_id}/chunks    切片列表（分页：?limit=&offset=）
DELETE /api/documents/{doc_id}         删除文档（PG + Milvus 同步删）
POST /api/ask                          问答（带 agent_heal / reflect / trace）
POST /api/retrieve                     仅检索（检索测试页用：返回召回片段，不调 LLM）
POST /api/health-check                 对单个文档做体检（跑一轮自愈/质检）
GET  /api/traces                       最近的 trace 记录（?limit=1~500，默认 50）

============================================================
设计原则
============================================================
- 惰性建立引擎依赖（和 serve.py 一致）：模型/连接首请求时建，缓存复用；
  存储临时不可达也能先起服务，/health 返回 degraded。
- 错误处理：用户输入错（坏 doc_id、坏策略）→ 400；
  存储/LLM 不可用 → 503；找不到资源 → 404。统一 JSON 错误体。
- 跨域：开发环境放行所有 origin（*），生产应收敛（注释里说明了）。

============================================================
运行
============================================================
    python scripts/api.py                 # 默认 http://127.0.0.1:8000
    python scripts/api.py --host 0.0.0.0 --port 8080
    浏览器打开 http://127.0.0.1:8000/docs  （FastAPI 自带 Swagger 试接口）
"""
import os
import sys
import time
import uuid
import shutil
import argparse
import threading
import contextlib
import dataclasses
import tempfile
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.bm25 import BM25
from core.reranker import Reranker
from core.rag import generate_answer
from core.agent import run_self_heal
from core.ingest import ingest_file
from core.retrieval import retrieve, hybrid_retrieve
from core.health_check import run_health_check


# ================================================================
# 进程级单例：和 serve.py 一样的惰性建 + 缓存
# ================================================================
STATE: dict = {
    "emb": None,
    "pg": None,
    "vec": None,
    "llm": None,
    "bm25": None,
    "reranker": None,
}

# BM25 增量更新（上传加块 / 删除去块）与查询搜索共享同一份索引，
# 用这把锁避免后台入库线程改索引时，查询线程正好在读导致状态错乱。
BM25_LOCK = threading.Lock()

# 引擎构建锁（和自己这行上面的 BM25_LOCK 是两把不同的锁，不要合并）：
# BM25_LOCK 保护的是"改已有索引"，本锁保护的是"建引擎单例"。详见 ensure_engine 的注释。
ENGINE_LOCK = threading.Lock()


def ensure_engine():
    """
    首请求时建引擎依赖；已建复用。和 serve.py 的 ensure_engine 同款。

    【为什么必须加锁】—— 这里踩过一个真实的坑：
      本文件的路由全是**同步函数**（`def` 而不是 `async def`），FastAPI 会把它们
      丢到线程池里执行，也就是**多个请求会真的并行进入本函数**。于是出现经典的
      check-then-act 竞态：

          线程A：看到 STATE["vec"] is None  → 开始构造 VecStore
          线程B：也看到 STATE["vec"] is None → 也去构造 VecStore
                 （此刻 A 还没把结果赋回 STATE，所以 B 读到的仍然是 None）

      Milvus Lite 是**单进程独占**的嵌入式库：同一路径被打开两次，第二次会抛
          milvus_lite.exceptions.DataDirLockedError:
              another process holds the lock on '.../milvus.db'
      注意 flock 是按「打开的文件描述」计的，所以**同一个进程**内开两次一样冲突
      —— 报错文案说 "another process"，其实同进程也会中招，很有迷惑性。

      典型现场：前端首屏并发打 /api/health 和 /api/documents，
      一个 200 OK、另一个 503 Service Unavailable（见 logs 里两条紧挨的 INFO）。

    【修法】双检锁（double-checked locking）：
      1) 锁外快路径：依赖都建好了就直接返回 —— 稳态下几乎零开销，不用每次抢锁
      2) 抢锁 → 锁内**再检查一次**（可能已被先到的线程建好了，命中就跳过）
    """
    # 快路径：六个槽位都已有值 → 直接返回，不进锁。
    # bm25 / reranker 建失败时会被置为 None（降级），那种情况会落到锁里重试，
    # 保持和加锁前完全一致的语义。
    if all(STATE[k] is not None for k in
           ("emb", "pg", "vec", "llm", "bm25", "reranker")):
        return

    with ENGINE_LOCK:
        # 慢路径：拿到锁后再判断一次 —— 这段时间里其他线程可能已经建好了
        if STATE["emb"] is None:
            STATE["emb"] = get_embedder("bge")
        if STATE["pg"] is None:
            STATE["pg"] = PGStore()
            STATE["pg"].init_schema()
        if STATE["vec"] is None:
            STATE["vec"] = VecStore(dim=STATE["emb"].dim)
            # 不强制 init_collection：空集合也能 upsert（VecStore.init_collection 是幂等的）
            with contextlib.suppress(Exception):
                STATE["vec"].init_collection()
        if STATE["llm"] is None:
            STATE["llm"] = llm_mod.get_llm(config.LLM_BACKEND)
        if STATE["bm25"] is None:
            # BM25 依赖 PG 全量 chunk；建失败不影响纯向量检索
            try:
                all_chunks = STATE["pg"].get_all_chunks()
                STATE["bm25"] = BM25().build(
                    [(c["chunk_id"], (c.get("content") or "")) for c in all_chunks]
                )
            except Exception:
                STATE["bm25"] = None
        if STATE["reranker"] is None:
            # 重排器：默认 auto（优先 cross-encoder，加载不到自动退 bi 复用 bge）。
            # 建失败/不可用也不影响主流程，hybrid_retrieve 会原样返回。
            try:
                STATE["reranker"] = Reranker(embedder=STATE["emb"])
            except Exception:
                STATE["reranker"] = None


# ================================================================
# 异步入库任务（上传后立即返回 job_id，后台线程跑，前端轮询进度）
# ================================================================
# 为什么需要它？
#   入库要做「解析 → 切片 → embedding → 双写」，几十个文件可能要几十秒到几分钟。
#   如果同步阻塞在 HTTP 请求里，前端只能干等到超时，看不到任何中间进度。
#   所以：上传接口只负责"收文件 + 建任务 + 起线程"就立刻返回，
#   真正的重活在后台线程里按阶段更新进度，前端轮询 GET /api/jobs/{id} 拿进度。
#
# 设计取舍：
#   - 不引 Celery/RQ 等任务队列（要 Redis/额外进程），单机 demo 用"内存任务表 + 线程"足够。
#   - 任务是进程内的：服务重启会丢任务记录（但已入库的数据不丢）。生产应换成持久化队列。
#   - embedding 是 CPU 密集，但 sentence-transformers 计算时会释放 GIL，后台线程不阻塞事件循环。
JOBS: dict = {}                 # job_id -> job 状态字典
JOBS_LOCK = threading.Lock()    # 保护 JOBS 的读写
_ENGINE_LOCK = threading.Lock() # 保护 ensure_engine（多任务并发时只建一次单例）
JOB_TTL_S = 3600                # 任务记录保留 1 小时，之后被清理

# 每个文件的阶段权重（和为 1）。向量化最耗时，给最大权重，进度条才"走得匀"。
STAGE_WEIGHTS = [("解析", 0.05), ("切片", 0.05), ("向量化", 0.80), ("入库", 0.10)]


def _prune_jobs():
    """清理超过 TTL 的旧任务记录（避免内存无限增长）。"""
    now = time.time()
    with JOBS_LOCK:
        for jid in [k for k, v in JOBS.items() if now - v["created_at"] > JOB_TTL_S]:
            JOBS.pop(jid, None)


def _create_job(filenames: List[str]) -> str:
    """建一个新任务，返回 job_id。"""
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "pending",          # pending / running / done / failed
            "total_files": len(filenames),
            "done_files": 0,
            "progress": 0.0,              # 0~100
            "current_file": "",
            "current_stage": "排队中",
            "filenames": list(filenames),
            "results": [],                # 逐文件结果
            "error": None,
            "created_at": time.time(),
            "finished_at": None,
        }
    return job_id


def _update_job(job_id: str, **kw):
    """线程安全地更新任务字段。"""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


def _public_job(job: dict) -> dict:
    """对外暴露的任务视图（拷一份 list，避免序列化时后台线程还在改）。"""
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total_files": job["total_files"],
        "done_files": job["done_files"],
        "progress": job["progress"],
        "current_file": job["current_file"],
        "current_stage": job["current_stage"],
        "filenames": list(job["filenames"]),
        "results": [dict(r) for r in job["results"]],
        "error": job["error"],
    }


def _make_progress_cb(job_id: str, file_idx: int, total: int):
    """
    为第 file_idx 个文件造一个进度回调：把 (stage, frac) 换算成"整批文件的总体百分比"。

    总体进度 = (已完成文件数 + 当前文件的完成比例) / 总文件数 * 100
    当前文件完成比例 = Σ(已完成的阶段权重) + 当前阶段权重 * 当前阶段frac
    """
    completed = {s: False for s, _ in STAGE_WEIGHTS}

    def cb(stage: str, frac: float):
        frac = max(0.0, min(1.0, float(frac)))
        file_frac = 0.0
        for s, w in STAGE_WEIGHTS:
            if s == stage:
                file_frac += w * frac
                break
            if completed.get(s):
                file_frac += w
        overall = (file_idx + file_frac) / max(1, total) * 100.0
        _update_job(job_id, progress=round(overall, 1), current_stage=stage)
        if frac >= 1.0:
            completed[stage] = True

    return cb


def _run_ingest_job(job_id: str, file_infos: List, strategy: str, tmpdir: str):
    """
    后台线程主体：逐个文件入库，按阶段更新任务进度。
    file_infos: [(原始文件名, 临时文件路径), ...]
    无论成功失败，最后都清理临时目录。
    """
    total = max(1, len(file_infos))
    try:
        _update_job(job_id, status="running", current_stage="加载模型", current_file="")
        # 首次会加载 bge 模型（较慢），用锁保证多任务并发时只建一次单例
        with _ENGINE_LOCK:
            ensure_engine()

        for idx, (orig_name, path) in enumerate(file_infos):
            _update_job(job_id, current_file=orig_name, current_stage="解析",
                        progress=round(idx / total * 100.0, 1))
            cb = _make_progress_cb(job_id, idx, total)
            try:
                info = ingest_file(
                    path, strategy=strategy,
                    pg=STATE["pg"], emb=STATE["emb"], vec=STATE["vec"],
                    progress_cb=cb,
                )
                result = {
                    "filename": orig_name, "ok": True,
                    "doc_id": info["doc_id"], "doc_name": info["doc_name"],
                    "chunks": info["chunks"], "char_count": info["char_count"],
                }
            except Exception as e:
                # 单文件失败不中断整批：记下错误继续下一个
                result = {"filename": orig_name, "ok": False,
                          "error": f"{type(e).__name__}: {e}"}
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["results"].append(result)
                    JOBS[job_id]["done_files"] = idx + 1
                    JOBS[job_id]["progress"] = round((idx + 1) / total * 100.0, 1)

        # 新文档入库后：不再把整库 BM25 置空（那会让下一次查询全库重建）。
        # 改为增量把本次成功入库文档的 chunk 加进现有索引 —— 块越多越省。
        with BM25_LOCK:
            bm = STATE["bm25"]
            if bm is not None:
                for r in JOBS[job_id].get("results", []):
                    if r.get("ok") and r.get("doc_id"):
                        rows, _ = STATE["pg"].get_chunks_by_doc(r["doc_id"], limit=100000)
                        bm.add_chunks(
                            [(row["chunk_id"], row.get("content") or "") for row in rows]
                        )
        _update_job(job_id, status="done", progress=100.0,
                    current_stage="完成", current_file="", finished_at=time.time())
    except Exception as e:
        _update_job(job_id, status="failed", current_stage="失败",
                    error=f"{type(e).__name__}: {e}", finished_at=time.time())
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：只做配置校验，缺凭据/缺语料直接 fail-fast（和 serve.py / ask.py 一致）
    errs = config.validate()
    if errs:
        raise RuntimeError("配置校验失败: " + "; ".join(errs))
    yield
    # 关闭：释放 PG 连接
    with contextlib.suppress(Exception):
        if STATE["pg"] is not None:
            STATE["pg"].close()


# ================================================================
# FastAPI app + CORS
# ================================================================
app = FastAPI(
    title="Self-Evolving RAG Agent — Frontend API",
    version="1.0",
    description="前端 React 页面要打的全 API：知识库管理 + 问答 + 可观测",
    lifespan=lifespan,
)

# CORS：开发环境放行所有 origin（Vite 默认 5173 端口）。
# 生产应改为白名单（如 ["https://your-frontend.example.com"]），
# 但本项目目前是本机开发演示，先放开。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # 用 * 时 credentials 必须 False（CORS 规范）
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# Pydantic 模型
# ================================================================
class AskRequest(BaseModel):
    """POST /api/ask 请求体"""
    question: str = Field(..., min_length=1)
    top_k: int = Field(config.RAG_TOP_K, ge=1, le=20)
    retrieval: str = Field("vector", description="vector | hybrid")
    self_heal: bool = Field(False, description="阶段5 拒答自愈")
    agent_heal: bool = Field(False, description="阶段6 缺口自愈（联网补库）")
    reflect: bool = Field(False, description="阶段8 Reflexion 事实核查")
    trace: bool = Field(False, description="阶段8 链路追踪落库")
    save: bool = Field(True, description="是否把问答写入 qa_log")


class RetrieveRequest(BaseModel):
    """POST /api/retrieve 请求体（检索测试：只召回，不调 LLM 生成）"""
    question: str = Field(..., min_length=1)
    top_k: int = Field(config.RAG_TOP_K, ge=1, le=50)
    retrieval: str = Field("vector", description="vector | hybrid")


class HealthCheckRequest(BaseModel):
    """POST /api/health-check 请求体"""
    dup_threshold: float = Field(0.95, ge=0.5, le=0.999, description="重复判定余弦阈值")
    skip_conflict: bool = Field(False, description="跳过矛盾检测（省 LLM 调用）")


# ================================================================
# 健康 / 总览
# ================================================================
@app.get("/api/health")
def health():
    """健康检查：返回状态 + 配置摘要 + 存储条数。PG 连不上也 200 + degraded。"""
    try:
        ensure_engine()
        cnt = STATE["pg"].count()
        pg_ok = True
    except Exception:
        cnt = {"documents": -1, "chunks": -1}
        pg_ok = False
    milvus_count = None
    with contextlib.suppress(Exception):
        if STATE["vec"] is not None:
            milvus_count = STATE["vec"].count()
    return {
        "status": "ok" if pg_ok else "degraded",
        "config": {
            "llm_backend": config.LLM_BACKEND,
            "llm_model": config.LLM_MODEL,
            "embed_model": config.EMBED_MODEL,
            "bm25_indexed": STATE["bm25"] is not None,
        },
        "pg": cnt,
        "milvus": milvus_count,
    }


@app.get("/api/stats")
def stats():
    """
    总览数据：知识库首页顶部统计卡用。
    staging_docs 是阶段6 写入、待人工审的影子库文档数（v1 先返回列表，前端可提示）。
    """
    try:
        ensure_engine()
        cnt = STATE["pg"].count()
        # 列出所有 staging 文档（数量一般不会多，O(n) 可接受）
        all_docs = STATE["pg"].list_documents()
        staging = [d for d in all_docs if d.get("status") == "staging"]
        milvus_count = None
        with contextlib.suppress(Exception):
            if STATE["vec"] is not None:
                milvus_count = STATE["vec"].count()
        return {
            "documents": cnt["documents"],
            "chunks": cnt["chunks"],
            "vectors": milvus_count,
            "staging_docs": len(staging),
            "staging_list": [
                {"doc_id": d["doc_id"], "doc_name": d["doc_name"],
                 "created_at": str(d.get("created_at"))}
                for d in staging
            ],
        }
    except Exception as e:
        raise HTTPException(503, f"获取总览失败：{e}")


@app.get("/api/llm-info")
def llm_info():
    """返回 LLM 后端/模型名（前端状态栏展示；不暴露 key）。"""
    return {
        "backend": config.LLM_BACKEND,
        "model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/api/bootstrap")
def bootstrap():
    """
    首屏一次性拿全初始数据（= /llm-info + /health + /stats + /documents 的合并）。

    【为什么需要它】
      前端首屏原本会**并发**打 4 个接口：
          Layout        → /api/llm-info + /api/health
          KnowledgeBase → /api/stats    + /api/documents
      冷启动时这 4 个请求会同时冲进 ensure_engine()，在"引擎还没建好"的窗口里
      反复触发重复构建。Milvus Lite 是单进程独占的嵌入式库，重复构造同一路径
      会直接抛 DataDirLockedError → 503（线上就是这么炸的：/api/health 200 OK、
      /api/documents 503 紧挨着出现）。
      ensure_engine 现在已有双检锁兜底，但"少打几个请求"本身也能让首屏更快、
      冷启动更稳 —— 合并成 1 个请求后，前端只需一次往返就能渲染首屏。

    【容错】
      引擎建不起来（PG 挂了等）也返回 200 + status=degraded，
      让前端能正常渲染"空态 + 降级提示"，而不是弹一个 503 错误页。
    """
    payload = {
        "llm": {
            "backend": config.LLM_BACKEND,
            "model": config.LLM_MODEL,
            "embed_model": config.EMBED_MODEL,
        },
        "health": {"status": "ok", "pg": None, "milvus": None},
        "stats": {"documents": 0, "chunks": 0, "vectors": None, "staging_docs": 0},
        "documents": [],
        "error": None,
    }

    try:
        ensure_engine()
    except Exception as e:
        payload["health"]["status"] = "degraded"
        payload["error"] = f"{type(e).__name__}: {e}"
        return payload

    try:
        cnt = STATE["pg"].count()
        docs = STATE["pg"].list_documents()
        # 时间字段转字符串，口径与 /api/documents 保持一致，前端可直接渲染
        for d in docs:
            if d.get("created_at"):
                d["created_at"] = str(d["created_at"])
        staging = [d for d in docs if d.get("status") == "staging"]

        milvus_count = None
        with contextlib.suppress(Exception):
            if STATE["vec"] is not None:
                milvus_count = STATE["vec"].count()

        payload["health"] = {"status": "ok", "pg": cnt, "milvus": milvus_count}
        payload["stats"] = {
            "documents": cnt["documents"],
            "chunks": cnt["chunks"],
            "vectors": milvus_count,
            "staging_docs": len(staging),
        }
        payload["documents"] = docs
    except Exception as e:
        # 引擎建起来了但查询失败（PG 中途断连等）：仍然 200，标记 degraded
        payload["health"]["status"] = "degraded"
        payload["error"] = f"{type(e).__name__}: {e}"

    return payload


# ================================================================
# 文档管理
# ================================================================
@app.get("/api/documents")
def list_documents():
    """列出全部文档（含每个文档的 chunk_count），按创建时间倒序。"""
    try:
        ensure_engine()
        docs = STATE["pg"].list_documents()
        # 时间字段转字符串，方便前端直接渲染
        for d in docs:
            if d.get("created_at"):
                d["created_at"] = str(d["created_at"])
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        raise HTTPException(503, f"列出文档失败：{e}")


@app.post("/api/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(..., description="要上传的文件列表（multipart field 名=files）"),
    strategy: str = Form("recursive", description="切片策略：recursive / fixed / structural"),
    wait: bool = Form(False, description="true=同步跑完再返回（调试用）；默认 false=立即返回 job_id"),
):
    """
    批量上传文件 → 转后台异步入库（解析 → 切片 → embedding → 写 PG + Milvus）。

    ★ 默认异步（wait=false）★
      接口只做两件事：① 把上传的字节落到临时目录；② 建一个任务并起后台线程。
      然后立刻返回 {"job_id": "xxx", "status": "pending", ...}，不阻塞。
      前端拿着 job_id 轮询 GET /api/jobs/{job_id} 就能看到实时进度：
          current_stage（解析/切片/向量化/入库）+ current_file + progress(0~100)
      进度权重：解析 5% / 切片 5% / 向量化 80% / 入库 10%（按单文件，再折算到整批）。

      为什么这么设计：入库几十个文件可能要几十秒到几分钟，
      同步阻塞 HTTP 会让前端等到超时且看不到任何中间进度。

    ★ 同步模式（wait=true）★
      给 curl / Swagger 调试用：直接在当前请求里跑完，返回最终 {"results": [...]}。
      前端不用这个（会被 Vite/浏览器超时截断）。

    注意：临时文件用**原始文件名**存于独立的临时目录，
    因为 ingest_file 用 basename(path) 生成 doc_name/doc_id，
    这样"同名文件重传 = 覆盖"的幂等语义才成立。
    """
    if strategy not in ("recursive", "fixed", "structural"):
        raise HTTPException(400, f"strategy 必须是 recursive / fixed / structural，当前 {strategy}")
    if not files:
        raise HTTPException(400, "files 不能为空")

    # ---- ① 落盘到独立临时目录（后台线程还要读，不能像旧版那样随用随删）----
    tmpdir = tempfile.mkdtemp(prefix="rag_upload_")
    file_infos: List[tuple] = []   # [(原始文件名, 临时文件路径), ...]
    seen_names: dict = {}          # 同名计数：同一批里出现重名时加后缀，避免互相覆盖
    try:
        for f in files:
            original_name = f.filename or "unnamed"
            safe_name = os.path.basename(original_name) or "unnamed"
            # 保留原扩展名：parse_file 按扩展名分发（md / pdf / docx / html ...）
            if safe_name in seen_names:
                seen_names[safe_name] += 1
                stem, ext = os.path.splitext(safe_name)
                safe_name = f"{stem}__{seen_names[safe_name]}{ext}"
            else:
                seen_names[safe_name] = 1
            tmp_path = os.path.join(tmpdir, safe_name)
            content = await f.read()
            with open(tmp_path, "wb") as out:
                out.write(content)
            file_infos.append((original_name, tmp_path))
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(400, f"接收上传文件失败：{type(e).__name__}: {e}")

    _prune_jobs()  # 顺手清理过期任务记录，避免内存无限增长

    # ---- ② 建任务 ----
    job_id = _create_job([name for name, _ in file_infos])

    # ---- ③a 同步模式：直接在当前请求里跑完（调试方便）----
    if wait:
        _run_ingest_job(job_id, file_infos, strategy, tmpdir)
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            snapshot = _public_job(job) if job else {}
        return {"results": snapshot.get("results", []), "total": len(file_infos), **snapshot}

    # ---- ③b 异步模式：起后台线程，立刻返回 job_id ----
    threading.Thread(
        target=_run_ingest_job,
        args=(job_id, file_infos, strategy, tmpdir),
        name=f"ingest-{job_id}",
        daemon=True,   # 主进程退出时不被这个线程拖住
    ).start()

    return {
        "job_id": job_id,
        "status": "pending",
        "total_files": len(file_infos),
        "filenames": [name for name, _ in file_infos],
        "strategy": strategy,
        "poll": f"/api/jobs/{job_id}",   # 前端轮询地址（方便调试直接照抄）
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    """
    查询异步入库任务的进度（前端进度条的数据源）。

    返回：
        {"job_id": "...", "status": "pending|running|done|failed",
         "total_files": 3, "done_files": 1, "progress": 62.5,
         "current_file": "论文.pdf", "current_stage": "向量化",
         "filenames": [...], "results": [{"filename":..., "ok":true, ...}],
         "error": null}
    """
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, f"任务不存在或已过期：{job_id}")
        return _public_job(job)


@app.get("/api/jobs")
def list_jobs():
    """列出当前进程内的全部入库任务（最近的在最后）。给调试/管理页用。"""
    with JOBS_LOCK:
        jobs = [_public_job(j) for j in JOBS.values()]
    # 按创建时间排序需要原始 job，这里用 job_id 保持稳定顺序即可
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str):
    """单个文档详情。不存在返回 404。"""
    try:
        ensure_engine()
        d = STATE["pg"].get_document(doc_id)
        if d is None:
            raise HTTPException(404, f"文档不存在：{doc_id}")
        if d.get("created_at"):
            d["created_at"] = str(d["created_at"])
        # 附 chunk_count
        rows, total = STATE["pg"].get_chunks_by_doc(doc_id, limit=1, offset=0)
        d["chunk_count"] = total
        return d
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"获取文档失败：{e}")


@app.get("/api/documents/{doc_id}/chunks")
def get_document_chunks(
    doc_id: str,
    limit: int = 50,
    offset: int = 0,
):
    """
    文档的切片列表（前端 Drawer 用），分页。
    limit/offset 走 query string，默认 50/0。
    """
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit 必须在 1~500 之间")
    if offset < 0:
        raise HTTPException(400, "offset 不能为负")
    try:
        ensure_engine()
        # 先确认文档存在，否则返回 404 而不是空列表（更明确）
        d = STATE["pg"].get_document(doc_id)
        if d is None:
            raise HTTPException(404, f"文档不存在：{doc_id}")
        rows, total = STATE["pg"].get_chunks_by_doc(doc_id, limit=limit, offset=offset)
        # 时间/JSON 字段转字符串
        for r in rows:
            if isinstance(r.get("meta"), (dict, list)):
                r["meta"] = str(r["meta"])
        return {"doc_id": doc_id, "total": total, "limit": limit, "offset": offset, "chunks": rows}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"获取切片失败：{e}")


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    """
    删除文档：先删 Milvus 向量（按 doc_id 过滤），再删 PG（级联删 chunks）。
    PG 是 source of truth；向量删失败也回 200 + warning（孤儿向量下次重建会被清掉）。
    """
    try:
        ensure_engine()
        d = STATE["pg"].get_document(doc_id)
        if d is None:
            raise HTTPException(404, f"文档不存在：{doc_id}")

        # 0) 先记下该文档的 chunk_id（必须在删 PG 之前取，删完就查不到了）
        _rows, _ = STATE["pg"].get_chunks_by_doc(doc_id, limit=100000)
        _chunk_ids = [row["chunk_id"] for row in _rows]

        # 1) 先删向量（不删的话，孤儿向量永远搜不到但占内存）
        with contextlib.suppress(Exception):
            if STATE["vec"] is not None:
                STATE["vec"].delete_by_doc(doc_id)

        # 2) 删 PG（级联删 chunks）
        n = STATE["pg"].delete_document(doc_id)

        # 3) BM25 增量移除该文档的所有 chunk（不再整库置空重建）
        with BM25_LOCK:
            bm = STATE["bm25"]
            if bm is not None:
                bm.remove_chunks(_chunk_ids)

        return {"doc_id": doc_id, "deleted_chunks": n, "ok": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"删除文档失败：{e}")


# ================================================================
# 问答
# ================================================================
@app.post("/api/ask")
def ask(req: AskRequest):
    """
    问答接口：和 serve.py 一样走 generate_answer，
    多了 agent_heal（阶段6）/ reflect / trace（阶段8）开关。
    """
    if req.retrieval not in ("vector", "hybrid"):
        raise HTTPException(400, "retrieval 必须是 vector 或 hybrid")
    try:
        ensure_engine()
        if req.agent_heal:
            # 阶段 6 缺口自愈走 run_self_heal
            return run_self_heal(
                req.question,
                top_k=req.top_k,
                retrieval=req.retrieval,
                pg=STATE["pg"], emb=STATE["emb"], vec=STATE["vec"],
                backend_llm=STATE["llm"], bm25=STATE["bm25"], save=req.save, trace=req.trace,
                reranker=STATE["reranker"],
            )
        return generate_answer(
            req.question,
            top_k=req.top_k,
            retrieval=req.retrieval,
            self_heal=req.self_heal,
            save=req.save,
            reflect=req.reflect,
            trace=req.trace,
            emb=STATE["emb"], vec=STATE["vec"], pg=STATE["pg"],
            backend_llm=STATE["llm"], bm25=STATE["bm25"],
            reranker=STATE["reranker"],
        )
    except HTTPException:
        raise
    except Exception as e:
        # 503：LLM 调不通 / PG 断连 / DashScope key 错误等
        raise HTTPException(503, f"服务内部错误：{e}")


# ================================================================
# 检索测试（RAGFlow "Retrieval testing"）：只召回，不调 LLM
# ================================================================
@app.post("/api/retrieve")
def retrieve_only(req: RetrieveRequest):
    """
    检索测试：问题 → 召回 top_k 块（带分数），**不调用 LLM 生成**。
    用于 RAGFlow 风格的"检索测试"页 —— 调不同 retrieval / top_k，看召回块变没变，
    比起问答页更能定位"检索质量"问题（问答页只能看到最终答案好/坏）。
    """
    if req.retrieval not in ("vector", "hybrid"):
        raise HTTPException(400, "retrieval 必须是 vector 或 hybrid")
    try:
        ensure_engine()
        if req.retrieval == "hybrid":
            # hybrid 需要 BM25；若索引没建好就退化纯向量（和 rag.py 一致）
            if STATE["bm25"] is not None:
                hits = hybrid_retrieve(
                    req.question, STATE["emb"], STATE["vec"], STATE["pg"], STATE["bm25"],
                    top_k=req.top_k, reranker=STATE["reranker"],
                )
            else:
                hits = retrieve(req.question, STATE["emb"], STATE["vec"], STATE["pg"], top_k=req.top_k)
        else:
            hits = retrieve(req.question, STATE["emb"], STATE["vec"], STATE["pg"], top_k=req.top_k)
        # [(score, chunk_dict), ...] → 可直接 JSON 的结构（chunk_dict 里可能含 datetime，需转 str）
        items = []
        for score, c in hits:
            item = {
                "score": round(float(score), 6),
                "chunk_id": c.get("chunk_id"),
                "doc_name": c.get("doc_name"),
                "heading": c.get("heading"),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
                "content": c.get("content"),
            }
            items.append(item)
        return {"question": req.question, "retrieval": req.retrieval, "top_k": req.top_k,
                "count": len(items), "chunks": items}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"检索失败：{e}")


# ================================================================
# 阶段 7：知识库体检
# ================================================================
@app.post("/api/health-check")
def health_check(req: HealthCheckRequest):
    """
    跑阶段 7 四个检测器（过时 / 矛盾 / 僵尸 / 重复），返回健康报告。
    - 只读不写：报告只是建议，删除/修改动作要人工执行。
    - llm 仅矛盾检测需要；skip_conflict=True 时不传 llm，省一次 LLM 调用。
    """
    try:
        ensure_engine()
        llm = None if req.skip_conflict else STATE["llm"]
        report = run_health_check(
            pg=STATE["pg"], vec=STATE["vec"], emb=STATE["emb"],
            llm=llm, dup_threshold=req.dup_threshold, report_path=None,
        )
        # HealthIssue 是 dataclass，转成 dict 才能 JSON 序列化
        issues = [dataclasses.asdict(it) for it in report["issues"]]
        # staging_docs 里 datetime 转字符串
        staging = []
        for d in report.get("staging_docs", []):
            staging.append({"doc_id": d.get("doc_id"), "doc_name": d.get("doc_name"),
                            "created_at": str(d.get("created_at"))})
        return {
            "summary": report["summary"],
            "markdown": report["markdown"],
            "skipped": report["skipped"],
            "staging_docs": staging,
            "issues": issues,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f"体检失败：{e}")


# ================================================================
# 可观测：trace 列表
# ================================================================
@app.get("/api/traces")
def list_traces(limit: int = 50):
    """最近的 trace 记录（前端可观测页面用，v1 暂不渲染，留接口）。"""
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit 必须在 1~500 之间")
    try:
        ensure_engine()
        traces = STATE["pg"].get_traces(limit=limit)
        for t in traces:
            if t.get("created_at"):
                t["created_at"] = str(t["created_at"])
        return {"traces": traces, "total": len(traces)}
    except Exception as e:
        raise HTTPException(503, f"获取 trace 失败：{e}")


# ================================================================
# 入口
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="前端要打的全 API：知识库管理 + 问答 + 可观测")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认只本机）")
    ap.add_argument("--port", type=int, default=8000, help="监听端口")
    args = ap.parse_args()
    import uvicorn
    # reload=False 避免开发热重载把单例 STATE 搞乱
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
