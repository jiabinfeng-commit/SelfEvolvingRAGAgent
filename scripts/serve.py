#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 4 服务化：把 RAG 问答包成 HTTP 服务（FastAPI）

为什么服务化？（不是为服务化而服务化，是前面几个阶段的"使用方式"逼出来的）
-----------------------------------------------------------------------
阶段 1~3 的 ask.py 只能在命令行跑，每次执行都要：
    - 重新加载 bge embedding 模型（~1s）
    - 重新连 PostgreSQL、重新连 Milvus Lite
    - 重新实例化 LLM 客户端
既慢，又无法被前端 / 其他微服务 / 定时任务调用。
服务化后：模型 + 连接只在**启动时加载一次**，常驻内存，多个请求共享，
对外暴露标准 HTTP 接口 —— streamlit 页面、其他业务系统都能直接 curl / 调 SDK。

接口一览：
- GET  /health   健康检查
      返回：服务状态 + 配置摘要 + PG 文档/块数 + Milvus 条数。
      PG 连不上也返回 200（status=degraded），方便探活脚本区分"活着但存储异常"。
- POST /ask      问答
      body: {"question": "...", "top_k": 5,
             "retrieval": "vector"|"hybrid", "self_heal": false, "save": true}
      返回：generate_answer 的结构化结果（答案 + 召回块列表 + 耗时 + 是否自愈）

启动：
    python scripts/serve.py                 # 默认 http://127.0.0.1:8000
    python scripts/serve.py --host 0.0.0.0 --port 8080
启动后浏览器打开 http://127.0.0.1:8000/docs  —— FastAPI 自带 Swagger 交互式文档，
可以直接在网页上填参数试 /ask，不用自己写 curl。

设计要点（和项目一贯风格一致）：
- 进程级单例 STATE：启动时只做配置校验 + 加载 embedding；PG / Vec / LLM / BM25
  在**首次请求时惰性建立并缓存**，避免启动即硬依赖存储（存储临时不可达也能先起服务）。
- BM25 索引首次混合检索时惰性建（494 块虽小，但纯向量场景用不到，不浪费）；
  建好后缓存复用。注意：入库新文档后索引会过期，需重启服务刷新（demo 阶段可接受）。
- 不引入额外配置：LLM 后端、模型、检索默认都读 core/config.py（阶段 3 起的纪律：
  后续阶段不改模型配置）。
"""
import os
import sys
import argparse
import contextlib
import threading
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.bm25 import BM25
from core.rag import generate_answer


# ---------- 进程级单例：整个服务生命周期只建一次，全程复用 ----------
STATE: dict = {
    "emb": None,       # Embedder 实例
    "pg": None,        # PGStore 实例
    "vec": None,       # VecStore 实例
    "llm": None,       # LLM 客户端实例
    "bm25": None,      # BM25 索引（首次 hybrid 时建）
}

# 引擎构建锁：见下面 ensure_engine 的说明。
# 不加锁时，两个并发首请求会各建一份 VecStore，而 Milvus Lite 单进程独占同一路径，
# 第二次构造直接抛 DataDirLockedError（同一个进程内也会冲突）。
ENGINE_LOCK = threading.Lock()


def ensure_engine():
    """
    惰性建立引擎依赖（首次请求时调用）。已建立的复用，不重复建。

    为什么惰性而不是在启动时全建？启动时若 PG 临时不可达，服务会直接起不来；
    惰性建则"存储挂了也能先起服务"，/health 返回 degraded，/ask 在真正需要时再报错，
    对本地开发更友好。生产若要严格启动即校验，可在 lifespan 里直接调用本函数。

    【并发安全】本文件的路由是同步 `def`，FastAPI 会丢到线程池执行，
    多个请求可能真的同时进入本函数 → 必须用双检锁挡住 check-then-act 竞态，
    否则会出现"两个线程都判定 STATE['vec'] is None，各建一份 VecStore"
    → Milvus Lite 抛 DataDirLockedError（flock 按打开文件描述计，同进程也冲突）。
    详见 scripts/api.py 里 ensure_engine 的完整注释。
    """
    # 快路径：五个槽位都有值 → 直接返回，不进锁（稳态几乎零开销）。
    # bm25 建失败会被置 None，那种情况落进锁里重试，语义与加锁前一致。
    if all(STATE[k] is not None for k in ("emb", "pg", "vec", "llm", "bm25")):
        return

    with ENGINE_LOCK:
        # 慢路径：拿到锁后再判断一次（可能已被先到的线程建好了）
        if STATE["emb"] is None:
            STATE["emb"] = get_embedder("bge")
        if STATE["pg"] is None:
            STATE["pg"] = PGStore()
        if STATE["vec"] is None:
            STATE["vec"] = VecStore(dim=STATE["emb"].dim)
        if STATE["llm"] is None:
            STATE["llm"] = llm_mod.get_llm(config.LLM_BACKEND)
        if STATE["bm25"] is None:
            # BM25 依赖 PG 全量 chunk，建一次缓存；建失败不影响纯向量检索
            try:
                all_chunks = STATE["pg"].get_all_chunks()
                STATE["bm25"] = BM25().build(
                    [(c["chunk_id"], (c.get("content") or "")) for c in all_chunks]
                )
            except Exception:
                STATE["bm25"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：只做配置校验（缺密码/缺语料直接 fail-fast，和 ask.py 一致）。
    # 重资源（模型/连接）留到首次请求惰性建，避免启动即硬依赖存储。
    errs = config.validate()
    if errs:
        raise RuntimeError("配置校验失败: " + "; ".join(errs))
    yield
    # 关闭时释放 PG 连接（其余对象随进程退出自动回收）
    with contextlib.suppress(Exception):
        if STATE["pg"] is not None:
            STATE["pg"].close()


app = FastAPI(
    title="Self-Evolving RAG Agent API",
    version="1.0",
    description="阶段4 服务化：把 RAG 问答暴露成 HTTP 接口（GET /health、POST /ask）",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    """POST /ask 的请求体（pydantic 自动做参数校验 + 生成文档）"""
    question: str = Field(..., min_length=1, description="要问的问题，不能为空")
    top_k: int = Field(config.RAG_TOP_K, ge=1, le=20, description="召回块数")
    retrieval: str = Field("vector", description="vector=纯向量；hybrid=向量+BM25 融合")
    self_heal: bool = Field(False, description="是否启用阶段5 Agent 自愈")
    save: bool = Field(True, description="是否把问答写入 PG qa_log")


@app.get("/health")
def health():
    """
    健康检查。返回配置摘要 + 存储条数。
    PG 连不上时 status=degraded（仍返回 200，方便探活区分"活着但存储异常"）。
    """
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
            "retrieval_default": "vector",
            "bm25_indexed": STATE["bm25"] is not None,
        },
        "pg": cnt,
        "milvus": milvus_count,
    }


@app.post("/ask")
def ask(req: AskRequest):
    """
    问答接口。把请求转给 core.rag.generate_answer（和 CLI / UI 同一份逻辑）。
    """
    if req.retrieval not in ("vector", "hybrid"):
        raise HTTPException(400, "retrieval 必须是 vector 或 hybrid")
    try:
        ensure_engine()
        return generate_answer(
            req.question,
            top_k=req.top_k,
            retrieval=req.retrieval,
            self_heal=req.self_heal,
            save=req.save,
            emb=STATE["emb"],
            vec=STATE["vec"],
            pg=STATE["pg"],
            backend_llm=STATE["llm"],
            bm25=STATE["bm25"],
        )
    except Exception as e:
        # LLM 调不通 / PG 断连 / DashScope key 错误等：返回 503，前端好处理
        # （用 Exception 而非 RuntimeError，是因为 PG 连不上抛的是 psycopg2 的原生异常，
        #   不在 RuntimeError 范围里；统一兜住，避免裸 500 把内部栈甩给调用方）
        raise HTTPException(503, f"服务内部错误：{e}")


def main():
    """命令行入口：python scripts/serve.py [--host] [--port]"""
    ap = argparse.ArgumentParser(description="阶段4 服务化：RAG HTTP 服务")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认只本机）")
    ap.add_argument("--port", type=int, default=8000, help="监听端口")
    args = ap.parse_args()
    import uvicorn
    # 用 uvicorn 拉起 FastAPI app；reload=False 避免开发热重载把单例 STATE 搞乱
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
