# -*- coding: utf-8 -*-
"""
全局配置

设计原则：
- 所有路径/凭据都可用环境变量覆盖，方便本地开发与云上部署切换
- 不把密码硬编码在业务代码里（这里填的是云上开发库的默认值）
"""
import os

# ---------- 项目根目录（云上为 /opt/rag-agent/app） ----------
BASE_DIR = os.getenv("RAG_BASE_DIR", "/opt/rag-agent")

# ---------- 语料 ----------
CORPUS_DIR = os.getenv("RAG_CORPUS_DIR", os.path.join(BASE_DIR, "corpus", "clean"))

# ---------- PostgreSQL（存 chunk 原文与元数据） ----------
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "rag")
PG_USER = os.getenv("PG_USER", "rag")
PG_PASSWORD = os.getenv("PG_PASSWORD", "rag_dev_2026")

# ---------- Milvus Lite（只存向量） ----------
MILVUS_PATH = os.getenv("MILVUS_PATH", os.path.join(BASE_DIR, "data", "milvus.db"))
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "chunks")

# ---------- Embedding ----------
# bge-small-zh-v1.5：512 维，约 100MB，中文效果好且轻量
#
# 国内机器访问 huggingface.co 不通，模型由 scripts/fetch_model.py 从
# ModelScope 预先下载到本地。这里优先用本地目录，不存在时再退回 HF 官方 id
# （本地开发机有代理时可以走网络）。
MODEL_ROOT = os.getenv("MODEL_ROOT", os.path.join(BASE_DIR, "models"))
_LOCAL_EMBED = os.path.join(MODEL_ROOT, "BAAI__bge-small-zh-v1.5")
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    _LOCAL_EMBED if os.path.isdir(_LOCAL_EMBED) else "BAAI/bge-small-zh-v1.5",
)
EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# bge 系列做检索时，查询侧建议加这句前缀（官方推荐，能小幅提升召回）
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：")


def summary() -> str:
    """打印配置摘要（隐藏密码）"""
    return (
        f"语料目录    : {CORPUS_DIR}\n"
        f"PostgreSQL  : {PG_USER}@{PG_HOST}:{PG_PORT}/{PG_DB}\n"
        f"Milvus Lite : {MILVUS_PATH} (collection={COLLECTION_NAME})\n"
        f"Embedding   : {EMBED_MODEL} ({EMBED_DIM} 维)"
    )


if __name__ == "__main__":
    print(summary())
