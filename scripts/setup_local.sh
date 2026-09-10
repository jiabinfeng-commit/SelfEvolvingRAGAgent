#!/usr/bin/env bash
# ============================================================================
# 本地开发环境一键搭建
#
# 思路：数据服务不装在本地，全部复用云上的。
#   - PostgreSQL  → SSH 隧道连云端（云上 PG 只监听 127.0.0.1，公网连不上，隧道是唯一解）
#   - 向量库      → 云上的 milvus.db 整个目录 scp 回来（就 2.6MB）
#   - 模型        → 从 ModelScope 下载到本地 models/
#
# 本地真正要装的只有 Python 依赖 + 模型，约 400MB。
#
# 用法：
#   bash scripts/setup_local.sh
#   RAG_CLOUD_HOST=root@1.2.3.4 bash scripts/setup_local.sh   # 换机器
# ============================================================================
set -euo pipefail

CLOUD_HOST="${RAG_CLOUD_HOST:-root@120.26.22.166}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "项目根: $ROOT"
echo "云主机: $CLOUD_HOST"
echo ""

# ---------- 1. 虚拟环境 ----------
echo "==> [1/6] 创建虚拟环境 .venv"
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "    已创建"
else
    echo "    已存在，跳过"
fi

# ---------- 2. 依赖 ----------
echo "==> [2/6] 安装 Python 依赖（torch 首次较慢，约 200MB）"
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q \
    pymilvus \
    milvus-lite \
    psycopg2-binary \
    python-dotenv \
    sentence-transformers
echo "    依赖就绪"

# ---------- 3. 模型 ----------
echo "==> [3/6] 下载 embedding 模型（从 ModelScope，约 183MB）"
.venv/bin/python scripts/fetch_model.py 2>&1 | tail -3

# ---------- 4. 向量库 ----------
echo "==> [4/6] 从云上同步向量库 milvus.db"
mkdir -p data
if [ -d data/milvus.db ]; then
    echo "    已存在，先备份为 data/milvus.db.bak"
    rm -rf data/milvus.db.bak
    mv data/milvus.db data/milvus.db.bak
fi
scp -q -r "$CLOUD_HOST:/opt/rag-agent/data/milvus.db" data/
echo "    向量库已就位: data/milvus.db"

# ---------- 5. .env ----------
echo "==> [5/6] 写入本地 .env（隧道模式）"
cat > .env <<'ENVFILE'
# 本地开发：PostgreSQL 走 SSH 隧道连云上，向量库用本地拷贝的 milvus.db
# 每次开发前先开隧道： bash scripts/tunnel_pg.sh

PG_HOST=localhost
PG_PORT=5432
PG_DB=rag
PG_USER=rag
# 这里必须是云上 PG 的真实密码（隧道过去用的是云上的账号体系）
PG_PASSWORD=rag_dev_2026

# 向量库：本地文件，由 setup_local.sh 从云上同步
MILVUS_PATH=
MILVUS_COLLECTION=chunks

# 模型：本地 models/ 目录，由 setup_local.sh 下载
MODEL_ROOT=
EMBED_MODEL=
EMBED_DIM=512
EMBED_BATCH_SIZE=32

# 路径留空 = 用项目根作为 BASE_DIR
RAG_BASE_DIR=
RAG_CORPUS_DIR=
ENVFILE
chmod 600 .env
echo "    .env 已写入（权限 600）"

# ---------- 6. 自检 ----------
echo "==> [6/6] 自检"
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from core import config
print(config.summary())
errs = config.validate()
if errs:
    print()
    for e in errs:
        print('  ✗', e)
    print()
    print('  若是 PG_PASSWORD 相关：云上密码不对')
    print('  若是语料目录：RAG_CORPUS_DIR 没指到 corpus/clean')
    sys.exit(1)
print()
print('  配置检查通过 ✓')
"

echo ""
echo "=============================================="
echo " 本地环境搭建完成"
echo "=============================================="
echo ""
echo " 接下来两步："
echo "   1. 开一个终端跑： bash scripts/tunnel_pg.sh   （保持开着）"
echo "   2. 另一个终端跑： .venv/bin/python scripts/search.py 'FastAPI 怎么做依赖注入'"
echo ""
