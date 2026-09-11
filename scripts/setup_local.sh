#!/usr/bin/env bash
# ============================================================================
# ⚠️ 已非默认路径（仅云主机复用场景才需要）
#
# 本项目的默认开发方式是「本地 Docker 起 PostgreSQL + 重新入库」，一条命令：
#   bash scripts/setup_new_machine.sh
# 新人/换机器请直接用上面那个，不需要任何云主机。
#
# 本脚本是「作者本人」专用：本地开发想**直接复用自己云上的 PG + 向量库**，
# 而不是在本地重新入库。需要你：
#   ① 有自己的云主机（RAG_CLOUD_HOST）
#   ② 接受本地 5432 被隧道占用 / 或停掉本地 Docker PG
#
# ⚠️ 若你本地已经在跑 Docker 的 rag-pg（默认 5432），本脚本的隧道会和它冲突——
#    那种情况请用 setup_new_machine.sh，别用本脚本。
#
# —— 原设计说明（历史参考）——
# 思路：数据服务不装在本地，全部复用云上的。
#   - PostgreSQL  → SSH 隧道连云端（云上 PG 只监听 127.0.0.1，公网连不上，隧道是唯一解）
#   - 向量库      → 云上的 milvus.db 整个目录 scp 回来（就 2.6MB）
#   - 模型        → 从 ModelScope 下载到本地 models/
#
# 用法：
#   RAG_CLOUD_HOST=root@你的服务器IP bash scripts/setup_local.sh
#   RAG_CLOUD_HOST=root@1.2.3.4 RAG_LOCAL_PG_PORT=5433 bash scripts/setup_local.sh
# ============================================================================
set -euo pipefail

CLOUD_HOST="${RAG_CLOUD_HOST:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 必须显式指定云主机：故意不给默认值，否则别人 clone 下来会去连脚本作者的服务器。
if [ -z "$CLOUD_HOST" ]; then
    cat <<'TIP'
✗ 未指定云主机地址（RAG_CLOUD_HOST）

  用法：
      RAG_CLOUD_HOST=root@你的服务器IP bash scripts/setup_local.sh

  ⚠ 没有云主机？这个脚本不适合你。用全本地的那套：
      bash scripts/setup_new_machine.sh
    或看 docs/14-拉取代码后如何跑起来.md
TIP
    exit 1
fi

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
# 云上 PG 密码故意不写死在脚本里（否则会随 git 提交泄露），请运行本脚本后手动填入 .env
PG_PASSWORD=

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
