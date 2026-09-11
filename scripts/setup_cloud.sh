#!/usr/bin/env bash
# ============================================================================
# setup_cloud.sh —— 在云服务器（Ubuntu/Debian ECS）上部署 / 更新本服务
#
# ★ 和 setup_new_machine.sh 的本质区别 ★
#   setup_new_machine.sh  面向"干净的本地机器"：建 venv、下模型、**用 Docker 起 PG**
#   本脚本                面向"云上已有的部署"：**复用一切已存在的东西**
#
# 你的 ECS 实测现状：
#   /opt/rag-agent/app             git checkout（代码）
#   /opt/rag-agent/venv            Python 3.10 venv（已有 torch+cpu、pymilvus、sentence-transformers）
#   /opt/rag-agent/models          184MB embedding 模型（已下好）
#   /opt/rag-agent/data/milvus.db  向量库（已有数据）
#   PostgreSQL                     ★ 原生安装、已在跑（不是 Docker！）
#   docker / node                  ★ 都没装
#
# 所以云上**不需要** docker、**不需要**下模型、**不需要**起 PG、**不应该**再建一个 venv。
# 本脚本只做"云上真正缺的那几件事"（幂等，可反复跑）：
#   [1/5] 环境体检（Python / venv / PG / 端口 / 磁盘）
#   [2/5] 补装 Python 依赖（云上缺 fastapi / uvicorn / python-multipart）
#   [3/5] 校验 .env（★ 云上最容易漏的一步：LLM 配置）
#   [4/5] 校验数据（语料 / 模型 / 向量库 / PG 条数）
#   [5/5] 前端（可选）：装 Node + 构建 dist
#   最后   打印运行方式 + 安全组 + systemd 建议（不自动改系统服务）
#
# 用法：
#   bash scripts/setup_cloud.sh                  # 只做后端（推荐先这样）
#   FRONTEND=1 bash scripts/setup_cloud.sh       # 顺带装 Node + 构建前端
#   RAG_APP_DIR=/opt/rag-agent bash scripts/setup_cloud.sh
#   SYSTEMD=1 bash scripts/setup_cloud.sh        # 额外写 systemd unit（常驻）
#   SKIP_APT=1 bash scripts/setup_cloud.sh       # 不用 apt 装系统包
# ============================================================================
set -euo pipefail

# ---------- 可调参数 ----------
RAG_APP_DIR="${RAG_APP_DIR:-/opt/rag-agent}"     # 云上部署根（config.py 也认这个路径）
CODE_DIR="${CODE_DIR:-$RAG_APP_DIR/app}"         # 代码目录
VENV_DIR="${VENV_DIR:-$RAG_APP_DIR/venv}"        # 虚拟环境（复用，不新建）
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
API_PORT="${API_PORT:-8000}"
FE_PORT="${FE_PORT:-5173}"
FRONTEND="${FRONTEND:-0}"
SYSTEMD="${SYSTEMD:-0}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_EXTRAS="${SKIP_EXTRAS:-0}"     # 1 = 不装 pdfplumber / python-docx
NODE_MAJOR="${NODE_MAJOR:-20}"

echo "============================================================"
echo " Self-Evolving RAG Agent —— 云服务器部署"
echo "============================================================"
echo " 部署根   : $RAG_APP_DIR"
echo " 代码目录 : $CODE_DIR"
echo " venv     : $VENV_DIR"
echo " pip 源   : $PIP_INDEX"
echo " 前端     : $([ "$FRONTEND" = "1" ] && echo '装 Node + 构建' || echo '跳过（FRONTEND=1 开启）')"
echo ""

# ============================================================================
# [1/5] 环境体检
# ============================================================================
echo "==> [1/5] 环境体检"

if [ ! -d "$CODE_DIR" ]; then
    echo "  ✗ 代码目录不存在：$CODE_DIR"
    echo "    云上应该已经有一份部署。若这是全新的 ECS，请先 clone："
    echo "      git clone <仓库地址> $CODE_DIR"
    exit 1
fi
echo "  ✓ 代码目录存在"
cd "$CODE_DIR"

# Python 版本（云上是 3.10，本项目在 3.10 上实测可用）
SYS_PY="$(command -v python3 || true)"
[ -n "$SYS_PY" ] || { echo "  ✗ 没找到 python3"; exit 1; }
echo "  ✓ 系统 Python $("$SYS_PY" -V 2>&1 | awk '{print $2}')"

if [ -x "$VENV_DIR/bin/python" ]; then
    echo "  ✓ venv 存在（复用，不新建）: $("$VENV_DIR/bin/python" -V 2>&1)"
    VENV_PY="$VENV_DIR/bin/python"
else
    echo "  ⚠ venv 不存在，创建到 $VENV_DIR"
    "$SYS_PY" -m venv "$VENV_DIR"
    VENV_PY="$VENV_DIR/bin/python"
fi

# PostgreSQL（云上是原生服务，不是容器）
PG_OK=0
if command -v pg_isready >/dev/null 2>&1 && pg_isready >/dev/null 2>&1; then
    PG_OK=1
    echo "  ✓ PostgreSQL 在跑（原生服务，非 Docker）"
elif systemctl is-active --quiet postgresql 2>/dev/null; then
    PG_OK=1
    echo "  ✓ PostgreSQL 服务 active"
else
    echo "  ⚠ PostgreSQL 似乎没在跑。云上是原生安装的，检查/启动："
    echo "      systemctl status postgresql"
    echo "      systemctl start postgresql"
fi

# 端口
if command -v lsof >/dev/null 2>&1; then
    for p in "$API_PORT" "$FE_PORT"; do
        if lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; then
            echo "  ⚠ 端口 $p 已被占用（若已跑着一个实例，先停掉再起新的）"
        fi
    done
fi

echo "  --- 资源 ---"
free -h 2>/dev/null | awk 'NR<=2 {print "    " $0}'
DISK_AVAIL="$(df -h / | awk 'NR==2 {print $4}')"
DISK_SIZE="$(df -h / | awk 'NR==2 {print $2}')"
DISK_USED="$(df -h / | awk 'NR==2 {print $5}')"
echo "    磁盘: 可用 $DISK_AVAIL / 共 $DISK_SIZE（已用 $DISK_USED）"
echo "    CPU : $(nproc) 核"

# ============================================================================
# [2/5] 补装 Python 依赖
# ============================================================================
echo ""
echo "==> [2/5] 检查 / 补装 Python 依赖"

# 云上实测缺的是这些：fastapi / uvicorn / python-multipart（★ 起 API 必需）
#                        + pdfplumber / python-docx（上传 PDF/DOCX 才需要）
#                        + streamlit（老 UI 页面才需要，本脚本不装）
# 其余（torch+cpu / pymilvus / sentence-transformers / psycopg2 …）云上已经有了
export SKIP_EXTRAS="${SKIP_EXTRAS:-0}"
MISSING="$("$VENV_PY" - <<'PY'
import importlib.util as u
import os

need = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "multipart": "python-multipart",     # UploadFile/Form 必需
    "pymilvus": "pymilvus",
    "psycopg2": "psycopg2-binary",
    "dotenv": "python-dotenv",
    "numpy": "numpy",
    "sentence_transformers": "sentence-transformers",
}
# 前端上传组件 accept 里有 .pdf/.docx，不装这两个的话上传这两种格式会报错。
# 不需要可以 SKIP_EXTRAS=1 跳过（省约 15MB）。
if os.environ.get("SKIP_EXTRAS") != "1":
    need["pdfplumber"] = "pdfplumber"
    need["docx"] = "python-docx"

for mod, pkg in need.items():
    if u.find_spec(mod) is None:
        print(pkg)
PY
)"

if [ -z "$MISSING" ]; then
    echo "  ✓ 依赖齐全，无需安装"
else
    echo "  缺这些：$(echo "$MISSING" | tr '\n' ' ')"
    echo "  安装中（走国内源）..."
    # shellcheck disable=SC2086
    "$VENV_PY" -m pip install -q --upgrade pip
    # shellcheck disable=SC2086
    "$VENV_PY" -m pip install -q $MISSING -i "$PIP_INDEX"
    echo "  ✓ 安装完成"
fi

# ============================================================================
# [3/5] 校验 .env（云上最容易漏的一步）
# ============================================================================
echo ""
echo "==> [3/5] 校验 .env"
if [ ! -f .env ]; then
    echo "  ⚠ 没有 .env，从模板生成一份"
    cp .env.example .env
    chmod 600 .env
    echo "    → 已生成，但**必须自己填** PG_PASSWORD 和 LLM 配置（见下）"
fi

VALIDATE_OUT="$("$VENV_PY" - <<'PY' 2>&1
import sys
sys.path.insert(0, '.')
from core import config
print(config.summary())
print('---VALIDATE---')
errs = config.validate()
for e in errs:
    print(e)
PY
)"
echo "$VALIDATE_OUT" | sed 's/^/  /' | sed '/---VALIDATE---/,$d'

ERRS="$(echo "$VALIDATE_OUT" | awk '/---VALIDATE---/{f=1;next} f')"
if [ -n "$ERRS" ]; then
    echo ""
    echo "  ✗ 配置校验失败（后端会直接起不来，因为 lifespan 里 fail-fast）："
    echo "$ERRS" | sed 's/^/     · /'
    echo ""
    case "$ERRS" in
      *OPENAI_API_KEY*)
        echo "  ★ 这是云上最常见的坑：.env 里 LLM_BACKEND=openai 但没有 OPENAI_API_KEY。"
        echo "    往 $CODE_DIR/.env 补上这四行（和本地 .env 里的一致）："
        echo "        LLM_BACKEND=openai"
        echo "        LLM_MODEL=qwen3.7-max"
        echo "        OPENAI_API_KEY=sk-你的key"
        echo "        OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1"
        echo "    或改用本地 Ollama：LLM_BACKEND=ollama + LLM_MODEL=qwen2.5:7b"
        ;;
      *PG_PASSWORD*)
        echo "  ★ PG_PASSWORD 没填。云上的 PG 是原生安装的，密码在当初创建用户时定的那个。"
        ;;
    esac
    exit 1
fi
echo "  ✓ 配置校验通过"

# ============================================================================
# [4/5] 校验数据
# ============================================================================
echo ""
echo "==> [4/5] 校验数据"
"$VENV_PY" - <<'PY' | sed 's/^/  /'
import os, sys
sys.path.insert(0, '.')
from core import config

def line(label, path, isdir=False):
    exists = os.path.isdir(path) if isdir else os.path.exists(path)
    print(f"{'✓' if exists else '✗'} {label}: {path}")

line('语料目录', config.CORPUS_DIR, isdir=True)
if os.path.isdir(config.CORPUS_DIR):
    n = len([f for f in os.listdir(config.CORPUS_DIR) if f.endswith('.md')])
    print(f"   └ 文档数：{n} 篇")
line('向量库  ', config.MILVUS_PATH)
line('模型目录', os.path.join(config.MODEL_ROOT, 'BAAI__bge-small-zh-v1.5'), isdir=True)
print(f"embedding = {config.EMBED_MODEL} ({config.EMBED_DIM} 维)")
PY

if [ "$PG_OK" = "1" ]; then
    echo "  --- PG 现有数据 ---"
    "$VENV_PY" - <<'PY' 2>&1 | sed 's/^/  /' || true
import sys
sys.path.insert(0, '.')
from core.storage.pg_store import PGStore
try:
    c = PGStore().count()
    print(f"document = {c['documents']} 篇 | chunk = {c['chunks']} 块")
    if c['documents'] == 0:
        print("⚠ 库里没数据，跑一次入库：.venv/bin/python scripts/ingest.py")
except Exception as e:
    print(f"⚠ 连 PG 失败：{type(e).__name__}: {e}")
PY
fi

# ============================================================================
# [5/5] 前端（可选）
# ============================================================================
echo ""
echo "==> [5/5] 前端"
if [ "$FRONTEND" != "1" ]; then
    echo "  跳过（要构建前端请用 FRONTEND=1 bash scripts/setup_cloud.sh）"
    echo "  ⚠ 云上没有 node/npm，前端在这个脚本里装不上 → 或者按 docs/15 的\"本地构建后上传 dist\"方案"
else
    if ! command -v node >/dev/null 2>&1; then
        if [ "$SKIP_APT" = "1" ]; then
            echo "  ✗ 没装 node 且 SKIP_APT=1，无法继续"
            exit 1
        fi
        echo "  Node 未安装，安装 Node ${NODE_MAJOR}.x（NodeSource）..."
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - >/dev/null 2>&1
            apt-get install -y -q nodejs >/dev/null 2>&1
            echo "  ✓ Node $(node -v) / npm $(npm -v)"
        else
            echo "  ✗ 没有 curl，装不了 Node。手动装：apt-get install -y curl && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - && apt-get install -y nodejs"
            exit 1
        fi
    else
        echo "  ✓ Node $(node -v) 已存在"
    fi

    cd "$CODE_DIR/frontend"
    echo "  npm install（首次较慢）..."
    npm install --silent --no-audit --no-fund
    echo "  npm run build ..."
    npm run build
    echo "  ✓ 构建完成：$CODE_DIR/frontend/dist"
    cd "$CODE_DIR"
fi

# ============================================================================
# systemd（可选）
# ============================================================================
if [ "$SYSTEMD" = "1" ]; then
    echo ""
    echo "==> 写 systemd unit（常驻服务）"
    UNIT=/etc/systemd/system/rag-agent-api.service
    cat > "$UNIT" <<UNITEOF
[Unit]
Description=Self-Evolving RAG Agent API
After=network.target postgresql.service

[Service]
Type=simple
WorkingDirectory=$CODE_DIR
Environment=RAG_BASE_DIR=$RAG_APP_DIR
# ★ 云上必须监听 0.0.0.0，否则浏览器访问不到（还要开安全组，见 docs/15）
ExecStart=$VENV_PY scripts/api.py --host 0.0.0.0 --port $API_PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNITEOF
    systemctl daemon-reload
    echo "  ✓ 已写入 $UNIT"
    echo "    启用并启动： systemctl enable --now rag-agent-api"
fi

# ============================================================================
# 收尾
# ============================================================================
cat <<NEXT

============================================================
 云上部署检查完成
============================================================

 起后端（云上必须 --host 0.0.0.0，否则外部访问不到）：
     cd $CODE_DIR
     $VENV_PY scripts/api.py --host 0.0.0.0 --port $API_PORT

 自检：
     curl -s localhost:$API_PORT/api/health
     curl -s localhost:$API_PORT/api/stats

 ★ 要在自己电脑的浏览器里访问，必须做两件事：
   1) 服务监听 0.0.0.0（上面已加 --host 0.0.0.0）
   2) 阿里云控制台 → 安全组 → 入方向放行端口 $API_PORT
      （前端 demo 再放行 $FE_PORT；正式部署建议只放 80/443 走 nginx 反代）

 前端两种上法（云上没有 node，二选一）：
   A) 服务器上装 Node 后开发模式跑（最快看到效果，只开一个端口）：
        FRONTEND=1 bash scripts/setup_cloud.sh
        cd $CODE_DIR/frontend && npm run dev -- --host 0.0.0.0 --port $FE_PORT
        # Vite 会在服务器内部把 /api 代理到 127.0.0.1:$API_PORT，浏览器只需访问 $FE_PORT
   B) 本地 build 后把 dist 传上来，用 nginx 托管（正式做法）：
        # 本地： cd frontend && npm run build
        # 上传： scp -r frontend/dist root@你的IP:$CODE_DIR/frontend/
        # nginx：root 指向 .../frontend/dist，location /api 反代 127.0.0.1:$API_PORT
        详见 docs/15-云服务器部署.md

 常驻（避免 SSH 断开就死）：
     SYSTEMD=1 bash scripts/setup_cloud.sh      # 写好 unit
     systemctl enable --now rag-agent-api
     或临时：nohup ... &  /  tmux

 详细排障：docs/15-云服务器部署.md
NEXT
