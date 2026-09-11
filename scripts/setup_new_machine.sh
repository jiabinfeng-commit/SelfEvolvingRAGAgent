#!/usr/bin/env bash
# ============================================================================
# setup_new_machine.sh —— 新人 / 换机器：从零把项目跑起来（全本地，零云依赖）
#
# 和 setup_local.sh 的区别：
#   setup_local.sh   是**作者本人**用的：本地开发复用自己云上的 PG + 向量库（要 SSH 隧道）
#   本脚本           是**任何拿到这份代码的人**用的：本地 Docker 起 PG + 自己下模型 + 重新入库
#
# 做的事（幂等，可以重复跑）：
#   [1/6] 检查环境（python3 / node / docker）
#   [2/6] 建 Python 虚拟环境 .venv
#   [3/6] 装 Python 依赖（requirements.txt）
#   [4/6] 下 embedding 模型（从 ModelScope，约 183MB）
#   [5/6] 起本地 PostgreSQL（Docker 容器 rag-pg）
#   [6/6] 生成 .env + 配置自检
#
# 用法：
#   bash scripts/setup_new_machine.sh
#   RAG_PG_PASSWORD=mypass bash scripts/setup_new_machine.sh   # 自定义 PG 密码
#   PYTHON_BIN=python3.12 bash scripts/setup_new_machine.sh    # 指定 python
#   SKIP_PG=1 bash scripts/setup_new_machine.sh                # 不起 PG（你已有 PG）
#
# 跑完后剩两步（脚本不会自动做，因为要你自己确认 LLM 配置）：
#   .venv/bin/python scripts/ingest.py     # 把 corpus/clean 灌进库
#   .venv/bin/python scripts/api.py        # 起后端
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------- 可调参数（都可用环境变量覆盖）----------
PYTHON_BIN="${PYTHON_BIN:-python3}"
PG_CONTAINER="${PG_CONTAINER:-rag-pg}"
PG_PORT="${PG_PORT:-5432}"
PG_USER="${PG_USER:-rag}"
PG_DB="${PG_DB:-rag}"
PG_PASSWORD="${RAG_PG_PASSWORD:-rag_dev}"     # 仅本地开发用的弱密码
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple/}"
SKIP_PG="${SKIP_PG:-0}"

echo "============================================================"
echo " Self-Evolving RAG Agent —— 全本地环境搭建"
echo "============================================================"
echo " 项目根   : $ROOT"
echo " Python   : $PYTHON_BIN"
echo " PG 容器  : $PG_CONTAINER (localhost:$PG_PORT, 用户=$PG_USER, 库=$PG_DB)"
echo " pip 源   : $PIP_INDEX"
echo ""

# ---------- 防误用：这个脚本是给"干净本地机器"用的，别在云服务器上跑 ----------
# 云上（/opt/rag-agent）已经有：git 代码 + venv + 已下好的模型 + 已有数据的向量库
# + 原生 PostgreSQL。在这里重跑本脚本会白建一个 .venv、重下 183MB 模型、
# 还想用 Docker 起 PG（云上没 Docker）——纯浪费。
# （CLOUD_MARKER 抽成变量，方便测试与自定义；正常不用改）
CLOUD_MARKER="${CLOUD_MARKER:-/opt/rag-agent}"
if [ -d "$CLOUD_MARKER" ] && [ "${FORCE:-0}" != "1" ]; then
    cat <<GUARD
✗ 检测到 $CLOUD_MARKER 存在 —— 这看起来是**云服务器**，不是干净的本地机器。

  本脚本是给"干净本地机器"用的（建 venv / 下模型 / Docker 起 PG）。
  云上请改用专门脚本，它会复用已有的 venv / 模型 / 向量库 / 原生 PostgreSQL：

      bash scripts/setup_cloud.sh
      FRONTEND=1 bash scripts/setup_cloud.sh     # 顺带装 Node + 构建前端
      SYSTEMD=1  bash scripts/setup_cloud.sh     # 顺便写 systemd 常驻服务

  详见 docs/15-云服务器部署.md

  （确实要强行在云上跑本脚本？FORCE=1 bash scripts/setup_new_machine.sh）
GUARD
    exit 1
fi

# ============================================================================
# [1/6] 环境检查
# ============================================================================
echo "==> [1/6] 检查环境"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "  ✗ 找不到 $PYTHON_BIN。请先装 Python 3.11+，或用 PYTHON_BIN=... 指定"
    exit 1
fi
PY_VER="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "  ✓ Python $PY_VER"

HAVE_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    HAVE_DOCKER=1
    echo "  ✓ Docker 可用"
else
    echo "  ⚠ Docker 不可用（没装或没启动）—— PG 那步会跳过，你需要自己准备 PostgreSQL"
fi

if command -v node >/dev/null 2>&1; then
    echo "  ✓ Node $(node -v)（前端需要）"
else
    echo "  ⚠ 没找到 node —— 前端跑不起来，请装 Node 18+"
fi

# ============================================================================
# [2/6] 虚拟环境
# ============================================================================
echo ""
echo "==> [2/6] Python 虚拟环境 .venv"
if [ -d .venv ]; then
    echo "  已存在，跳过创建"
else
    "$PYTHON_BIN" -m venv .venv
    echo "  已创建 .venv"
fi
VENV_PY="$ROOT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$ROOT/.venv/Scripts/python.exe"   # Windows(Git Bash) 兜底

# ============================================================================
# [3/6] Python 依赖
# ============================================================================
echo ""
echo "==> [3/6] 安装 Python 依赖（torch 首次较慢；已用国内源）"
echo "  提示：只用 CPU 的话，先按 requirements.txt 头注释装 CPU 版 torch 能快很多"
"$VENV_PY" -m pip install -q --upgrade pip
"$VENV_PY" -m pip install -q -r requirements.txt -i "$PIP_INDEX"
echo "  ✓ 依赖就绪"

# ============================================================================
# [4/6] embedding 模型
# ============================================================================
echo ""
echo "==> [4/6] 下载 embedding 模型（ModelScope，约 183MB）"
if [ -d "$ROOT/models/BAAI__bge-small-zh-v1.5" ]; then
    echo "  模型已存在，跳过"
else
    "$VENV_PY" scripts/fetch_model.py
fi

# ============================================================================
# [5/6] PostgreSQL
# ============================================================================
echo ""
echo "==> [5/6] PostgreSQL"
PG_READY=0
if [ "$SKIP_PG" = "1" ]; then
    echo "  SKIP_PG=1，跳过（假设你已有 PG，记得在 .env 里填对连接信息）"
elif [ "$HAVE_DOCKER" = "0" ]; then
    echo "  ⚠ Docker 不可用，跳过。请自己装一个 PostgreSQL，或在 .env 里指向已有实例。"
    echo "    最省事的替代方案（如果你机器上有 PostgreSQL）："
    echo "      createdb $PG_DB && createuser -P $PG_USER"
else
    if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
        if docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
            echo "  容器 $PG_CONTAINER 已在运行"
        else
            echo "  容器 $PG_CONTAINER 存在但没跑，启动它"
            docker start "$PG_CONTAINER" >/dev/null
        fi
        PG_READY=1
    elif lsof -nP -iTCP:"$PG_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "  ⚠ 端口 $PG_PORT 已被别的进程占用（可能是你已有的 PostgreSQL）"
        echo "    跳过创建容器。请在 .env 里把 PG_PASSWORD 改成**你那个 PG 的**密码。"
    else
        echo "  创建并启动容器 $PG_CONTAINER ..."
        docker run -d --name "$PG_CONTAINER" -p "$PG_PORT:5432" \
            -e "POSTGRES_USER=$PG_USER" \
            -e "POSTGRES_PASSWORD=$PG_PASSWORD" \
            -e "POSTGRES_DB=$PG_DB" \
            postgres:16 >/dev/null
        echo "  等待 PG 就绪..."
        for _ in $(seq 1 30); do
            if docker exec "$PG_CONTAINER" pg_isready -U "$PG_USER" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        echo "  ✓ PG 容器就绪（密码：$PG_PASSWORD；这是本地开发用的弱密码）"
        PG_READY=1
    fi
fi

# ============================================================================
# [6/6] .env + 自检
# ============================================================================
echo ""
echo "==> [6/6] 生成 .env 并自检"
if [ -f .env ]; then
    echo "  .env 已存在，**不覆盖**（要重建请先自己删掉或改名）"
else
    "$VENV_PY" - "$ROOT" "$PG_PORT" "$PG_USER" "$PG_DB" "$PG_PASSWORD" <<'PY'
import os, sys
root, port, user, db, pwd = sys.argv[1:6]
src = os.path.join(root, ".env.example")
dst = os.path.join(root, ".env")
text = open(src, encoding="utf-8").read()
# 把模板里的占位密码换成本次实际使用的密码
text = text.replace(
    "PG_PASSWORD=请填写真实密码，不要提交到 git",
    f"PG_PASSWORD={pwd}",
)
# 保险：万一模板文案变了，兜底把所有 PG_* 行重写一遍
import re
text = re.sub(r"^PG_PORT=.*$", f"PG_PORT={port}", text, flags=re.M)
text = re.sub(r"^PG_USER=.*$", f"PG_USER={user}", text, flags=re.M)
text = re.sub(r"^PG_DB=.*$", f"PG_DB={db}", text, flags=re.M)
open(dst, "w", encoding="utf-8").write(text)
os.chmod(dst, 0o600)          # 只本人可读（里面有密码）
print("  已生成 .env（权限 600）")
PY
fi

echo ""
"$VENV_PY" -c "
import sys; sys.path.insert(0, '.')
from core import config
print('  --- 配置摘要 ---')
print('  ' + config.summary().replace('\n', '\n  '))
errs = config.validate()
if errs:
    print()
    for e in errs:
        print('  ✗', e)
    print()
    print('  提示：')
    print('   · PG_PASSWORD 相关 → 检查 .env 里的密码 / PG 是否起来')
    print('   · 语料目录相关    → corpus/clean 应该已在仓库里，检查是否被删')
    print('   · OPENAI_API_KEY  → LLM_BACKEND=openai 时必须填')
    sys.exit(1)
print()
print('  ✓ 配置校验通过')
"

# ============================================================================
# 收尾：告诉用户还剩哪两步
# ============================================================================
cat <<'NEXT'

============================================================
 环境就绪！还剩两步（脚本不代做，因为要你先确认 LLM 配置）
============================================================

 ★ 第 0 步（重要）：编辑 .env 的 LLM 部分，二选一
     A) 用云 API（快）：LLM_BACKEND=openai
                        LLM_MODEL=qwen3.7-max
                        OPENAI_API_KEY=sk-xxx
                        OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
     B) 纯本地免费  ：LLM_BACKEND=ollama
                        LLM_MODEL=qwen2.5:7b          （需先装 Ollama 并 ollama pull qwen2.5:7b）
   不配的话：知识库上传/检索都能用，但"问答"会报 503。

 第 1 步：入库（把仓库里的 corpus/clean 灌进 PG + Milvus，几分钟）
     .venv/bin/python scripts/ingest.py

 第 2 步：起服务
     # 后端 → http://127.0.0.1:8000/docs
     .venv/bin/python scripts/api.py
     # 前端 → http://127.0.0.1:5173
     cd frontend && npm install && npm run dev

 自检：
     curl localhost:8000/api/health     # 期望 {"status":"ok", ...}
     curl localhost:8000/api/stats      # documents/chunks/vectors 有数字

 详细排障：docs/14-拉取代码后如何跑起来.md
NEXT
