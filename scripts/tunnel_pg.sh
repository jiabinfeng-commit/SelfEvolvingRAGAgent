#!/usr/bin/env bash
# ============================================================================
# ⚠️ 已非默认路径（仅云主机复用场景才需要）
#
# 本项目的默认开发方式是「本地 Docker 起 PostgreSQL」：
#   docker run -d --name rag-pg -p 5432:5432 \
#     -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag_dev -e POSTGRES_DB=rag postgres:16
# 然后 .env 里 PG_HOST=localhost 即可，根本不需要隧道。新人请直接用：
#   bash scripts/setup_new_machine.sh
#
# 这个脚本只在「你确实有云主机、且想从本机直接连云上的 PG」时才用，
# 例如临时查云上数据、和云上 1029 块做对比。
#
# —— 端口冲突说明（重要）——
#   本地 Docker 的 rag-pg 已经占了 5432。本脚本检测到 5432 被占用会**直接退出并提示**，
#   这是预期行为，不是 bug：请先 `docker stop rag-pg`（或换 RAG_LOCAL_PG_PORT=5433）再开隧道。
#   注意：一旦隧道把本地 5432 指向云上，你的应用就会**悄悄读云上数据**，排查时务必分清连的是哪边。
#
# 原用途：建立 SSH 隧道，本地 5432 → 云上 127.0.0.1:5432
#   云上 PostgreSQL 只监听 127.0.0.1，公网连不上（安全）。
#   SSH 隧道在云主机内部把请求转发给 localhost:5432，不用改 PG 配置、不开安全组。
#
# 用法：
#   RAG_CLOUD_HOST=root@你的服务器IP bash scripts/tunnel_pg.sh
# 保持终端开着，Ctrl-C 结束。
# ============================================================================
set -euo pipefail

# 云主机地址：**必须显式指定**，故意不给默认值。
# 原因：① 仓库可能被别人 clone，默认指向某个人的服务器会让脚本去连陌生主机；
#       ② 服务器 IP 不该硬编码进代码。用法见文件头的 RAG_CLOUD_HOST。
CLOUD_HOST="${RAG_CLOUD_HOST:-}"
LOCAL_PORT="${RAG_LOCAL_PG_PORT:-5432}"

if [ -z "$CLOUD_HOST" ]; then
    cat <<'TIP'
✗ 未指定云主机地址（RAG_CLOUD_HOST）

  用法：
      RAG_CLOUD_HOST=root@你的服务器IP bash scripts/tunnel_pg.sh

  例：
      RAG_CLOUD_HOST=root@1.2.3.4 bash scripts/tunnel_pg.sh

  不想每次都敲？在 ~/.zshrc 里加一行（只在你本机生效，不会进 git）：
      export RAG_CLOUD_HOST=root@1.2.3.4

  ⚠ 没有云主机 / 不想用隧道？改用本地 Docker 起 PG 就行，不需要这个脚本：
      docker run -d --name rag-pg -p 5432:5432 \
        -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=你的密码 -e POSTGRES_DB=rag postgres:16
    然后把 .env 的 PG_HOST 设为 localhost 即可。（详见 docs/14-拉取代码后如何跑起来.md）
TIP
    exit 1
fi

# 端口占用检查
if lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "⚠️  本地 $LOCAL_PORT 端口已被占用："
    lsof -nP -iTCP:"$LOCAL_PORT" -sTCP:LISTEN | tail -n +2
    echo ""
    echo "   如果是别的 PostgreSQL，换一个端口再跑："
    echo "   RAG_LOCAL_PG_PORT=5433 bash scripts/tunnel_pg.sh"
    echo "   （同时把 .env 里的 PG_PORT 改成 5433）"
    exit 1
fi

echo "建立隧道：localhost:$LOCAL_PORT → $CLOUD_HOST 的 127.0.0.1:5432"
echo "保持这个终端开着，Ctrl-C 结束。"
echo ""

ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "$LOCAL_PORT":127.0.0.1:5432 -N "$CLOUD_HOST"
