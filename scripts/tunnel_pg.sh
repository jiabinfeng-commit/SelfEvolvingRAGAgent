#!/usr/bin/env bash
# ============================================================================
# 建立 SSH 隧道：本地 5432 端口 → 云上 127.0.0.1:5432
#
# 为什么必须这样：
#   云上 PostgreSQL 只监听 127.0.0.1，公网根本连不上（这是好事，安全）。
#   SSH 隧道在云主机内部把请求转发给 localhost:5432，
#   所以既不用改 PG 配置，也不用开阿里云安全组端口。
#
# 用法：
#   bash scripts/tunnel_pg.sh
#
# 这个终端要一直开着，关掉隧道就断了。
# 想停：Ctrl-C
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
