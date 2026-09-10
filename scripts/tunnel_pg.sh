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

CLOUD_HOST="${RAG_CLOUD_HOST:-root@120.26.22.166}"
LOCAL_PORT="${RAG_LOCAL_PG_PORT:-5432}"

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
