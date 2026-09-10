# -*- coding: utf-8 -*-
"""
peek_vec.py —— 本地查看 Milvus Lite 向量库里存了什么

为什么要这个脚本？
- 官方可视化工具 Attu / 命令行 milvus_cli 都要求 Milvus 跑成「服务端」（监听 19530 端口）。
- 我们用的是 Milvus Lite（嵌入式，数据在本地 milvus.db 目录，没有服务端端口），
  所以 Attu 连不上。这个脚本用和入库完全相同的 pymilvus API 直接打开本地文件，
  把集合、字段、条数、抽样数据打印出来，相当于一个「纯读版」的迷你浏览器。

注意：
- 本脚本【只依赖本地 milvus.db】，不需要 PostgreSQL，也不需要 SSH 隧道。
- 它直接读 core/config.py 里的 MILVUS_PATH / COLLECTION_NAME 默认值，
  所以既能在本地跑，也能在云上 /opt/rag-agent 下直接跑（自动识别 BASE_DIR）。

用法：
    # 默认看 chunks 集合，抽样 10 条
    .venv/bin/python scripts/peek_vec.py

    # 抽样 20 条，并把结果导出成 CSV（方便用 Excel / Navicat 打开看）
    .venv/bin/python scripts/peek_vec.py --limit 20 --csv out.csv

    # 指定别的集合名 / 别的库路径
    .venv/bin/python scripts/peek_vec.py --collection chunks --milvus /opt/rag-agent/data/milvus.db
"""
import os
import sys
import argparse
import csv

# 把项目根目录加入 sys.path，这样「无论在哪个目录执行」都能 import core
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient, DataType

from core import config


def load_if_needed(client: MilvusClient, collection: str):
    """
    检索 / 查询前必须确保集合已 load 到内存。

    Milvus（含 Lite）的集合在进程结束后会回到 released 状态，
    新进程直接 query/search 会报 "Collection is in state 'released'"。
    这里复制 core/storage/vec_store.py 里 ensure_loaded() 的兜底逻辑：
    先看 load 状态，没 loaded 就 load，异常就再 try 一次。
    """
    if not client.has_collection(collection):
        return
    try:
        state = client.get_load_state(collection)
        if "Loaded" not in str(state):
            client.load_collection(collection)
    except Exception:
        # 不同 pymilvus 版本 API 行为有差异，兜底直接 load 一次
        try:
            client.load_collection(collection)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser(description="查看本地 Milvus Lite 向量库内容")
    # --limit：抽样打印多少条，默认 10，避免 494 条一次性刷屏
    ap.add_argument("--limit", type=int, default=10, help="抽样打印的条数（默认 10）")
    # --collection：集合名，默认用 config 里的 chunks
    ap.add_argument("--collection", default=config.COLLECTION_NAME, help="集合名")
    # --milvus：直接指定库路径，优先级高于 config 默认值
    ap.add_argument("--milvus", default=config.MILVUS_PATH, help="milvus.db 路径（默认读 config）")
    # --csv：可选导出路径
    ap.add_argument("--csv", default=None, help="把抽样结果导出成 CSV 的路径（可选）")
    args = ap.parse_args()

    # 1. 打开本地 Milvus Lite 文件（uri 传本地路径即嵌入式模式，无需服务端）
    print("=" * 68)
    print("查看 Milvus Lite 向量库")
    print("=" * 68)
    print(f"库路径    : {args.milvus}")
    print(f"集合名    : {args.collection}")
    client = MilvusClient(uri=args.milvus)

    # 2. 列出所有集合
    collections = client.list_collections()
    print(f"\n[1] 库内集合: {collections}")

    # 3. 目标集合不存在就直接退出
    if args.collection not in collections:
        print(f"\n✗ 集合 {args.collection} 不存在，无法查看。")
        sys.exit(1)

    # 4. 打印集合字段结构（schema）：每个字段名 + 类型 + 是否主键
    print(f"\n[2] 集合 {args.collection} 的字段结构:")
    schema = client.describe_collection(args.collection)
    # pymilvus 3.x 的 describe_collection 返回 fields 是列表，每个字段含：
    #   name        字段名
    #   type        字段类型的整数编码（21=VARCHAR, 101=FLOAT_VECTOR, 5=INT64...）
    #   is_primary  是否主键（bool）
    # 用 DataType(编码) 反查可读名字，比直接打数字更直观
    for f in schema.get("fields", []):
        type_code = f.get("type")
        type_name = DataType(type_code).name if type_code is not None else "?"
        pk = " ← 主键" if f.get("is_primary") else ""
        print(f"    - {f.get('name'):<12} {type_name}{pk}")

    # 5. 打印总条数
    # get_collection_stats 返回 dict，row_count 是向量条数
    stats = client.get_collection_stats(args.collection)
    row_count = stats.get("row_count", 0)
    print(f"\n[3] 总条数: {row_count}")

    # 6. 抽样查询（query 与 search 不同，是按条件捞原始字段，不需要 query 向量）
    # filter="" 表示不过滤，limit 控制最多返回多少条，output_fields=["*"] 返回全部字段
    load_if_needed(client, args.collection)
    rows = client.query(
        collection_name=args.collection,
        filter="",
        output_fields=["chunk_id", "doc_id", "vector"],
        limit=args.limit,
    )

    print(f"\n[4] 抽样 {len(rows)} 条（每条只展示向量前 5 维，避免刷屏）:")
    print("-" * 68)
    for r in rows:
        cid = r.get("chunk_id")
        did = r.get("doc_id")
        vec = r.get("vector") or []
        # 向量太长（512 维），只打印前 5 维 + 省略号，并算个 L2 范数方便肉眼看是否归一化
        head = ", ".join(f"{x:.4f}" for x in vec[:5])
        norm = sum(x * x for x in vec) ** 0.5  # L2 范数：COSINE 检索前通常会归一化 → 接近 1.0
        print(f"chunk_id={cid}")
        print(f"  doc_id = {did}")
        print(f"  vector[0:5] = [{head}, ...]  L2范数={norm:.4f}")

    # 7. 可选：导出 CSV（chunk_id, doc_id, 以及 512 维向量用逗号拼成一列）
    if args.csv:
        # 为了导出完整数据，重新按 limit 捞一次完整向量（上面抽样已含 vector，复用即可）
        with open(args.csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            # 表头：前两列是文本字段，后面 512 列是 dim_0..dim_511
            header = ["chunk_id", "doc_id"] + [f"dim_{i}" for i in range(len(config.EMBED_DIM))]
            writer.writerow(header)
            for r in rows:
                v = r.get("vector") or []
                writer.writerow([r.get("chunk_id"), r.get("doc_id")] + v)
        print(f"\n[5] 已导出 {len(rows)} 条到 {args.csv}")

    print("\n完成 ✓（本脚本只读本地库，未连接 PG / 未建隧道）")


if __name__ == "__main__":
    main()
