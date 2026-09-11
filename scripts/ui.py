#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 5 可视化：Streamlit 交互式问答页面

为什么做这个页面？（前面阶段"只能敲命令行"的体验太劝退）
-----------------------------------------------------------------------
阶段 1~4 要试 RAG 效果，只能开终端敲 `python scripts/ask.py "..."`。
产品/同事想看效果还得先学怎么开终端、怎么配环境 —— 门槛太高。
Streamlit 用极少量代码就能做出一个能跑的 Web 界面：
    左边选检索模式 / 勾自愈 / 拖 top_k → 中间输入问题 → 点发送
    → 右边看到答案 + 召回的原文块（带相似度分数，可展开看）。

两种调用姿势（本项目选「直连 core」，最省事）：
- 方式 A（本文件采用）：直接 import core.rag.generate_answer，在页面进程内跑。
    优点：不用先起 FastAPI 服务，一条 `streamlit run` 就能用，依赖最少。
- 方式 B（可选）：页面当 HTTP 客户端去调 serve.py 的 POST /ask。
    优点：和线上服务完全一致；缺点：得先起服务、还要处理网络异常。
  （想切方式 B，把 generate_answer(...) 那行换成 requests.post("http://127.0.0.1:8000/ask", ...)
   即可，逻辑不变。）

运行：
    streamlit run scripts/ui.py
然后浏览器自动打开 http://localhost:8501

注意：本页复用 core.rag.generate_answer —— 也就是和 ask.py / serve.py 完全相同的
问答逻辑，所以页面看到的效果 = 命令行/接口的效果，三端一致（呼应 core/rag.py 的去重）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from core import config
from core.rag import generate_answer


# ---------- 页面基础设置 ----------
st.set_page_config(page_title="Self-Evolving RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 Self-Evolving RAG Agent")
st.caption("阶段 1~5 全链路：切片双存储 → RAG 闭环 → 评估 → 混合检索 → Agent 自愈")


# ---------- 侧边栏：检索参数 ----------
with st.sidebar:
    st.header("⚙️ 检索设置")
    retrieval = st.radio(
        "检索模式",
        options=["vector", "hybrid"],
        index=0,
        help="vector=纯向量语义检索；hybrid=向量 + BM25 关键词融合（阶段4）",
    )
    self_heal = st.checkbox(
        "启用 Agent 自愈（阶段5）",
        value=True,
        help="拒答且召回分够高时，换宽松指令 + 扩大召回自动重试一次",
    )
    top_k = st.slider(
        "召回块数 top_k",
        min_value=1, max_value=10, value=config.RAG_TOP_K,
        help="拼进 prompt 的上下文块数",
    )
    save = st.checkbox(
        "记录到 qa_log",
        value=True,
        help="把每次问答写入 PostgreSQL（阶段2 闭环落库）",
    )


# ---------- 主区：提问 ----------
question = st.text_input(
    "输入你的问题：",
    placeholder="例如：FastAPI 怎么做依赖注入？ / 怎么配置 HTTPS？",
)
ask_btn = st.button("提问", disabled=not question.strip(), type="primary")


if ask_btn and question.strip():
    # 复用共享核心逻辑（与 CLI / HTTP 服务完全一致）
    try:
        with st.spinner("检索 + 生成中…（首次会加载 embedding 模型，稍等）"):
            res = generate_answer(
                question,
                top_k=top_k,
                retrieval=retrieval,
                self_heal=self_heal,
                save=save,
            )
    except RuntimeError as e:
        st.error(f"调用失败：{e}\n\n请确认：① 已开 PG 隧道（bash scripts/tunnel_pg.sh）；"
                 f"② .env 的 OPENAI_API_KEY 有效；③ embedding 模型在 models/ 下。")
        st.stop()

    # ---------- 答案区 ----------
    st.subheader("💡 答案")
    st.write(res["answer"])

    if res.get("self_healed"):
        st.success("✅ 本次触发了阶段5 Agent 自愈并成功救回（原拒答被判为误拒）")
    elif res.get("refusal"):
        st.warning("⚠️ 模型判定为拒答（未触发自愈，或被安全阀拦下——可能是真没答案）")

    st.caption(f"⏱️ 生成耗时 {res['latency_s']}s ｜ 召回 {len(res['retrieved'])} 块")

    # ---------- 召回上下文区（可展开看原文）----------
    st.subheader("📚 召回的上下文块")
    if not res["retrieved"]:
        st.info("没有召回到任何块（可能语料库里确实没有相关内容）")
    for i, c in enumerate(res["retrieved"], 1):
        heading = (c.get("heading") or "").strip()
        label = f"{c['doc_name']} / {heading}" if heading else c["doc_name"]
        with st.expander(f"[{i}] score={c['score']:.4f}  {label}"):
            st.write(c["content"])
