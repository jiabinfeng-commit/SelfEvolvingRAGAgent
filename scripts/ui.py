#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 5/6/7/8 可视化：Streamlit 交互式页面（三标签页）

为什么做这个页面？
-----------------------------------------------------------------------
阶段 1~4 只能敲命令行 `ask.py`，产品/同事想看效果还得先学开终端、配环境，门槛太高。
Streamlit 用极少量代码就能做出能跑的 Web 界面。但在阶段 5 只覆盖了"问答 + 拒答自愈"。

阶段 6/7/8 做完后，项目有了三件"看不见"的能力：
  - 阶段 6 缺口自愈：联网调研 → 生成 → 质量门禁 → 写影子库 → 再答（过程很精彩，但只有 CLI 能看到）
  - 阶段 7 知识库体检：过时/矛盾/僵尸/重复 四检测器 + Markdown 健康报告（只有 CLI/MD 能看到）
  - 阶段 8 链路追踪：每次问答的召回/分数/prompt/耗时/token/反射 落 PG（只有命令行 get_traces 能查）

本文件把这三件也搬进界面，做成三个标签页，整个项目**全链路可可视化、可演示**：
  标签 1 💬 问答     ：问答 + 阶段5拒答自愈 + 阶段6缺口自愈（开关）+ 阶段8反射/追踪（开关）
  标签 2 🩺 体检     ：跑阶段7 体检，展示概览指标 + Markdown 健康报告 + 待审影子库
  标签 3 🔎 链路追踪 ：读阶段8 trace_log，展示近 50 条问答链路，可展开看细节

两种调用姿势（本项目选「直连 core」，最省事）：
- 方式 A（本文件采用）：直接 import core.*，在页面进程内跑。一条 `streamlit run` 就能用。
- 方式 B（可选）：页面当 HTTP 客户端去调 serve.py 的接口。需先起服务、处理网络异常。
  这里选 A，和 ask.py / serve.py 复用完全相同的 core 逻辑，三端效果一致。

依赖缓存：embedding 模型 / PG 连接 / LLM 客户端较重，用 st.cache_resource 在每个会话里
只建一次，避免每次点击都重连重加载（否则点一次卡半分钟）。

运行：
    streamlit run scripts/ui.py
然后浏览器自动打开 http://localhost:8501

前提（和所有入口一致）：先 `bash scripts/tunnel_pg.sh` 开 PG 隧道；`.env` 里
OPENAI_API_KEY 有效；本地 models/ 下有 bge 模型。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from core import config
from core.rag import generate_answer
from core.agent import run_self_heal
from core.health_check import run_health_check
from core.storage.pg_store import PGStore
from core.embedder import get_embedder
from core import llm as llm_mod


# ================================================================
# 依赖缓存（每个会话只建一次）
# ================================================================

@st.cache_resource
def get_pg():
    """PG 连接缓存（重连成本高）。"""
    return PGStore()


@st.cache_resource
def get_emb():
    """Embedding 模型缓存（首次加载约几秒~十几秒）。"""
    return get_embedder("bge")


@st.cache_resource
def get_llm():
    """LLM 客户端缓存。"""
    return llm_mod.get_llm(config.LLM_BACKEND)


@st.cache_resource
def get_vec():
    """向量库客户端缓存（Milvus Lite 同进程复用）。"""
    from core.storage.vec_store import VecStore
    return VecStore(dim=get_emb().dim)


# ================================================================
# 页面基础设置
# ================================================================
st.set_page_config(page_title="Self-Evolving RAG Agent", page_icon="🤖", layout="wide")
st.title("🤖 Self-Evolving RAG Agent")
st.caption("阶段 1~9 全链路：切片双存储 → RAG 闭环 → 混合检索 → 拒答自愈 → 缺口自愈 → "
           "体检 → 反射核查 → 链路追踪")


# ================================================================
# 侧边栏：共享检索/自愈参数
# ================================================================
with st.sidebar:
    st.header("⚙️ 检索与自愈设置")
    retrieval = st.radio(
        "检索模式", options=["vector", "hybrid"], index=0,
        help="vector=纯向量语义检索；hybrid=向量 + BM25 关键词融合（阶段4）",
    )
    top_k = st.slider("召回块数 top_k", min_value=1, max_value=10, value=config.RAG_TOP_K)
    save = st.checkbox("记录到 qa_log", value=True, help="把每次问答写入 PostgreSQL（阶段2 闭环）")

    st.divider()
    st.subheader("自愈 / 质检开关")
    self_heal = st.checkbox("阶段5 拒答自愈", value=True,
                            help="拒答且召回分够高时，换宽松指令 + 扩大召回自动重试一次")
    agent_heal = st.checkbox("阶段6 缺口自愈（联网补库）", value=False,
                             help="知识库答不上时，联网调研→生成→质量门禁→写影子库→再答")
    reflect = st.checkbox("阶段8 反射事实核查", value=False,
                          help="生成后做事实核查，无依据断言则重检索/重写，仍不过则降级拒答")
    trace = st.checkbox("阶段8 链路追踪落库", value=False,
                        help="把本次问答链路（召回/分数/prompt/耗时/token/反射）写入 trace_log")


# ================================================================
# 标签页
# ================================================================
tab_qa, tab_health, tab_trace = st.tabs(
    ["💬 问答", "🩺 知识库体检（阶段7）", "🔎 链路追踪（阶段8）"]
)


# ----------------------------------------------------------------
# 标签 1：问答（含 阶段5/6/8）
# ----------------------------------------------------------------
with tab_qa:
    st.header("💬 问答")
    question = st.text_input(
        "输入你的问题：",
        placeholder="例如：FastAPI 怎么做依赖注入？ / 怎么集成 Celery？",
    )
    ask_btn = st.button("提问", disabled=not question.strip(), type="primary", key="qa_btn")

    if ask_btn and question.strip():
        try:
            with st.spinner("检索 + 生成中…（首次会加载 embedding 模型，稍等）"):
                if agent_heal:
                    # —— 阶段 6 缺口自愈（联网补库）路径 ——
                    res = run_self_heal(
                        question, top_k=top_k, retrieval=retrieval,
                        pg=get_pg(), emb=get_emb(), vec=get_vec(),
                        backend_llm=get_llm(), bm25=None, save=save,
                    )
                else:
                    # —— 常规路径（阶段 2/4/5/8）——
                    res = generate_answer(
                        question, top_k=top_k, retrieval=retrieval,
                        self_heal=self_heal, save=save,
                        reflect=reflect, trace=trace,
                        pg=get_pg(), emb=get_emb(), vec=get_vec(), backend_llm=get_llm(),
                    )
        except RuntimeError as e:
            st.error(f"调用失败：{e}\n\n请确认：① 已开 PG 隧道；② OPENAI_API_KEY 有效；"
                     f"③ embedding 模型在 models/ 下。")
            st.stop()

        # ---------- 答案区 ----------
        st.subheader("💡 答案")
        st.write(res.get("answer", ""))

        # 状态徽标（阶段5/6/8 拼起来）
        if res.get("self_healed_by_agent"):
            st.success("✅ 阶段6 缺口自愈成功：联网调研 → 过质量门禁 → 写影子库 → 再答")
        elif res.get("self_healed"):
            st.success("✅ 阶段5 拒答自愈成功（原拒答被判为误拒，已救回）")
        elif res.get("refusal"):
            st.warning("⚠️ 模型判定为拒答（未触发自愈，或被安全阀拦下——可能真没答案）")

        if res.get("reflected"):
            rp = res.get("reflect_pass")
            if rp:
                st.info("🔎 阶段8 反射核查通过：答案均有依据")
            else:
                st.warning("🔎 阶段8 反射核查未通过 → 已降级为安全拒答（拦截可能编造的内容）")

        if res.get("request_id"):
            st.caption(f"🔗 trace request_id: {res['request_id']}（可在「链路追踪」标签查看）")

        st.caption(f"⏱️ 生成耗时 {res.get('latency_s', 0)}s ｜ 召回 {len(res.get('retrieved', []))} 块")

        # ---------- 阶段6 专用：调研过程明细 ----------
        if res.get("gap_detected"):
            st.divider()
            st.subheader("🕸️ 阶段6 缺口自愈过程")
            st.write(f"**缺口判定**：{res.get('gap_reason', '')}")
            searched = res.get("searched") or []
            if searched:
                st.write(f"**调研来源（{len(searched)} 个链接）**：")
                for u in searched[:8]:
                    st.markdown(f"- {u}")
            if res.get("quality_detail"):
                qd = res.get("quality_detail")
                st.write(f"**质量门禁**：{res.get('quality_score', 0):.2f} "
                         f"{'通过 ✅' if res.get('quality_pass') else '不通过 ❌'} —— {qd.get('reason', '')}")
            st.write(f"**影子库入库**：{res.get('ingested_chunks', 0)} 个 chunk"
                     f"（doc_id={res.get('shadow_doc_id', '-')}，status=staging）")
            st.write("**执行轨迹**：")
            for s in res.get("steps", []):
                st.markdown(f"- `{s}`")

        # ---------- 召回上下文区 ----------
        st.divider()
        st.subheader("📚 召回的上下文块")
        retrieved = res.get("retrieved", [])
        if not retrieved:
            st.info("没有召回到任何块（可能语料库里确实没有相关内容）")
        for i, c in enumerate(retrieved, 1):
            heading = (c.get("heading") or "").strip()
            label = f"{c.get('doc_name', '')} / {heading}" if heading else c.get("doc_name", "")
            tag = "  ⚠️[未验证来源]" if c.get("unverified") else ""
            with st.expander(f"[{i}] score={c.get('score', 0):.4f}  {label}{tag}"):
                st.write(c.get("content", ""))


# ----------------------------------------------------------------
# 标签 2：知识库体检（阶段7）
# ----------------------------------------------------------------
with tab_health:
    st.header("🩺 知识库体检（阶段7）")
    st.write("跑四个检测器（过时 / 矛盾 / 僵尸 / 重复），产出健康报告。**只读不写**："
             "报告只给建议，删/改动作需人工执行。")
    col1, col2 = st.columns(2)
    with col1:
        no_conflict = st.checkbox("跳过矛盾检测（省 LLM）", value=False,
                                  help="矛盾检测需把同文档 chunk 两两交 LLM 判冲突，关掉则只跑其余三个")
    with col2:
        dup_threshold = st.slider("重复阈值", min_value=0.80, max_value=0.99,
                                  value=0.95, step=0.01,
                                  help="两 chunk 余弦相似度 ≥ 此值判为重复（越高越保守）")

    run_btn = st.button("运行体检", type="primary", key="hc_btn")
    if run_btn:
        try:
            with st.spinner("体检中…（重编码全库 chunk + 可能的 LLM 矛盾判定，稍等）"):
                llm = None if no_conflict else get_llm()
                report = run_health_check(
                    pg=get_pg(), emb=get_emb(), llm=llm,
                    dup_threshold=dup_threshold, report_path=None,
                )
        except RuntimeError as e:
            st.error(f"体检失败：{e}\n\n请确认 PG 隧道已开、配置正确。")
            st.stop()

        s = report["summary"]
        st.subheader("📊 概览")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("文档数", s["documents"])
        m2.metric("块数", s["chunks"])
        m3.metric("问题总数", s["total_issues"])
        bk = s["by_kind"]
        m4.metric("待审影子库", len(report.get("staging_docs", [])))
        st.write(f"分类：过时 {bk.get('stale',0)} / 矛盾 {bk.get('conflict',0)} / "
                 f"僵尸 {bk.get('zombie',0)} / 重复 {bk.get('duplicate',0)}")

        for skip in report.get("skipped", []):
            st.warning(f"跳过：{skip}")

        # 健康报告（Markdown 直接渲染；展开可见完整原文）
        st.subheader("📄 健康报告")
        with st.expander("点击展开完整 Markdown 报告", expanded=True):
            st.markdown(report["markdown"])

        # 待审影子库（阶段6 写入、待人工转 active）
        if report.get("staging_docs"):
            st.subheader("🟡 待人工审核的影子库（staging → active）")
            for d in report["staging_docs"]:
                st.write(f"- `{d['doc_id']}` {d['doc_name']}（写入于 {d['created_at']}）")
            st.caption("处置：人工确认正确后把 document.status 由 staging 改 active；"
                       "错误则删除。未经审核不要直接转 active（防投毒最后一公里）。")


# ----------------------------------------------------------------
# 标签 3：链路追踪（阶段8）
# ----------------------------------------------------------------
with tab_trace:
    st.header("🔎 链路追踪（阶段8）")
    st.write("读取 `trace_log`：每次开启 `--trace` 的问答都会在这里留痕，用于复盘"
             "「为什么答成这样 / 成本花在哪 / 哪些问题常触发反射」。")
    refresh = st.button("刷新", type="primary", key="tr_btn")
    if refresh or "traces" not in st.session_state:
        try:
            with st.spinner("读取 trace_log…"):
                st.session_state["traces"] = get_pg().get_traces(50)
        except RuntimeError as e:
            st.error(f"读取失败：{e}\n\n请确认 PG 隧道已开。")
            st.stop()

    traces = st.session_state.get("traces", [])
    if not traces:
        st.info("trace_log 为空。去「问答」标签勾选「阶段8 链路追踪落库」后问答，这里就会出现记录。")
    else:
        rows = [{
            "时间": str(t["created_at"]),
            "问题": t["question"][:40],
            "检索": t["retrieval"],
            "反射": "✅" if t["reflected"] else "—",
            "核查通过": ("✅" if t["reflect_pass"] else ("❌" if t["reflect_pass"] is False else "—")),
            "耗时(s)": t["latency_s"],
            "估算token": t["est_tokens"],
        } for t in traces]
        st.dataframe(rows, use_container_width=True)

        st.subheader("🔍 明细（点击展开）")
        for t in traces:
            with st.expander(f"{t['created_at']} ｜ {t['question'][:50]} ｜ "
                             f"反射{'✅' if t['reflected'] else '—'}"):
                st.write(f"**request_id**：{t['request_id']}")
                st.write(f"**召回块（{len(t['recalled_ids'])} 个）**："
                         f"{', '.join(t['recalled_ids'][:20])}")
                st.write(f"**召回分数**：{[round(float(x), 3) for x in t['recalled_scores'][:20]]}")
                st.write("**Prompt（发给 LLM 的 user 部分，前 1500 字）**：")
                st.code((t["prompt"] or "")[:1500], language="text")
                st.write("**答案**：")
                st.write(t["answer"])
