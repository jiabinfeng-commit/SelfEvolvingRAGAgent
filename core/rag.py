# -*- coding: utf-8 -*-
"""
RAG 问答核心逻辑（阶段 2 / 4 / 5 共用，单一事实来源）

为什么单独抽这一层？（这一层和 core/retrieval.py 的去重是同一个思路）
-----------------------------------------------------------------------
ask.py（命令行）、serve.py（HTTP 服务）、ui.py（streamlit 页面）这三个入口，
底层要做的其实是同一件事：

    问题 → 召回（向量 / 混合）→ 拼 prompt → LLM 生成 → （阶段5 自愈）→ 落库

之前 ask.py 把整段流程写死在 main() 里，serve.py / ui.py 若再各抄一份，
就会出现「一处改了检索口径、另一处忘了改」的经典漂移 bug。
所以把核心逻辑抽到这里，三个入口都调 generate_answer()，保证业务口径只此一份。

返回值是一个 dict（而不是 print），这样三种入口能各自渲染：
    - CLI：打印到终端
    - HTTP 服务：直接 json 化返回（FastAPI 会自动序列化）
    - streamlit：塞进各种组件

设计要点：
- 依赖（emb / vec / pg / llm / bm25）都允许从外部注入：
  CLI 一次性调用时传 None，函数内部惰性创建并自己关闭；
  服务/页面这种「常驻」场景，把已建好的实例传进来复用，避免每问一次重连重加载。
- dry_run：只做召回 + 拼 prompt，不调 LLM、不落库（ask.py 的 --dry-run 用）。
"""
import time
import uuid
from typing import List, Dict, Any, Optional

from core import config
from core.embedder import get_embedder
from core.storage.pg_store import PGStore
from core.storage.vec_store import VecStore
from core import llm as llm_mod
from core.prompt import build_prompt, SYSTEM_PROMPT, RELAXED_SYSTEM_PROMPT
from core.retrieval import retrieve, hybrid_retrieve
from core.self_heal import is_refusal, should_heal, SELF_HEAL_TOPK_MULT
from core.bm25 import BM25
from core.reflexion import run_reflexion
from core.tracing import TraceRecorder


def generate_answer(question: str,
                    top_k: int = None,
                    retrieval: str = "vector",
                    self_heal: bool = False,
                    save: bool = True,
                    dry_run: bool = False,
                    emb=None, vec=None, pg=None,
                    backend_llm=None, bm25=None,
                    reflect: bool = False,
                    trace: bool = False,
                    request_id: str = None,
                    trace_recorder=None) -> Dict[str, Any]:
    """
    端到端回答一个问题，返回结构化结果。

    :param question:    用户问题
    :param top_k:       召回多少块（None → config.RAG_TOP_K，默认 5）
    :param retrieval:   "vector"（纯向量语义检索）| "hybrid"（向量 + BM25 关键词融合）
    :param self_heal:   是否启用阶段 5 的 Agent 自愈（拒答且召回分够高时换策略重试）
    :param save:        是否把这次问答写入 PG 的 qa_log（闭环落库）
    :param dry_run:     只召回 + 拼 prompt，不调 LLM、不落库（开发调试用）
    :param reflect:     阶段 8：生成答案后做事实核查（Reflexion）。发现无依据断言会
                        重检索/重写，仍不通过则降级为安全拒答。默认关。
    :param trace:       阶段 8：把本次问答链路（召回/分数/prompt/耗时/估算token/反射结果）
                        写入 PG 的 trace_log，供可观测看板。默认关。
    :param request_id:  链路追踪用的请求 id（一次会话一个 uuid）；trace=True 且不传则自动生成。
    :param trace_recorder: 可注入已建好的 TraceRecorder（常驻场景复用）；不传则现场建。
    :param emb/vec/pg/backend_llm/bm25: 可注入已建好的实例（服务/页面常驻复用场景）。
                         任一为 None 时，本函数按需要惰性创建；其中 pg 由本函数创建时会
                         在函数结束时自动关闭，注入的 pg 由调用方负责生命周期。
    :return: dict：
        {
          "question":   str,
          "answer":     str,                 # dry_run 时为 ""
          "retrieved":  [ {chunk_id, score, doc_name, heading, content}, ... ],
          "latency_s":  float,               # LLM 生成耗时（dry_run 时为 0）
          "refusal":    bool,                # 最终答案是否仍属拒答
          "self_healed":bool,                # 是否触发自愈且成功救回
        }
    """
    top_k = top_k or config.RAG_TOP_K
    if retrieval not in ("vector", "hybrid"):
        raise ValueError(f"retrieval 必须是 vector 或 hybrid，收到: {retrieval}")

    # pg 是否由本函数自己创建：只有自己创建的才在函数结束时关闭
    own_pg = pg is None
    if own_pg:
        pg = PGStore()

    try:
        # 惰性创建其余依赖（CLI / 一次性调用场景）
        if emb is None:
            emb = get_embedder("bge")
        if vec is None:
            vec = VecStore(dim=emb.dim)
        if backend_llm is None:
            backend_llm = llm_mod.get_llm(config.LLM_BACKEND)

        # —— 第一步：召回 ——
        if retrieval == "hybrid":
            # 混合检索需要 BM25 关键词索引。没注入就现场建（从 PG 全量拉 chunk）。
            # 注意：这里建的 bm25 是函数局部的，不会缓存到调用方；
            # 服务场景会在启动时建好并注入，避免每问一次重建。
            if bm25 is None:
                all_chunks = pg.get_all_chunks()
                bm25 = BM25().build(
                    [(c["chunk_id"], (c.get("content") or "")) for c in all_chunks]
                )
            retrieved = hybrid_retrieve(question, emb, vec, pg, bm25, top_k=top_k)
        else:
            retrieved = retrieve(question, emb, vec, pg, top_k)

        # —— 第二步：拼 prompt（user 部分）——
        contexts = [c for _, c in retrieved]
        user_prompt = build_prompt(question, contexts,
                                   max_chars=config.RAG_CONTEXT_MAX_CHARS)

        # dry-run：到此为止，返回召回结果 + 拼好的 prompt，不调 LLM、不落库
        if dry_run:
            return {
                "question": question,
                "answer": "",
                "user_prompt": user_prompt,            # 调试用：能看到发给 LLM 的原文
                "system_prompt": SYSTEM_PROMPT,
                "retrieved": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "score": float(s),
                        "doc_name": c.get("doc_name"),
                        "heading": c.get("heading"),
                        "content": c.get("content"),
                    }
                    for s, c in retrieved
                ],
                "latency_s": 0.0,
                "refusal": False,
                "self_healed": False,
            }

        # —— 第三步：LLM 首轮生成 ——
        t0 = time.time()
        answer = backend_llm.generate(SYSTEM_PROMPT, user_prompt)
        latency = time.time() - t0

        self_healed = False

        # —— 第四步：阶段 5 自愈 ——
        # 触发条件（is_refusal + 召回分够高）已在 should_heal 内部判定。
        if self_heal and retrieved and should_heal(answer, retrieved[0][0]):
            heal_k = max(int(top_k * SELF_HEAL_TOPK_MULT), top_k)  # 扩大召回给更多素材
            try:
                retrieved_h = (
                    hybrid_retrieve(question, emb, vec, pg, bm25, top_k=heal_k)
                    if retrieval == "hybrid"
                    else retrieve(question, emb, vec, pg, heal_k)
                )
                contexts_h = [c for _, c in retrieved_h]
                prompt_h = build_prompt(question, contexts_h,
                                        max_chars=config.RAG_CONTEXT_MAX_CHARS)
                ans_h = backend_llm.generate(RELAXED_SYSTEM_PROMPT, prompt_h)
                # 只有"重试后确实不再拒答"才采纳，避免把拒答救成幻觉
                if ans_h and not is_refusal(ans_h):
                    answer = ans_h
                    retrieved = retrieved_h
                    self_healed = True
            except RuntimeError:
                # 自愈过程中 LLM/召回异常：保持首轮原答案，不自欺欺人
                pass

        # —— 第四步半：阶段 8（可选）Reflexion 事实核查 ——
        # 放在 self_heal 之后：先尽力答出/救回，再对"最终答案"做一次抗幻觉核查。
        reflected = False
        reflect_pass = None
        reflect_detail: Dict[str, Any] = {}
        if reflect:
            final_a, reflected, reflect_pass, reflect_detail = run_reflexion(
                question, answer, retrieved, emb, vec, pg, backend_llm,
                bm25=bm25, top_k=top_k, retrieval=retrieval,
            )
            answer = final_a   # 可能已被重写，或降级为安全拒答
            # 若降级成了拒答，refusal 标志同步更新（下面组装时会用 is_refusal 再判一次）

        # —— 第五步：落库（阶段 2 闭环的"答案落库"）——
        if save:
            ctx_ids = [c.get("chunk_id") for _, c in retrieved]
            ctx_scores = [float(s) for s, _ in retrieved]
            pg.save_qa(
                question=question,
                answer=answer,
                ctx_chunk_ids=ctx_ids,
                ctx_scores=ctx_scores,
                model=config.LLM_MODEL,
                backend=config.LLM_BACKEND,
                latency_s=latency,
            )

        # —— 第六步：阶段 8（可选）链路追踪落库 ——
        # 旁路：写失败不影响主回答。trace_recorder 可注入（常驻复用），否则现场建一个。
        if trace:
            try:
                recorder = trace_recorder or TraceRecorder(pg)
                rid = recorder.record(
                    question=question,
                    retrieval=retrieval,
                    backend=config.LLM_BACKEND,
                    recalled_ids=[c.get("chunk_id") for _, c in retrieved],
                    recalled_scores=[float(s) for s, _ in retrieved],
                    prompt=user_prompt,
                    answer=answer,
                    latency_s=latency,
                    reflected=reflected,
                    reflect_pass=reflect_pass,
                    request_id=request_id,
                )
                request_id = rid
            except Exception:
                pass   # 追踪是旁路，任何异常都不该影响返回

        # —— 第七步：组装返回 ——
        return {
            "question": question,
            "answer": answer,
            "retrieved": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "doc_id": c.get("doc_id"),
                    "score": float(s),
                    "doc_name": c.get("doc_name"),
                    "heading": c.get("heading"),
                    "content": c.get("content"),
                }
                for s, c in retrieved
            ],
            "latency_s": round(latency, 2),
            "refusal": is_refusal(answer),     # 用最终答案判定（自愈/降级成功则 False）
            "self_healed": self_healed,
            # 阶段 8 新增字段
            "reflected": reflected,
            "reflect_pass": reflect_pass,
            "reflect_detail": reflect_detail,
            "request_id": request_id,
        }
    finally:
        # 只有本函数自己创建的 pg 才负责关闭；注入的由调用方管理
        if own_pg:
            pg.close()
