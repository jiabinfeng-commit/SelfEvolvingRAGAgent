# -*- coding: utf-8 -*-
"""
core/reflexion.py —— 阶段 8（上半）：Reflexion 事实核查自检闭环

====================================================================
Reflexion 在这里到底做什么？
====================================================================
前面阶段 5（拒答自愈）、阶段 6（缺口自愈）解决的是"答不出 / 答不全"。
但还有一个更阴险的问题没解决：**答出来了，但里面有一句是编的**（幻觉）。

阶段 2 的 SYSTEM_PROMPT 已经要求"只依据上下文、不编造"，但模型不一定每次都听话。
Reflexion 就是在"生成答案之后、返回用户之前"插一道**事实核查**：
    让 LLM 当考官，逐条核对"答案里的每个说法，能不能在参考资料里找到依据"。
    - 全有依据 → 通过，直接返回。
    - 有依据不足的断言 → 不通过：
        ① 重新检索（扩大召回，拿更多素材）
        ② 用更严的"必须有据"指令重写答案
        ③ 再核查一遍
        - 还过不了 → 降级为安全拒答（明确说"资料里答不了"），
          绝不把带编造的答案甩给用户。

这就是 Reflexion 的核心思想：生成 → 自我反思 →（不行就）行动重试，而不是一次就完。
和阶段 5 的区别：阶段 5 是"拒答了才救"，这里是"答了但可能编造，也要救"。

====================================================================
为什么不全依赖 prompt 约束？
====================================================================
单一 system 指令是"靠模型自觉"，不可控。Reflexion 把"是否幻觉"变成了一个
**可程序化判断 + 可重试**的闭环：不通过就真的去多搜、去重写，直到通过或降级。
这样"抗幻觉"从"希望模型听话"升级成"系统级保障"。

====================================================================
对外暴露
====================================================================
- reflect(answer, contexts, llm) -> (grounded, issues, reason)
- run_reflexion(question, answer, retrieved, emb, vec, pg, llm, bm25, top_k, retrieval)
      -> (final_answer, reflected, reflect_pass, detail)
"""
import json
import re
from typing import List, Dict, Any, Tuple, Optional

from core import config
from core.retrieval import retrieve, hybrid_retrieve
from core.prompt import build_prompt


# 反思重试次数上限（防无限循环烧 token；一次不通过最多再尝试这么多次）
REFLECT_MAX_RETRY = 1

# 重新检索时把 top_k 放大几倍（给模型更多素材，提高"找到依据"的概率）
REFLECT_TOPK_MULT = 2


# ================================================================
# 事实核查 prompt
# ================================================================

_REFLECT_SYSTEM = (
    "你是严谨的答案事实核查员。下面给你一份【模型回答】和对应的【参考资料】。\n"
    "请逐条核对：回答里每一个具体的**事实性断言**（数字、配置、步骤、结论等），"
    "是否都能在参考资料中找到依据。\n"
    "判断标准：\n"
    "1. 只要回答里的关键事实在参考资料里有对应支撑，就算有依据；表述不同不算无依据。\n"
    "2. 若某断言在参考资料里完全找不到、且明显是外部知识/编造，就是无依据。\n"
    "3. 泛泛的衔接语（'综上所述'、'需要注意的是'）不算断言，不考核。\n"
    "只输出一个 JSON，不要多余文字，格式：\n"
    '{"grounded": true/false, "issues": ["无依据的具体说法1", ...], "reason": "一句话总结"}'
)

_REFLECT_USER_TMPL = (
    "【参考资料】\n{contexts}\n\n"
    "【模型回答】\n{answer}\n\n"
    "请核查该回答是否全部基于上述参考资料（无编造/无引入外部未证实知识）。"
)


def _extract_json(text: str) -> Optional[dict]:
    """从 LLM 输出抠 JSON（兼容 ```json 包裹 / 前后废话）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return None
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return None


def reflect(answer: str, contexts: List[Dict], llm) -> Tuple[bool, List[str], str]:
    """
    事实核查：判断答案是否全部基于参考资料（无幻觉）。

    :param answer:    模型生成的答案
    :param contexts: 实际拼进 prompt 的 chunk 列表（来自 generate_answer 的 retrieved）
    :param llm:       BaseLLM 实例
    :return: (grounded 是否通过, issues 无依据的具体说法列表, reason 一句话)
    """
    ctx_text = "\n\n".join(
        f"[{i+1}] {c.get('doc_name','')}：{(c.get('content') or '')[:400]}"
        for i, c in enumerate(contexts)
    )
    user = _REFLECT_USER_TMPL.format(contexts=ctx_text or "（无）", answer=answer or "（空）")
    try:
        raw = llm.generate(_REFLECT_SYSTEM, user)
    except Exception as e:
        # 核查调用挂了：保守判"不通过"，让上层走重试/降级，而不是把可能编造的答案放出去
        return False, [], f"核查调用失败: {e}"
    obj = _extract_json(raw)
    if not obj:
        return False, [], "核查未返回合法 JSON，保守判不通过"
    grounded = bool(obj.get("grounded"))
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]
    return grounded, issues, obj.get("reason", "")


# ================================================================
# 重写用 prompt（更强调"必须有据"）
# ================================================================

_REWRITE_SYSTEM = (
    "你是严谨的技术文档问答助手。请只根据下面【上下文】重写答案。\n"
    "规则：\n"
    "1. 每一句事实性断言都必须能在上下文中找到依据，找不到依据的**不要写**。\n"
    "2. 若上下文无法支撑原回答中的某些点，直接省略或用「资料中未提及」说明，不要编造。\n"
    "3. 回答简洁、准确，可引用上下文要点。"
)


def run_reflexion(question: str, answer: str, retrieved: List[Any],
                  emb, vec, pg, llm, bm25=None, top_k: int = None,
                  retrieval: str = "vector",
                  max_retry: int = REFLECT_MAX_RETRY) -> Tuple[str, bool, bool, dict]:
    """
    阶段 8 反射闭环：对一份答案做事实核查，不通过则重试，仍不通过则降级拒答。

    :param question:   原问题（重检索用）
    :param answer:     首轮（可能已自愈）的答案
    :param retrieved:  generate_answer 的 retrieved（[(score, chunk)] 或已取回的 chunk dict 列表）
                       这里统一从它取 chunk 原文作为 contexts
    :param emb/vec/pg/llm/bm25: 重检索需要的依赖（与 generate_answer 一致）
    :param top_k/retrieval: 召回参数
    :param max_retry:  重试上限
    :return: (final_answer, reflected 是否触发了反射, reflect_pass 最终是否通过, detail)
    """
    top_k = top_k or config.RAG_TOP_K

    # 把 retrieved 规整成 contexts（兼容 [(score, chunk)] 与 [{chunk...}] 两种形态）
    contexts = []
    for item in retrieved:
        if isinstance(item, tuple):
            contexts.append(item[1])
        else:
            contexts.append(item)

    # —— 第一轮核查 ——
    grounded, issues, reason = reflect(answer, contexts, llm)
    if grounded:
        # 一次通过：无需反射行动
        return answer, False, True, {"issues": issues, "reason": reason, "retries": 0}

    detail = {"issues": issues, "reason": reason, "retries": 0}
    cur_answer = answer
    cur_contexts = contexts
    reflected = True
    passed = False

    # —— 不通过 → 重试（重新检索 + 重写 + 再核查）——
    for attempt in range(1, max_retry + 1):
        # ① 重新检索：扩大召回，拿更多素材
        heal_k = max(int(top_k * REFLECT_TOPK_MULT), top_k)
        try:
            if retrieval == "hybrid":
                new_ret = hybrid_retrieve(question, emb, vec, pg, bm25, heal_k)
            else:
                new_ret = retrieve(question, emb, vec, pg, heal_k)
            new_ctx = [c for _, c in new_ret] if new_ret else cur_contexts
        except Exception:
            new_ctx = cur_contexts   # 检索失败就用原上下文，不中断

        # ② 用更严的"必须有据"指令重写
        prompt_h = build_prompt(question, new_ctx, max_chars=config.RAG_CONTEXT_MAX_CHARS)
        try:
            rewritten = llm.generate(_REWRITE_SYSTEM, prompt_h)
        except Exception as e:
            rewritten = ""           # 重写失败 → 后续降级
        if not rewritten:
            break                    # 写不出东西，进降级

        # ③ 再核查
        g2, iss2, r2 = reflect(rewritten, new_ctx, llm)
        detail["retries"] = attempt
        detail["issues"] = iss2
        detail["reason"] = r2
        cur_answer = rewritten
        cur_contexts = new_ctx
        if g2:
            passed = True
            break

    if passed:
        return cur_answer, reflected, True, detail

    # —— 兜底降级：仍不通过 → 安全拒答（绝不把可能编造的答案甩给用户）——
    safe = "根据提供的资料无法回答该问题（答案中部分内容缺乏依据，已降级处理）。"
    detail["degraded"] = True
    return safe, reflected, False, detail
