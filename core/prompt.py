# -*- coding: utf-8 -*-
"""
Prompt 拼装（阶段 2：把"检索到的上下文 + 问题"拼成发给 LLM 的话）

为什么单独成一个模块？
- 拼 prompt 是 RAG 里最容易"写歪"的地方：指令不清晰，模型就会瞎编（幻觉）。
- 把拼装逻辑和"调模型 / 查数据库"解耦，方便后续调指令、做 few-shot、换语言，
  而不用动 ask.py 的主流程。

设计要点：
1. system 指令明确两点：只依据【上下文】作答；上下文没有就老实说"不知道"，不要编造。
2. 上下文按相关度从高到低编号 [1][2]... 拼进去，并在每块前标注它来自哪篇文档
   （doc_name + heading），方便模型引用、也方便事后溯源。
3. 每块截断到 RAG_CONTEXT_MAX_CHARS，避免极长块把 prompt 撑爆、还顺带稀释了重点。
4. 纯函数：不碰数据库、不碰网络，单测即可验证。
"""
from typing import List, Dict


# system 指令：定下" grounded（有据可依）"的基调，是抗幻觉的第一道防线
SYSTEM_PROMPT = (
    "你是一个严谨的技术文档问答助手。请只根据下面提供的【上下文】回答用户问题。\n"
    "规则：\n"
    "1. 如果【上下文】里找不到答案，请明确回答「根据提供的资料无法回答该问题」，不要编造。\n"
    "2. 回答尽量简洁、准确，可引用上下文中的要点；不要添加上下文之外的知识。\n"
    "3. 如上下文相关，可在回答中提及信息来源（如文档章节名）。"
)


def build_prompt(question: str, contexts: List[Dict], max_chars: int = 1200) -> str:
    """
    拼出发给 LLM 的 user 文本（system 另算，见 SYSTEM_PROMPT）。

    :param question:  用户原始问题
    :param contexts:  检索回表的 chunk 列表，每项至少含
                      content(正文)、doc_name(文档名)、heading(标题路径)
                      建议按相关度从高到低传入
    :param max_chars: 单块上下文最大字符数（截断用）
    :return:          拼好的 user 文本
    """
    if not contexts:
        # 没有任何上下文：直接把问题抛给模型，但 system 已要求"无据不说"
        return f"【上下文】\n（无相关材料）\n\n【问题】\n{question}"

    blocks = []
    for i, c in enumerate(contexts, 1):
        content = (c.get("content") or "").strip()
        if not content:
            continue
        # 超长截断：保留前 max_chars 个字符，避免单块撑爆 prompt
        if len(content) > max_chars:
            content = content[:max_chars] + "…（已截断）"
        # 来源标注：文档名 + 标题路径（标题可能为空）
        src = c.get("doc_name", "?")
        heading = (c.get("heading") or "").strip()
        src_label = f"{src} / {heading}" if heading else src
        blocks.append(f"[资料 {i}] 来源：{src_label}\n{content}")

    context_text = "\n\n".join(blocks)
    return f"【上下文】\n{context_text}\n\n【问题】\n{question}"
