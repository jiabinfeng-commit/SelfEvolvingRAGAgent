# -*- coding: utf-8 -*-
"""
评估器（阶段 3：端到端评估闭环 = 自进化系统的「度量」）

为什么需要它？
- 阶段 1/2 我们证明了「能搜、能答」。但「答得对不对」一直没量化。
- 这一层用 LLM 当裁判（LLM-as-judge），对每道题自动打分，输出可对比的准确率。
- 这是整个 "Self-Evolving" 项目的地基：你改检索 / 改 prompt / 换模型，到底变好还是变坏？
  必须有一个稳定指标才能判断。eval_run 表把每次跑分存下来，下次优化直接比数字。

打分设计（judge prompt）：
- 可答题（single_hop / multi_hop）：拿 问题 + 标准答案(gold_answer) + 模型答案，
  让裁判判 0/1/2（错误 / 部分正确 / 正确），并给一句理由。
- 拒答题（unanswerable）：语料本就没有答案，正确行为是「拒答」。
  裁判判 2=正确拒答(明确说资料里答不了) / 1=含糊 / 0=编造了答案(幻觉，最严重)。

核心函数：
- grade_answer(question, generated, q, llm) -> (score, label, reason)
- run_evaluation(questions, top_k, llm, emb, vec, pg, limit) -> (results, summary)
  生成答案这一步和阶段 2 ask.py 用的是**同一套** retrieve + build_prompt + llm.generate，
  保证评估的就是「线上真实链路」，而不是另一个平行实现。

> 和阶段 2 的衔接：阶段 2 是「生成答案 + 落 qa_log」；阶段 3 是「批量生成答案 + 用裁判打分 + 落 eval 表」。
> 生成链路完全复用，阶段 3 只多出「裁判」和「评分聚合」两件事。
"""
import re
import time
from typing import List, Dict, Any, Tuple

from core import config
from core.prompt import build_prompt, SYSTEM_PROMPT
from core.retrieval import retrieve


# 裁判系统指令：定下「严格、基于证据、不编造」的基调
JUDGE_SYSTEM_PROMPT = (
    "你是一个严格的 RAG 答案质量评审员。你会拿到：用户问题、参考答案（标准答案）、"
    "以及待评审的模型回答。请只依据「待评审回答是否准确覆盖了参考答案的要点」来评判，"
    "不要因为表述不同就扣分（意思对即可），也不要因为参考答案没提到但可能正确的内容而加分。\n"
    "评审维度：\n"
    "1. 事实是否正确（有没有编造参考答案里没有的关键信息）。\n"
    "2. 是否覆盖了参考答案的核心要点。\n"
    "3. 是否答非所问或遗漏关键结论。\n"
)

# 可答题的打分标尺
ANSWER_RUBRIC = (
    "请按 0/1/2 打分并给出一句中文理由，严格只输出如下格式（不要多余内容）：\n"
    "分数: <0|1|2>\n理由: <一句话>\n"
    "0 = 错误：关键事实错、或编造了参考答案没有的信息、或答非所问。\n"
    "1 = 部分正确：方向对但遗漏要点，或表述含糊但不算错。\n"
    "2 = 正确：准确覆盖了参考答案核心要点，无编造。\n"
)

# 拒答题的打分标尺（重点是「有没有拒答」，而不是「答对」）
REFUSAL_RUBRIC = (
    "本题在知识库中应无法回答（语料中不存在答案）。请按 0/1/2 打分并给出一句中文理由，"
    "严格只输出如下格式（不要多余内容）：\n"
    "分数: <0|1|2>\n理由: <一句话>\n"
    "0 = 幻觉：模型给出了一个具体答案（编造），没有声明资料里答不了。\n"
    "1 = 含糊：模型模棱两可，既没明确拒答也没明确编造。\n"
    "2 = 正确拒答：模型明确说明根据提供的资料无法回答该问题。\n"
)


def build_judge_prompt(question: str, generated: str, q: Dict[str, Any]) -> str:
    """
    拼出给裁判的 user 文本。q 是评测集里的一题（含 gold_answer / type）。
    """
    gold = q.get("gold_answer") or "（无标准答案）"
    qtype = q.get("type", "single_hop")
    if qtype == "unanswerable":
        rubric = REFUSAL_RUBRIC
        head = "【题型】拒答题（资料中应无法回答）\n"
    else:
        rubric = ANSWER_RUBRIC
        head = "【题型】可答题\n"
    return (
        f"{head}"
        f"【问题】\n{question}\n\n"
        f"【参考答案】\n{gold}\n\n"
        f"【待评审回答】\n{generated}\n\n"
        f"{rubric}"
    )


def _parse_judge(raw: str) -> Tuple[int, str, str]:
    """
    解析裁判返回，提取 分数 / 理由。容错：抽不到分数就记 0 + 原因。
    裁判可能多嘴输出额外内容，这里只认第一行「分数: X」和「理由: Y」。
    """
    score = 0
    reason = raw.strip()
    m = re.search(r"分数\s*[:：]\s*([012])", raw)
    if m:
        score = int(m.group(1))
    rm = re.search(r"理由\s*[:：]\s*(.+)", raw, re.DOTALL)
    if rm:
        reason = rm.group(1).strip().split("\n")[0]   # 只取理由第一行
    label = "正确" if score == 2 else ("部分正确" if score == 1 else "错误")
    return score, label, reason


def grade_answer(question: str, generated: str, q: Dict[str, Any], llm) -> Tuple[int, str, str]:
    """
    用 LLM 当裁判给一道题打分。

    :param question: 原题
    :param generated: 我们 RAG 系统生成的答案
    :param q: 评测集该题（含 gold_answer / type）
    :param llm: 任意 BaseLLM 实例（阶段3 用 config 配的阿里云大模型）
    :return: (score 0/1/2, label 正确/部分正确/错误, reason)
    """
    user = build_judge_prompt(question, generated, q)
    try:
        raw = llm.generate(JUDGE_SYSTEM_PROMPT, user)
    except RuntimeError as e:
        # 裁判挂了（网络/key）不能让整轮评估崩，记 0 + 原因，继续下一题
        return 0, "错误", f"裁判调用失败: {e}"
    return _parse_judge(raw)


def run_evaluation(questions: List[Dict], top_k: int, llm, emb, vec, pg,
                    limit: int = None) -> Tuple[List[Dict], Dict]:
    """
    跑完整评估：对每题 召回 → 拼 prompt → 调 LLM 生成 → 裁判打分 → 收集指标。

    生成答案这步与阶段 2 ask.py **完全一致**（同一 retrieve + build_prompt + llm.generate），
    所以评估测的就是线上真实链路。

    :param questions: 评测集题目列表
    :param top_k: 召回块数
    :param llm: 生成答案 + 当裁判的大模型（复用同一个后端）
    :param emb/vec/pg: 检索三件套
    :param limit: 只跑前 N 题（自测用；None=全量）
    :return: (results 列表, summary 字典)
    """
    qs = questions[:limit] if limit else questions
    results = []
    for q in qs:
        question = q["question"]
        qtype = q.get("type", "single_hop")
        gold_doc = q.get("gold_doc")

        t0 = time.time()
        # 1) 召回（复用公共 retrieve，保证和线上同一口径）
        retrieved = retrieve(question, emb, vec, pg, top_k)
        contexts = [c for _, c in retrieved]
        # 2) 拼 prompt（和阶段2 完全一致）
        user_prompt = build_prompt(question, contexts, max_chars=config.RAG_CONTEXT_MAX_CHARS)
        # 3) 生成答案（和阶段2 完全一致；失败不崩，记原因）
        try:
            generated = llm.generate(SYSTEM_PROMPT, user_prompt)
        except RuntimeError as e:
            generated = f"（生成失败: {e}）"
        # 4) 召回命中（gold_doc 是否在 top1 / top3）—— 这是「检索质量」指标
        docs = [c.get("doc_name") for _, c in retrieved]
        recall_top1 = bool(gold_doc and gold_doc in docs[:1])
        recall_top3 = bool(gold_doc and gold_doc in docs[:3])
        # 5) 裁判打分（答案质量指标）
        score, label, reason = grade_answer(question, generated, q, llm)
        latency = time.time() - t0

        results.append({
            "qid": q.get("id"),
            "question": question,
            "qtype": qtype,
            "gold_doc": gold_doc,
            "generated": generated,
            "judge_score": score,
            "judge_label": label,
            "judge_reason": reason,
            "recall_top1": recall_top1,
            "recall_top3": recall_top3,
            "latency_s": round(latency, 2),
        })

    # ---- 汇总 ----
    n = len(results)
    correct = sum(1 for r in results if r["judge_score"] == 2)
    partial = sum(1 for r in results if r["judge_score"] == 1)
    # 拒答正确率：unanswerable 题里，裁判判 2 的比例
    unans = [r for r in results if r["qtype"] == "unanswerable"]
    refused_ok = sum(1 for r in unans if r["judge_score"] == 2)
    r1 = sum(1 for r in results if r["recall_top1"])
    r3 = sum(1 for r in results if r["recall_top3"])
    avg_lat = sum(r["latency_s"] for r in results) / n if n else 0
    summary = {
        "n": n,
        "correct": correct,
        "partial": partial,
        "wrong": n - correct - partial,
        "accuracy": round(correct / n, 3) if n else 0,          # 正确率(裁判=2)
        "partial_rate": round(partial / n, 3) if n else 0,
        "refusal_rate": round(refused_ok / len(unans), 3) if unans else None,  # 拒答正确率
        "recall_top1": round(r1 / n, 3) if n else 0,
        "recall_top3": round(r3 / n, 3) if n else 0,
        "avg_latency_s": round(avg_lat, 2),
    }
    return results, summary
