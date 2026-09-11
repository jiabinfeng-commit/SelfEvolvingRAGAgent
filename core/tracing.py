# -*- coding: utf-8 -*-
"""
core/tracing.py —— 阶段 8（下半）：可观测链路追踪

====================================================================
为什么需要"可观测"？
====================================================================
前面阶段把"答得对不对"用 eval 量化了，但线上每次回答"内部到底发生了什么"
还是黑盒：召回了哪些块？prompt 长什么样？耗时多少？这次有没有触发 Reflexion、
最终过没过事实核查？

阶段 8 把每一次问答的链路信息落进 PG 的 trace_log 表，形成可复盘的数据底座：
    - 用户问 X → 召回了 [块A,块B]（分数 0.8/0.6）→ 发的 prompt → 答案 →
      耗时 1.2s → 估算 token 320 → 触发 Reflexion 且通过
事后就能回答："为什么这道题答成这样？"、"哪类问题经常触发 Reflexion？"、
"token 成本主要花在哪？"——这是把 RAG 从 demo 变成可运维系统的关键一环。

====================================================================
token 怎么算（诚实说明）
====================================================================
真正的 token 数要按模型 tokenizer 算（如 tiktoken），那会引入依赖且各模型不同。
这里用通用启发式：估算 token ≈ 字符数 / 4（中英文混排下的经验值，偏保守）。
代码里标注清楚这是"估算"，不冒充精确值；若以后接了真实 tokenizer 再替换。

====================================================================
对外暴露
====================================================================
- estimate_tokens(text) -> int
- TraceRecorder 类：record(...) 把一条 trace 写入 PG（底层 pg.save_trace）
"""
import uuid
from typing import List, Dict, Any, Optional


def estimate_tokens(text: str) -> int:
    """
    估算 token 数（启发式：字符数 / 4）。

    中文一个字常对应 1~2 个 token，英文一个词约 1 个 token；
    中英文混排时"字符数/4"是个粗略但够用的上界估计。明确标注为估算值。
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


class TraceRecorder:
    """
    链路追踪记录器：把一次问答的链路信息写入 PG 的 trace_log。

    用法：在问答入口建一个（或直接用 pg），每次回答后 record 一条。
    request_id 由调用方传入（建议一次 HTTP 请求一个 uuid），便于串联同一次会话。
    """

    def __init__(self, pg):
        self.pg = pg

    def record(self, question: str, retrieval: str, backend: str,
               recalled_ids: List[str], recalled_scores: List[float],
               prompt: str, answer: str, latency_s: float,
               reflected: bool, reflect_pass: Optional[bool],
               request_id: str = None) -> str:
        """
        写一条 trace。返回 request_id（便于调用方回填到返回结构里）。

        :param request_id: 不传则新生成一个 uuid
        :param reflected:  本次是否触发了阶段 8 Reflexion
        :param reflect_pass: Reflexion 最终是否通过事实核查（未触发则为 None）
        """
        rid = request_id or uuid.uuid4().hex
        est = estimate_tokens(prompt) + estimate_tokens(answer)
        try:
            self.pg.save_trace(
                request_id=rid,
                question=question,
                retrieval=retrieval,
                backend=backend,
                recalled_ids=recalled_ids,
                recalled_scores=recalled_scores,
                prompt=prompt,
                answer=answer,
                latency_s=latency_s,
                est_tokens=est,
                reflected=reflected,
                reflect_pass=reflect_pass,
            )
        except Exception as e:
            # 追踪写库失败绝不能影响主回答流程（可观测是旁路）
            print(f"[trace] 写入失败（已忽略）: {e}")
        return rid
