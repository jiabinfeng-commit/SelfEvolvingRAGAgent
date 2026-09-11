# 阶段 8：Reflexion 自检 + 可观测链路追踪

## 一句话

阶段 5/6 解决"答不出/答不全"，阶段 8 解决"**答了但可能编了一句**"（幻觉）。
靠两件事：① 答案生成后插一道事实核查闭环（Reflexion）；② 把每次问答的链路信息落库（可观测）。

## ① Reflexion 事实核查闭环（core/reflexion.py）

### 流程

```
生成答案
  → reflect(answer, contexts, llm) 让 LLM 当考官逐条核对"每句断言能否在参考资料找到依据"
      ├─ 全有依据 → 通过，直接返回
      └─ 有依据不足的断言 → 不通过：
            ① 重新检索（扩大召回 top_k × 2，拿更多素材）
            ② 用更严的"必须有据"指令重写答案（_REWRITE_SYSTEM）
            ③ 再 reflect 一次
                ├─ 通过 → 返回重写版
                └─ 仍不通过（最多 REFLECT_MAX_RETRY=1 次）→ 降级为安全拒答
                   "根据提供的资料无法回答该问题（答案中部分内容缺乏依据，已降级处理）"
```

### 和阶段 5 的区别

- 阶段 5：模型**拒答**了才救（判定误拒 → 换宽松指令重试）。
- 阶段 8：模型**答了但可能编造**也要救（判定无依据 → 重检索/重写，否则降级）。
- 两者互补：一个管"不敢答"，一个管"答不实"。

### 为什么比"单靠 prompt 约束"强

阶段 2 的 `SYSTEM_PROMPT` 已经要求"只依据上下文、不编造"，但那是靠模型自觉，不可控。
Reflexion 把"是否幻觉"变成**可程序化判断 + 可重试**的闭环：不通过就真的去多搜、去重写，
直到通过或降级。抗幻觉从"希望模型听话"升级成"系统级保障"。

### 安全失败

- 核查 LLM 调用挂了 → 保守判"不通过" → 走重试/降级，绝不把可能编造的答案放出去。
- 降级是**最终兜底**：宁可答"资料里答不了"，也不甩带编造的答案。

### API

```python
reflect(answer, contexts, llm) -> (grounded: bool, issues: List[str], reason: str)
run_reflexion(question, answer, retrieved, emb, vec, pg, llm, bm25=None,
              top_k=None, retrieval="vector", max_retry=1)
    -> (final_answer, reflected: bool, reflect_pass: bool, detail: dict)
```

集成进 `generate_answer(..., reflect=True)`（默认关，向后兼容）。返回结构新增字段：
`reflected / reflect_pass / reflect_detail / request_id`。

## ② 链路追踪（core/tracing.py + PG trace_log）

### 记录什么

每次问答（开启 `trace=True`）写一条 `trace_log`：

| 字段 | 含义 |
|---|---|
| request_id | 一次会话一个 uuid，串联同次请求 |
| recalled_ids / recalled_scores | 召回了哪些块 + 分数（定位检索质量） |
| prompt | 实际发给 LLM 的 user prompt（定位提示词问题） |
| answer | 最终答案 |
| latency_s | 耗时 |
| est_tokens | 估算 token（字符数/4 启发式，标注为估算非精确） |
| reflected / reflect_pass | 是否触发 Reflexion、最终是否通过事实核查 |

### token 估算（诚实说明）

真实 token 数要按模型 tokenizer 算（会引入依赖、且各模型不同）。这里用通用启发式
`估算 token ≈ 字符数 / 4`（中英文混排的经验上界）。代码明确标注"估算"，不冒充精确值；
将来接了真实 tokenizer 再替换即可。

### API

```python
estimate_tokens(text) -> int
TraceRecorder(pg).record(question, retrieval, backend,
                         recalled_ids, recalled_scores, prompt, answer,
                         latency_s, reflected, reflect_pass, request_id=None) -> str
```

集成进 `generate_answer(..., trace=True)`。追踪是**旁路**：写库失败只打印告警，绝不影响主回答。

## 运行入口

```bash
# 阶段8 反射 + 追踪（单题）
python scripts/ask.py "FastAPI 怎么做依赖注入？" --reflect --trace

# 看板：取最近 trace
python -c "from core.storage.pg_store import PGStore; \
pg=PGStore(); [print(t['question'], t['reflect_pass'], t['est_tokens']) for t in pg.get_traces(20)]"
```

## 指标 / 产出数据

- **幻觉率抽检**：用 `scripts/evaluate.py` 跑 50 题，对 `reflect_pass=False`（触发降级）的题
  人工判定是否真为幻觉，得出"反射拦截的幻觉占多少"。这是阶段 8 的硬指标。
- **trace_log 趋势**：按天聚合 `reflected` 比例、平均 `est_tokens`、平均 `latency_s`，
  看"哪些问题经常触发反射""成本花在哪"。

## 自测

```bash
python scripts/selftest_stage8.py   # 桩模式：反射三种路径 + 集成含 trace 落库
```

## 已知限制

- 事实核查本身也是 LLM 判断，可能误判（把有依据的判无依据 → 多一次重写；或反之）。
- 重写/再核查会多烧 1~2 次 LLM（成本考量，默认 `REFLECT_MAX_RETRY=1`）。
- token 为估算值；精确值需接真实 tokenizer。
- 只在 `reflect=True / trace=True` 时生效，默认关（避免无谓开销，按需开启）。
